import torch
import triton
import triton.language as tl
from typing import Tuple

# ✅ 这个 kernel 只有在 N <= BLOCK_SIZE 时才正确！
# ❌ 如果 N > BLOCK_SIZE（比如 N=100,000，BLOCK_SIZE=256），它会 只处理前 256 个元素，忽略其余 99,744 个！
@triton.jit
def layer_norm_fwd_naive(
    input_ptr, output_ptr, gamma_ptr, beta_ptr, eps,
    mean_ptr, rsqrt_var_ptr, # 这两个是为了后续的反向传播准备的，前向传播先算出来，存起来
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    row_start = input_ptr + pid * N
    row_ptrs = row_start + tl.arange(0, BLOCK_SIZE)
    row_mask = tl.arange(0, BLOCK_SIZE) < N
    x = tl.load(row_ptrs, mask=row_mask, other=0.0)

    gamma_ptrs = gamma_ptr + tl.arange(0, BLOCK_SIZE)
    beta_ptrs = beta_ptr + tl.arange(0, BLOCK_SIZE)
    gamma = tl.load(gamma_ptrs, mask=row_mask, other=0.0)
    beta = tl.load(beta_ptrs, mask=row_mask, other=0.0)

    mean = tl.sum(x, axis=0) / N

    # var = tl.sum((x - mean) ** 2, axis=0) / N  这是错误写法 ❌
    # → 会把 padding 的 0 也当成真实数据去减均值，导致：
    # 方差偏大（因为多了几个 -mean 的平方项）
    x_minus_mean = tl.where(row_mask, x - mean, 0.)
    var = tl.sum(x_minus_mean * x_minus_mean, axis=0) / N
    rsqrt_var = tl.rsqrt(var + eps)

    # 算出前向传播的真正结果
    y = (x - mean) * rsqrt_var * gamma + beta

    # 这一步是用于给反向传播的：
    tl.store(mean_ptr + pid, mean)
    tl.store(rsqrt_var_ptr + pid, rsqrt_var)

    output_ptrs = output_ptr + pid * N + tl.arange(0, BLOCK_SIZE)
    tl.store(output_ptrs, y, mask=row_mask)

# 还是一个block处理一整行，不过考虑的是N>NUM_THREADS的情况，那么就需要使用for循环分批处理
@triton.jit
def layer_norm_fwd(
    input_ptr, output_ptr, gamma_ptr, beta_ptr, eps,
    mean_ptr, rsqrt_var_ptr, 
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    mean = 0.0
    var = 0.0
    for col_start in tl.range(0,N,BLOCK_SIZE):
        col_ptrs = input_ptr + pid*N +col_start + tl.arange(0, BLOCK_SIZE)#我们要把地址块计算出来，用于去load global memory里面的数据
        col_mask = (col_start + tl.arange(0, BLOCK_SIZE)) < N
        x = tl.load(col_ptrs, mask=col_mask, other=0.0)
        mean += tl.sum(x, axis=0)
    mean /= N

    for col_start in tl.range(0,N,BLOCK_SIZE):
        col_ptrs = input_ptr + pid*N +col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = (col_start + tl.arange(0, BLOCK_SIZE)) < N
        x = tl.load(col_ptrs, mask=col_mask, other=0.0)
        x_minus_mean = tl.where(col_mask, x - mean, 0.)
        var += tl.sum(x_minus_mean * x_minus_mean, axis=0)
    var /= N
    rsqrt_var = tl.rsqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(rsqrt_var_ptr + pid, rsqrt_var)
    
    for col_start in tl.range(0,N,BLOCK_SIZE):
        col_ptrs = input_ptr + pid*N +col_start + tl.arange(0, BLOCK_SIZE)
        col_mask = (col_start + tl.arange(0, BLOCK_SIZE)) < N
        x = tl.load(col_ptrs, mask=col_mask, other=0.0)

        gamma_ptrs = gamma_ptr + col_start + tl.arange(0, BLOCK_SIZE)
        beta_ptrs = beta_ptr + col_start + tl.arange(0, BLOCK_SIZE)
        gamma = tl.load(gamma_ptrs, mask=col_mask, other=0.0)
        beta = tl.load(beta_ptrs, mask=col_mask, other=0.0)

        y = (x - mean) * rsqrt_var * gamma + beta

        output_ptrs = output_ptr + pid*N + col_start + tl.arange(0, BLOCK_SIZE)
        tl.store(output_ptrs, y, mask=col_mask)
    

# ============================================================
# Wrapper functions for your two kernels
# ============================================================

def layer_norm_fwd_naive_wrapper(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """
    Only works correctly when N <= BLOCK_SIZE (e.g., N=256, BLOCK_SIZE=256).
    For larger N, it silently ignores tail elements → DO NOT use in general.
    """
    M, N = x.shape
    assert gamma.shape == (N,)
    assert beta.shape == (N,)

    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rsqrt_var = torch.empty((M,), device=x.device, dtype=torch.float32)

    # Choose BLOCK_SIZE >= N (but this is fragile!)
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE < N:
        raise ValueError("naive kernel requires BLOCK_SIZE >= N")

    grid = lambda meta: (M,)
    layer_norm_fwd_naive[grid](
        x, y, gamma, beta, eps,
        mean, rsqrt_var,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y, mean, rsqrt_var


def layer_norm_fwd_wrapper(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """
    General version that handles any N via loop tiling.
    """
    M, N = x.shape
    assert gamma.shape == (N,)
    assert beta.shape == (N,)

    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rsqrt_var = torch.empty((M,), device=x.device, dtype=torch.float32)

    # Use a reasonable block size (e.g., 256 or 512)
    BLOCK_SIZE = 256  # can be tuned

    grid = lambda meta: (M,)
    layer_norm_fwd[grid](
        x, y, gamma, beta, eps,
        mean, rsqrt_var,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )
    return y, mean, rsqrt_var


def layer_norm_torch(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5):
    """
    PyTorch reference using built-in LayerNorm (note: LayerNorm applies over last dim by default).
    We mimic the same behavior: normalize over dim=-1.
    """
    # Note: torch.nn.functional.layer_norm expects normalized_shape = (N,)
    y = torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[-1],), weight=gamma, bias=beta, eps=eps)
    # Compute mean and rsqrt(var) manually for comparison
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    rsqrt_var = torch.rsqrt(var + eps).squeeze(-1)
    mean = mean.squeeze(-1)
    return y, mean, rsqrt_var


# ============================================================
# Correctness Test
# ============================================================
def test_correctness():
    torch.manual_seed(0)
    device = 'cuda'
    eps = 1e-5

    # Test cases: small (N <= 256) and large (N > 256)
    test_cases = [
        (8, 128),    # small
        (4, 256),    # boundary
        (2, 1000),   # large
        (16, 4096),  # very large
    ]

    for M, N in test_cases:
        print(f"Testing M={M}, N={N}...")

        x = torch.randn(M, N, device=device, dtype=torch.float32)
        gamma = torch.randn(N, device=device, dtype=torch.float32)
        beta = torch.randn(N, device=device, dtype=torch.float32)

        # Reference
        y_ref, mean_ref, rsqrt_var_ref = layer_norm_torch(x, gamma, beta, eps)

        # Naive kernel: only valid if N <= 256 (or chosen BLOCK_SIZE)
        if N <= 256:
            try:
                y_naive, mean_naive, rsqrt_var_naive = layer_norm_fwd_naive_wrapper(x, gamma, beta, eps)
                torch.testing.assert_close(y_naive, y_ref, atol=1e-4, rtol=1e-4)
                torch.testing.assert_close(mean_naive, mean_ref, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(rsqrt_var_naive, rsqrt_var_ref, atol=1e-5, rtol=1e-5)
                print("  ✅ naive kernel passed")
            except Exception as e:
                print(f"  ❌ naive kernel failed: {e}")
        else:
            print("  ⚠️ skipping naive kernel (N too large)")

        # General kernel
        y_gen, mean_gen, rsqrt_var_gen = layer_norm_fwd_wrapper(x, gamma, beta, eps)
        torch.testing.assert_close(y_gen, y_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(mean_gen, mean_ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rsqrt_var_gen, rsqrt_var_ref, atol=1e-5, rtol=1e-5)
        print("  ✅ general kernel passed")


# ============================================================
# Benchmark
# ============================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[256 * i for i in range(1, 17)],  # 256 to 4096
        x_log=True,
        line_arg='provider',
        line_vals=['triton_general', 'torch'],
        line_names=['Triton General', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-forward',
        args={'M': 1024},  # fix batch size
    )
)
def benchmark(N, M, provider):
    device = 'cuda'
    eps = 1e-5

    x = torch.randn((M, N), device=device, dtype=torch.float32)
    gamma = torch.randn(N, device=device, dtype=torch.float32)
    beta = torch.randn(N, device=device, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'triton_general':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer_norm_fwd_wrapper(x, gamma, beta, eps),
            quantiles=quantiles
        )
    elif provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: layer_norm_torch(x, gamma, beta, eps),
            quantiles=quantiles
        )

    # Compute effective memory bandwidth (GB/s)
    # Total bytes: read x (4*M*N), gamma/beta (4*2*N), write y (4*M*N)
    # Plus mean/rsqrt_var (8*M) — negligible
    total_bytes = (4 * M * N) + (8 * N) + (4 * M * N)
    gb_per_s = lambda ms: total_bytes / (ms * 1e-3) / 1e9
    return gb_per_s(ms), gb_per_s(max_ms), gb_per_s(min_ms)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Running correctness tests...")
    test_correctness()

    print("\nRunning benchmark...")
    benchmark.run(show_plots=True, print_data=True)