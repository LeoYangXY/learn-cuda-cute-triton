import torch
import triton
import triton.language as tl
DEVICE = torch.device("cuda:0")

def auto_tune_and_benchmark(
    triton_func,
    torch_func,
    sizes,
    block_sizes=None,
    num_warmup_tune=5,
    num_iter_tune=20,
    num_warmup_bench=10,
    num_iter_bench=100,
    dtype=torch.float32,
    device=DEVICE,
    correctness_size=98432,  # 用于正确性验证的固定 size
):
    """
    1. 先做正确性检查
    2. 对每个 size 自动调优 BLOCK_SIZE 并 benchmark vs PyTorch
    """
    # ======================
    # 正确性检查
    # ======================
    print("🔍 Running correctness check...")
    x_test = torch.randn(correctness_size, device=device, dtype=dtype)
    y_test = torch.randn(correctness_size, device=device, dtype=dtype)
    triton_out = triton_func(x_test, y_test)
    torch_out = torch_func(x_test, y_test)
    if not torch.allclose(triton_out, torch_out, atol=1e-6):
        max_diff = torch.max(torch.abs(triton_out - torch_out)).item()
        raise RuntimeError(f"❌ Correctness check failed! Max diff: {max_diff}")
    print("✅ Correctness verified.\n")

    # ======================
    # 自动调优 + Benchmark
    # ======================
    if block_sizes is None:
        block_sizes = [64, 128, 256, 512, 1024]

    print(f"{'Size':>10} | {'Best BS':>8} | {'Triton (ms)':>12} | {'Torch (ms)':>12} | {'Speedup':>8}")
    print("-" * 70)

    for size in sizes:
        x = torch.randn(size, device=device, dtype=dtype)
        y = torch.randn(size, device=device, dtype=dtype)

        # --- Step 1: Tune best BLOCK_SIZE ---
        best_bs = block_sizes[0]
        best_time = float('inf')

        for bs in block_sizes:
            if bs > 1024 or bs % 32 != 0:
                continue

            # Warmup
            for _ in range(num_warmup_tune):
                _ = triton_func(x, y, BLOCK_SIZE=bs)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_iter_tune):
                _ = triton_func(x, y, BLOCK_SIZE=bs)
            end.record()
            torch.cuda.synchronize()
            avg_ms = start.elapsed_time(end) / num_iter_tune

            if avg_ms < best_time:
                best_time = avg_ms
                best_bs = bs

        # --- Step 2: Final benchmark with best BS ---
        for _ in range(num_warmup_bench):
            _ = triton_func(x, y, BLOCK_SIZE=best_bs)
            _ = torch_func(x, y)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iter_bench):
            _ = triton_func(x, y, BLOCK_SIZE=best_bs)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / num_iter_bench

        start.record()
        for _ in range(num_iter_bench):
            _ = torch_func(x, y)
        end.record()
        torch.cuda.synchronize()
        torch_ms = start.elapsed_time(end) / num_iter_bench

        speedup = torch_ms / triton_ms if triton_ms > 0 else float('inf')
        print(f"{size:>10} | {best_bs:>8} | {triton_ms:>12.3f} | {torch_ms:>12.3f} | {speedup:>8.2f}x")
