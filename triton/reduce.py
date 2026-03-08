import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_1_block(x_ptr,      # 输入矩阵指针
    output_ptr, # 输出向量指针
    M, N,       # 矩阵的行数和列数
    stride_xm, stride_xn, # 输入矩阵的 stride
    BLOCK_SIZE_N: tl.constexpr # 列维度的 Block 大小，必须是 2 的幂
):
    pid = tl.program_id(0) 
    row_start = x_ptr + pid * stride_xm # 每个 block 负责一行
    col_offsets = tl.arange(0, BLOCK_SIZE_N) 
    mask = col_offsets < N
    ptrs = row_start + col_offsets

    x_tensor = tl.load(ptrs, mask=mask, other=0.0)

    s = tl.sum(x_tensor, axis=0)

    out_row_start = output_ptr + pid
    tl.store(out_row_start, s) 



"""
处理“长行”（N 很大）的两种并行策略：权衡与适用场景

在 GPU 编程中，当单行长度 N 超出单个 Block 的处理能力时，
有两种主流策略：单 Block 循环处理 vs. 多 Block 协作处理。
选择哪种取决于操作语义、数据规模和硬件效率。

──────────────────────────────────────────────
✅ 策略一：单 Block + 循环（每行一个 Block）
──────────────────────────────────────────────
- 核心思想：每个 Block 负责一整行，通过循环分块加载/处理数据。
- 典型应用：LayerNorm、Row-wise Softmax 等 per-row 独立操作。

✔️ 优势：
  • 实现简洁：无需跨 Block 同步或通信，逻辑清晰。
  • 内存访问高效：顺序遍历全局内存，利于缓存和带宽利用。
  • 语义匹配：天然契合“每行独立”的计算模式。
  • 官方推荐：Triton/CUDA 教程中对 row-wise 操作的标准范式。

❌ 局限：
  • 并行度受限：每行仅使用 1 个 Block（最多 1024 线程），
    无法充分利用 GPU 上万级线程的并发能力。
  • 长尾延迟：当 N 极大（如 >10⁷）时，循环次数过多，
    导致单个 Block 执行时间变长，可能成为吞吐瓶颈。
  • 不适合重计算场景：若后续有复杂算子（如 attention），
    单 Block 可能无法提供足够计算强度掩盖访存延迟。

📌 推荐场景：
  • N 中等规模（几千 ~ 几十万）
  • 操作为 per-row 独立（如 LayerNorm、RMSNorm、行归一化等）

──────────────────────────────────────────────
✅ 策略二：多 Block 协作（多个 Block 共同处理一行）
──────────────────────────────────────────────
- 核心思想：将一行划分为多个段，每个 Block 处理一段，
  最终通过归约（reduction）合并结果。
- 典型应用：全局统计量计算（如 L2 norm）、超长序列处理（如 LLM 中 N=65536）

✔️ 优势：
  • 高并行度：一行可调度数百至数千个 Blocks，充分压满 GPU 计算单元。
  • 适合超长序列：有效缓解单 Block 循环过深的问题。
  • 可与高级优化结合：如 FlashAttention 中的 block-wise 分块与在线归约。

❌ 挑战：
  • 必须跨 Block 归约：需额外写 partial results 到全局内存，
    并启动第二阶段 kernel（或使用原子操作），引入同步开销。
  • 内存访问模式敏感：若分块不连续，可能降低带宽效率。
  • 对轻量操作不划算：例如 LayerNorm 中 mean/var 仅为标量，
    归约开销可能超过并行收益，反而降低性能。

📌 推荐场景：
  • 需要计算整行的全局标量（如 sum、L2 norm、max 等）
  • N 极大（>1M），单 Block 循环成为瓶颈
  • 后续有 heavy compute，值得用更多资源加速（如自定义 kernel）

──────────────────────────────────────────────
💡 总结建议：
对于 LayerNorm 等 per-row 独立操作，**优先使用策略一（单 Block + 循环）** ——
它简单、高效、且与硬件访存模式高度契合。
只有在真正需要全局归约或 N 极端巨大时，才考虑策略二。
"""
# 现在问题升级了：假设我们有一个超长的向量 X（长度 N 可能是几百万），我们要计算它的总和。
# 挑战数据量太大：一个 Block 处理不完，必须并行启动多个 Block。
# 跨 Block 通信：Block 之间是不共享 Shared Memory 的。如果不使用复杂的同步机制，Block A 无法直接把数据传给 Block B。
@triton.jit
def reduce_sum_multi_blocks(x_ptr,      # 输入矩阵指针
    output_ptr, # 输出向量指针
    M, N,       # 矩阵的行数和列数
    stride_xm, stride_xn, # 输入矩阵的 stride
    BLOCK_SIZE_N: tl.constexpr
):
    block_num_per_row = tl.cdiv(N, BLOCK_SIZE_N) # 每行需要多少个 Block 来处理
    pid = tl.program_id(0)
    row_id = pid // block_num_per_row # 当前 Block 负责的行号
    col_start = (pid % block_num_per_row) * BLOCK_SIZE_N # 当前 Block 负责的列起始位置
    ptrs = x_ptr + row_id * stride_xm + col_start * stride_xn + tl.arange(0, BLOCK_SIZE_N) * stride_xn
    mask = (col_start + tl.arange(0, BLOCK_SIZE_N)) < N

    x = tl.load(ptrs, mask=mask, other=0.0)#每个block把自己负责的那一块数据load进来
    block_sum = tl.sum(x, axis=0) 

    output_addr = output_ptr + row_id
    tl.atomic_add(output_addr, block_sum) #使用原子操作进行跨block的信息同步，其实就是往global_memory里面累加



#原子操作可能成为瓶颈，下面介绍多级规约的方式来进行跨box的通信：
# 第一阶段：
# 每个 Block 把自己的 block_sum 写入一个 中间的global memory buffer（partial_sums[pid]）
# 第二阶段：
# 对每行的 partial sums 再做一次归约（通常用一个 Block 即可）
# 这样完全避免原子操作，且第二阶段数据量极小（每行只有 block_num_per_row 个元素），因此一个block可以处理完毕
@triton.jit
def reduce_sum_partial(
    x_ptr,
    partial_sums_ptr,  # 输出：每个 block 一个值
    M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE_N: tl.constexpr
):
    block_num_per_row = tl.cdiv(N, BLOCK_SIZE_N)
    pid = tl.program_id(0)
    row_id = pid // block_num_per_row
    col_start = (pid % block_num_per_row) * BLOCK_SIZE_N

    offsets = tl.arange(0, BLOCK_SIZE_N)
    cols = col_start + offsets
    mask = cols < N

    ptrs = x_ptr + row_id * stride_xm + cols * stride_xn
    x = tl.load(ptrs, mask=mask, other=0.0)
    block_sum = tl.sum(x)

    # 每个 block 写自己的 partial sum（无并发冲突，不需要用原子操作去写）
    tl.store(partial_sums_ptr + pid, block_sum)

@triton.jit
def reduce_sum_final(
    partial_sums_ptr,
    output_ptr,
    M,
    blocks_per_row: tl.constexpr,  # 上一轮用到的，其值=ceil(N / BLOCK_SIZE_N)
    BLOCK_SIZE: tl.constexpr = 1024
):
    pid = tl.program_id(0)
    #每个block需要负责搞出一整行的结果，现在它会拿到的是上一轮的多个block协作算出来的结果，也就是会拿到blocks_per_row个值
    start = pid * blocks_per_row
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < blocks_per_row
    ptrs = partial_sums_ptr + start + offsets

    x = tl.load(ptrs, mask=mask, other=0.0)
    row_sum = tl.sum(x, axis=0)

    out_addr = output_ptr + pid
    tl.store(out_addr, row_sum)

# 总结：reduce类算子的流程：
# 每个thread自己把自己负责的元素做reduce，然后传入block_reduce_sum函数
# block_reduce_sum函数就是一个block，接收的是每个thread传入的那个值，使用warp_shuffle操作先让每个warp都拿到值，然后写入shared_memory，然后再是开一个warp去把shared_memory中的东西拿出来，继续做一遍warp shuffle
# 上面的都是triton的那个tl.sum函数已经在编译器层面做好了
# 然后如果我们的一行很长的话，就可以有2种选择：
# 1.一个block做多次for循环处理一整行
# 2.多个block协作，那么不同block的数据交换又是有2种选择：要么直接对于global memory用atomic操作，要么是用多阶段reduce，不过也是需要写回global memory的



import triton.testing

# ==============================
# Helper functions
# ==============================

def _next_power_of_2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def get_block_size_n(N, max_block_size=2048):
    bs = _next_power_of_2(N)
    if bs > max_block_size:
        bs = max_block_size
    return bs

# ==============================
# Host wrappers for your kernels
# ==============================

def triton_reduce_sum_1block(x, BLOCK_SIZE_N=None):
    M, N = x.shape
    if BLOCK_SIZE_N is None:
        BLOCK_SIZE_N = get_block_size_n(N, max_block_size=8192)
    if N > BLOCK_SIZE_N:
        raise ValueError(f"N={N} > BLOCK_SIZE_N={BLOCK_SIZE_N}, use multi-block version")
    output = torch.empty(M, device=x.device, dtype=x.dtype)
    grid = (M,)
    reduce_sum_1_block[grid](
        x, output, M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return output

def triton_reduce_sum_atomic(x, BLOCK_SIZE_N=None):
    M, N = x.shape
    if BLOCK_SIZE_N is None:
        BLOCK_SIZE_N = get_block_size_n(N, max_block_size=1024)
    blocks_per_row = triton.cdiv(N, BLOCK_SIZE_N)
    total_blocks = M * blocks_per_row
    output = torch.zeros(M, device=x.device, dtype=x.dtype)  # ⚠️ 必须初始化为 0！
    grid = (total_blocks,)
    reduce_sum_multi_blocks[grid](
        x, output, M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return output

def triton_reduce_sum_two_stage(x, BLOCK_SIZE_N=None):
    M, N = x.shape
    if BLOCK_SIZE_N is None:
        BLOCK_SIZE_N = get_block_size_n(N, max_block_size=2048)
    blocks_per_row = triton.cdiv(N, BLOCK_SIZE_N)
    total_blocks = M * blocks_per_row

    # Stage 1: partial sums
    partial_sums = torch.empty(total_blocks, device=x.device, dtype=x.dtype)
    grid1 = (total_blocks,)
    reduce_sum_partial[grid1](
        x, partial_sums, M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    # Stage 2: final reduction per row
    output = torch.empty(M, device=x.device, dtype=x.dtype)
    grid2 = (M,)
    reduce_sum_final[grid2](
        partial_sums, output, M,
        blocks_per_row=blocks_per_row,
        BLOCK_SIZE=1024
    )
    return output


def test_correctness():
    torch.manual_seed(0)
    device = "cuda"
    for M in [1, 32, 1024]:
        for N in [100, 1024, 8192, 65536, 1000000]:
            print(f"Testing M={M}, N={N}")
            x = torch.randn(M, N, device=device, dtype=torch.float32)
            ref = torch.sum(x, dim=-1)

            # Test 1-block (only if N small)
            if N <= 2048:
                out1 = triton_reduce_sum_1block(x)
                assert torch.allclose(ref, out1, atol=1e-4), f"1-block failed at N={N}"

            # Test atomic
            out2 = triton_reduce_sum_atomic(x, BLOCK_SIZE_N=512)
            assert torch.allclose(ref, out2, atol=1e-2), f"atomic failed at N={N}"

            # Test two-stage
            out3 = triton_reduce_sum_two_stage(x, BLOCK_SIZE_N=1024)
            assert torch.allclose(ref, out3, atol=1e-2), f"two-stage failed at N={N}"
    print("✅ All correctness tests passed!")

# ────────────────────────────────────────
# 性能 Benchmark
# ────────────────────────────────────────
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[2**i for i in range(10, 21)],  # 1K ~ 1M
        line_arg="provider",
        line_vals=["torch", "atomic", "two_stage"],
        line_names=["PyTorch", "Atomic", "Two-Stage"],
        styles=[("red", "-"), ("blue", "--"), ("green", "-.")],
        ylabel="GB/s",
        plot_name="row-wise-sum-performance",
        args={"M": 1024},
    )
)
def benchmark(M, N, provider, device="cuda"):
    x = torch.randn(M, N, device=device, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.sum(x, dim=-1), quantiles=quantiles
        )
    elif provider == "atomic":
        BLOCK_SIZE_N = 512
        def fn():
            return triton_reduce_sum_atomic(x, BLOCK_SIZE_N=BLOCK_SIZE_N)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    elif provider == "two_stage":
        BLOCK_SIZE_N = 1024
        def fn():
            return triton_reduce_sum_two_stage(x, BLOCK_SIZE_N=BLOCK_SIZE_N)
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # 计算带宽：读 M*N*4 bytes，写 M*4 bytes
    total_bytes = (M * N + M) * x.element_size()
    gb_s = total_bytes / (ms * 1e-3) / 1e9
    return gb_s, min_ms, max_ms

if __name__ == "__main__":
    test_correctness()
    benchmark.run(show_plots=True, print_data=True)