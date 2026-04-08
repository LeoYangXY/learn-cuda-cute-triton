"""
=============================================================================
教程 03: Tiling 与 Partition —— CuTE 的分块代数
=============================================================================

在 GPU 编程中，我们不可能让一个线程处理整个矩阵。
核心思路是：把大问题切成小块（Tile），分配给不同的 Block 和 Thread。

CuTE 提供了一套优雅的代数操作来完成这件事：

1. local_tile(tensor, tiler, coord, proj)
   - 将全局张量按 tiler 大小切块，取出 coord 指定的那一块
   - proj 用于投影：选择哪些维度参与 tiling
   - 这是 CTA 级别的分块

2. zipped_divide(tensor, tiler)
   - 将张量按 tiler 大小切分，返回 ((tile_shape), (num_tiles)) 的层次张量
   - 可以用 [tile_coord, inner_coord] 来索引

3. local_partition(tensor, layout, idx)
   - 将一个 tile 进一步分配给各个线程
   - 这是 Thread 级别的分区

GEMM 中的 Tiling 层次：
  ┌──────────────────────────────────────────┐
  │  全局矩阵 C (M × N)                      │
  │  ┌──────────┐                             │
  │  │ CTA Tile │ ← local_tile 切出           │
  │  │ (BM × BN)│                             │
  │  │ ┌──┬──┐  │                             │
  │  │ │T0│T1│  │ ← partition 分给各线程       │
  │  │ ├──┼──┤  │                             │
  │  │ │T2│T3│  │                             │
  │  │ └──┴──┘  │                             │
  │  └──────────┘                             │
  └──────────────────────────────────────────┘

本教程通过一个 Naive GEMM 来演示这些概念。
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch


# =============================================================================
# 第一部分：Naive GEMM（CUDA 风格，不用 CuTE 分块）
# =============================================================================
# 先写一个最朴素的 GEMM，每个线程计算 C 的一个元素
# C[i,j] = sum_k A[i,k] * B[j,k]  （注意 B 是转置存储的：B[N,K]）

BM_NAIVE, BN_NAIVE = 16, 16  # 每个 Block 处理 16×16 的 C 子块

@cute.kernel
def naive_gemm_kernel(
    gA: cute.Tensor,    # A: [M, K]
    gB: cute.Tensor,    # B: [N, K]（注意是 N×K，即 B^T 的行优先存储）
    gC: cute.Tensor,    # C: [M, N]
    M: int, N: int, K: int,
):
    # 每个 Block 用 (BM, BN) 个线程，每个线程算 C 的一个元素
    bidx, bidy, _ = cute.arch.block_idx()    # Block 在 Grid 中的位置
    tidx, tidy, _ = cute.arch.thread_idx()   # Thread 在 Block 中的位置

    # 这个线程负责 C[row, col]
    row = bidy * BM_NAIVE + tidy   # M 维度
    col = bidx * BN_NAIVE + tidx   # N 维度

    # 累加器（用 Float32 避免精度损失）
    acc = cutlass.Float32(0)

    # 沿 K 维度累加：C[row,col] = Σ_k A[row,k] * B[col,k]
    for k in range(K):
        acc += cutlass.Float32(gA[row, k]) * cutlass.Float32(gB[col, k])

    gC[row, col] = acc


@cute.jit
def naive_gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]

    # Grid: 每个 Block 处理 BM×BN 的 C 子块
    # 注意 grid 的 x 对应 N 维度，y 对应 M 维度
    naive_gemm_kernel(mA, mB, mC, M, N, K).launch(
        grid=(N // BN_NAIVE, M // BM_NAIVE, 1),
        block=(BN_NAIVE, BM_NAIVE, 1)  # x=col, y=row
    )


# =============================================================================
# 第二部分：CuTE 风格 GEMM —— 使用 local_tile + MMA 分区
# =============================================================================
# 这是 CuTE 的标准写法，引入了：
# - local_tile：CTA 级别的分块
# - tiled_mma + partition：线程级别的分区
# - make_fragment：寄存器级别的数据

BM, BN, BK = 16, 32, 16  # CTA Tile 大小

@cute.kernel
def cute_gemm_kernel(
    gA: cute.Tensor,    # A: [M, K]
    gB: cute.Tensor,    # B: [N, K]
    gC: cute.Tensor,    # C: [M, N]
):
    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # ---- 第一步：CTA 级别分块（local_tile）----
    # tiler = (BM, BN, BK) 定义了 CTA 处理的子块大小
    # coord = (bidx, bidy, None) 指定当前 CTA 负责哪个子块
    #   - bidx 选择 M 维度的第几个 tile
    #   - bidy 选择 N 维度的第几个 tile
    #   - None 表示 K 维度不固定（后面循环遍历）
    # proj 投影：选择哪些维度参与 tiling
    #   - A[M,K]: proj=(1, None, 1) → 用 M 和 K 维度的 tiler
    #   - B[N,K]: proj=(None, 1, 1) → 用 N 和 K 维度的 tiler
    #   - C[M,N]: proj=(1, 1, None) → 用 M 和 N 维度的 tiler

    tiler = (BM, BN, BK)
    coord = (bidx, bidy, None)

    # gA_tile: (BM, BK, K//BK) — 当前 CTA 的 A 子块，K 维度被切成多个 tile
    gA_tile = cute.local_tile(gA, tiler, coord, proj=(1, None, 1))
    # gB_tile: (BN, BK, K//BK)
    gB_tile = cute.local_tile(gB, tiler, coord, proj=(None, 1, 1))
    # gC_tile: (BM, BN)
    gC_tile = cute.local_tile(gC, tiler, coord, proj=(1, 1, None))

    # ---- 第二步：创建 MMA 分区 ----
    # MmaUniversalOp 是最基础的 MMA 原子操作（1×1 标量乘加）
    # atoms_layout 定义了如何将 256 个线程排列成 16×16 的网格
    # 每个线程负责 C 的一个元素
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)

    # ---- 第三步：线程级别分区（partition）----
    # get_slice(tidx) 获取当前线程的 MMA 分区视图
    thr_mma = tiled_mma.get_slice(tidx)

    # partition_C: 将 gC_tile 按 MMA 模式分区，得到当前线程负责的 C 元素
    tCgC = thr_mma.partition_C(gC_tile)
    # make_fragment_C: 在寄存器中创建对应大小的累加器
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0)  # 初始化为 0

    # ---- 第四步：主循环 ----
    K_tiles = gA_tile.shape[2]  # K 维度被切成了多少个 tile
    for k in range(K_tiles):
        # 取出第 k 个 K-tile
        gA_k = gA_tile[None, None, k]  # (BM, BK)
        gB_k = gB_tile[None, None, k]  # (BN, BK)

        # 对 A 和 B 也做线程级分区
        tCgA = thr_mma.partition_A(gA_k)
        tCgB = thr_mma.partition_B(gB_k)

        # 创建寄存器 fragment 并从全局内存加载
        tCrA = tiled_mma.make_fragment_A(tCgA)
        tCrB = tiled_mma.make_fragment_B(tCgB)
        tCrA.store(tCgA.load())  # GMEM → Register
        tCrB.store(tCgB.load())  # GMEM → Register

        # 执行矩阵乘加：C += A × B
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    # ---- 第五步：写回全局内存 ----
    # 数据类型一致（都是 Float32），直接写回
    tCgC.store(tCrC.load())


@cute.jit
def cute_gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    grid_m = (mA.shape[0] + BM - 1) // BM
    grid_n = (mB.shape[0] + BN - 1) // BN

    cute_gemm_kernel(mA, mB, mC).launch(
        grid=(grid_m, grid_n, 1),
        block=(256, 1, 1)  # 16×16 = 256 线程
    )


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    M_SIZE, N_SIZE, K_SIZE = 1024, 1024, 1024

    # 使用 Float32 以避免类型转换的复杂性（后续教程会讲 Float16 + Tensor Core）
    A = torch.randn((M_SIZE, K_SIZE), device="cuda", dtype=torch.float32)
    B = torch.randn((N_SIZE, K_SIZE), device="cuda", dtype=torch.float32)
    ref = torch.matmul(A, B.T)  # PyTorch 参考结果

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)

    # ---- Part 1: Naive GEMM ----
    print("=" * 60)
    print("第一部分：Naive GEMM（CUDA 风格）")
    print("=" * 60)

    C1 = torch.empty((M_SIZE, N_SIZE), device="cuda", dtype=torch.float32)
    C1_ = from_dlpack(C1, assumed_align=16)

    compiled_naive = cute.compile(naive_gemm_host, A_, B_, C1_)
    compiled_naive(A_, B_, C1_)

    assert torch.allclose(C1, ref, atol=1e-1, rtol=1e-1), "Naive GEMM 验证失败！"
    print("✅ Naive GEMM 正确性验证通过！")

    time_naive = benchmark(compiled_naive, kernel_arguments=JitArguments(A_, B_, C1_))
    tflops_naive = (2 * M_SIZE * N_SIZE * K_SIZE) / (time_naive * 1e6)
    print(f"⏱  耗时: {time_naive:.2f} µs | TFLOPS: {tflops_naive:.4f}")

    # ---- Part 2: CuTE GEMM ----
    print("\n" + "=" * 60)
    print("第二部分：CuTE 风格 GEMM（local_tile + MMA 分区）")
    print("=" * 60)

    C2 = torch.empty((M_SIZE, N_SIZE), device="cuda", dtype=torch.float32)
    C2_ = from_dlpack(C2, assumed_align=16)

    compiled_cute = cute.compile(cute_gemm_host, A_, B_, C2_)
    compiled_cute(A_, B_, C2_)

    assert torch.allclose(C2, ref, atol=1e-1, rtol=1e-1), "CuTE GEMM 验证失败！"
    print("✅ CuTE GEMM 正确性验证通过！")

    time_cute = benchmark(compiled_cute, kernel_arguments=JitArguments(A_, B_, C2_))
    tflops_cute = (2 * M_SIZE * N_SIZE * K_SIZE) / (time_cute * 1e6)
    print(f"⏱  耗时: {time_cute:.2f} µs | TFLOPS: {tflops_cute:.4f}")

    print(f"\n📊 CuTE vs Naive 加速比: {time_naive / time_cute:.2f}x")

    # ---- PyTorch 性能对比 ----
    print("\n" + "=" * 60)
    print("📊 性能对比：CuTeDSL vs PyTorch")
    print("=" * 60)

    C_pt = torch.empty_like(ref)
    for _ in range(10):
        torch.matmul(A, B.T, out=C_pt)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    num_iter = 100
    for _ in range(num_iter):
        torch.matmul(A, B.T, out=C_pt)
    end.record()
    torch.cuda.synchronize()
    pt_time_us = start.elapsed_time(end) * 1000 / num_iter
    pt_tflops = (2 * M_SIZE * N_SIZE * K_SIZE) / (pt_time_us * 1e6)
    print(f"⏱  PyTorch    耗时: {pt_time_us:.2f} µs | TFLOPS: {pt_tflops:.4f}")
    print(f"⏱  Naive GEMM 耗时: {time_naive:.2f} µs | TFLOPS: {tflops_naive:.4f}")
    print(f"⏱  CuTE  GEMM 耗时: {time_cute:.2f} µs | TFLOPS: {tflops_cute:.4f}")
    print(f"📊 PyTorch / CuTE 速度比: {pt_time_us / time_cute:.2f}x")

    print("\n🎉 教程 03 全部完成！")
