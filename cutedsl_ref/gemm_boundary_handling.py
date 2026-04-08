"""
=============================================================================
GEMM 不整除边界处理 — 当 M/N/K 不能被 Tile 大小整除时怎么办？
=============================================================================

在 sliced-k tiled GEMM 中，我们按如下方式计算 grid 维度：
    grid.x = N / kTileN
    grid.y = M / kTileM

但当 M、N、K 不能被 tile 大小整除时，上面的整除除法会丢掉"尾巴"。
本文件演示 3 种处理方法，全部使用不整除尺寸验证正确性：

  方法 1: Grid 向上取整 + kernel 内逐元素边界检查（最直白）
  方法 2: Grid 向上取整 + 加载时 fill-zero 边界保护（SMEM 版本）
  方法 3: Host 端 Padding — 把矩阵 pad 到整除，kernel 无需改动

语义：C[M,N] = A[M,K] × B[K,N]  (B 非转置，fp32)

测试尺寸：M=130, N=67, K=53  (故意对 tile 大小不整除)
Tile 参数：BM=BN=32, BK=32, TM=TN=4
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

# =============================================================================
# 公共参数
# =============================================================================
BM, BN, BK = 32, 32, 32    # CTA tile 大小
TM, TN = 4, 4              # 每个线程计算的 C 子块大小
# 每个 block 有 (BM/TM) × (BN/TN) = 8×8 = 64 个线程
THREADS_X = BN // TN  # 8
THREADS_Y = BM // TM  # 8


# =============================================================================
# 方法 1: Grid 向上取整 + kernel 内逐元素边界检查
# =============================================================================
# 这是最直白的方法：
#   - grid 维度用 ceil 向上取整，确保覆盖所有元素
#   - 边界 block 中，每个线程在加载和写回时检查下标是否越界
#   - 越界的加载填 0（对乘法无影响），越界的写回跳过
#
# 优点：不浪费显存，通用性强
# 缺点：kernel 内有大量 if 判断，对性能有一定影响
#
# 原生 CUDA 等价伪代码：
#   int row = blockIdx.y * BM + threadIdx.y * TM;
#   int col = blockIdx.x * BN + threadIdx.x * TN;
#   // 加载时：if (row+i < M && k+j < K) load else 0
#   // 写回时：if (row+i < M && col+j < N) store

@cute.kernel
def sgemm_boundary_check_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    sA_layout: cute.Layout, sB_layout: cute.Layout,
    M_val: cutlass.Constexpr[int], N_val: cutlass.Constexpr[int],
    K_val: cutlass.Constexpr[int],
):
    """方法 1: 每次加载/写回都做边界检查"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # 分配 shared memory
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, sA_layout)
    sB = smem.allocate_tensor(cutlass.Float32, sB_layout)

    # 当前 block 负责的 C 子块的起始行/列
    block_row = bidy * BM
    block_col = bidx * BN

    # 当前线程在 block 内的位置（用于计算和加载）
    compute_row = block_row + tidy * TM
    compute_col = block_col + tidx * TN
    load_row = tidy * TM
    load_col = tidx * TN

    # 寄存器累加器，手动展开 TM=TN=4 的 16 个元素
    c00 = cutlass.Float32(0.0); c01 = cutlass.Float32(0.0)
    c02 = cutlass.Float32(0.0); c03 = cutlass.Float32(0.0)
    c10 = cutlass.Float32(0.0); c11 = cutlass.Float32(0.0)
    c12 = cutlass.Float32(0.0); c13 = cutlass.Float32(0.0)
    c20 = cutlass.Float32(0.0); c21 = cutlass.Float32(0.0)
    c22 = cutlass.Float32(0.0); c23 = cutlass.Float32(0.0)
    c30 = cutlass.Float32(0.0); c31 = cutlass.Float32(0.0)
    c32 = cutlass.Float32(0.0); c33 = cutlass.Float32(0.0)

    # K 方向循环：向上取整，最后一个 tile 可能不满 BK
    # 原生 CUDA：for (int k0 = 0; k0 < (K + BK - 1) / BK; k0++)
    num_k_tiles = (K_val + BK - 1) // BK
    for k0 in range(num_k_tiles):
        k_offset = k0 * BK

        # ===== 加载 A[BM, BK] 到 SMEM（带边界检查）=====
        # 原生 CUDA 等价：
        #   if (block_row + load_row + i < M && k_offset + load_col + j < K)
        #       s_a[load_row+i][load_col+j] = a[...];
        #   else
        #       s_a[load_row+i][load_col+j] = 0.0f;  // 填 0
        for i in range(TM):
            for j in range(TN):
                a_row = block_row + load_row + i
                a_col = k_offset + load_col + j
                if a_row < M_val and a_col < K_val:
                    sA[load_row + i, load_col + j] = gA[a_row, a_col]
                else:
                    sA[load_row + i, load_col + j] = cutlass.Float32(0.0)

        # ===== 加载 B[BK, BN] 到 SMEM（带边界检查）=====
        for i in range(TM):
            for j in range(TN):
                b_row = k_offset + load_row + i
                b_col = block_col + load_col + j
                if b_row < K_val and b_col < N_val:
                    sB[load_row + i, load_col + j] = gB[b_row, b_col]
                else:
                    sB[load_row + i, load_col + j] = cutlass.Float32(0.0)

        cute.arch.sync_threads()

        # ===== 计算 TM×TN 个 C 元素（SMEM 内计算，无需边界检查）=====
        # 因为越界位置已经填了 0，所以 0 × anything = 0，不影响结果
        s_row_base = tidy * TM
        s_col_base = tidx * TN

        for p in range(BK):
            a0 = sA[s_row_base, p]; a1 = sA[s_row_base+1, p]
            a2 = sA[s_row_base+2, p]; a3 = sA[s_row_base+3, p]
            b0 = sB[p, s_col_base]; b1 = sB[p, s_col_base+1]
            b2 = sB[p, s_col_base+2]; b3 = sB[p, s_col_base+3]

            c00 = c00 + a0 * b0; c01 = c01 + a0 * b1; c02 = c02 + a0 * b2; c03 = c03 + a0 * b3
            c10 = c10 + a1 * b0; c11 = c11 + a1 * b1; c12 = c12 + a1 * b2; c13 = c13 + a1 * b3
            c20 = c20 + a2 * b0; c21 = c21 + a2 * b1; c22 = c22 + a2 * b2; c23 = c23 + a2 * b3
            c30 = c30 + a3 * b0; c31 = c31 + a3 * b1; c32 = c32 + a3 * b2; c33 = c33 + a3 * b3

        cute.arch.sync_threads()

    # ===== 写回 GMEM（带边界检查）=====
    # 原生 CUDA 等价：
    #   if (compute_row + i < M && compute_col + j < N)
    #       c[(compute_row+i)*N + compute_col+j] = c_reg[i][j];
    if compute_row < M_val and compute_col < N_val:
        gC[compute_row, compute_col] = c00
    if compute_row < M_val and compute_col+1 < N_val:
        gC[compute_row, compute_col+1] = c01
    if compute_row < M_val and compute_col+2 < N_val:
        gC[compute_row, compute_col+2] = c02
    if compute_row < M_val and compute_col+3 < N_val:
        gC[compute_row, compute_col+3] = c03

    if compute_row+1 < M_val and compute_col < N_val:
        gC[compute_row+1, compute_col] = c10
    if compute_row+1 < M_val and compute_col+1 < N_val:
        gC[compute_row+1, compute_col+1] = c11
    if compute_row+1 < M_val and compute_col+2 < N_val:
        gC[compute_row+1, compute_col+2] = c12
    if compute_row+1 < M_val and compute_col+3 < N_val:
        gC[compute_row+1, compute_col+3] = c13

    if compute_row+2 < M_val and compute_col < N_val:
        gC[compute_row+2, compute_col] = c20
    if compute_row+2 < M_val and compute_col+1 < N_val:
        gC[compute_row+2, compute_col+1] = c21
    if compute_row+2 < M_val and compute_col+2 < N_val:
        gC[compute_row+2, compute_col+2] = c22
    if compute_row+2 < M_val and compute_col+3 < N_val:
        gC[compute_row+2, compute_col+3] = c23

    if compute_row+3 < M_val and compute_col < N_val:
        gC[compute_row+3, compute_col] = c30
    if compute_row+3 < M_val and compute_col+1 < N_val:
        gC[compute_row+3, compute_col+1] = c31
    if compute_row+3 < M_val and compute_col+2 < N_val:
        gC[compute_row+3, compute_col+2] = c32
    if compute_row+3 < M_val and compute_col+3 < N_val:
        gC[compute_row+3, compute_col+3] = c33


@cute.jit
def sgemm_boundary_check(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                          M_val: cutlass.Constexpr[int], N_val: cutlass.Constexpr[int],
                          K_val: cutlass.Constexpr[int]):
    """方法 1 的 host 端：grid 向上取整"""
    # 关键：向上取整！ceil(N/BN), ceil(M/BM)
    grid_x = (N_val + BN - 1) // BN
    grid_y = (M_val + BM - 1) // BM

    sA_layout = cute.make_ordered_layout((BM, BK), order=(1, 0))
    sB_layout = cute.make_ordered_layout((BK, BN), order=(1, 0))

    sgemm_boundary_check_kernel(
        mA, mB, mC, sA_layout, sB_layout, M_val, N_val, K_val
    ).launch(grid=(grid_x, grid_y, 1), block=(THREADS_X, THREADS_Y, 1))


# =============================================================================
# 方法 2: Grid 向上取整 + 加载时 fill-zero + 写回时跳过整个 block
# =============================================================================
# 和方法 1 类似，但有两个区别：
#   - 先统一把 SMEM 填 0，再只写有效部分（逻辑更清晰）
#   - 用 if block_row < M and block_col < N 包裹整个逻辑体，
#     完全越界的 block 整块跳过所有计算
#
# 本质区别：方法 1 每个 load/store 都有独立的 if 检查
#           方法 2 先统一清零再覆写有效值，且用外层 if 跳过越界 block
#
# 注意：CuTeDSL kernel 不支持 early return（会报 DSLAstPreprocessorError），
#       但可以用 if 条件包裹整个逻辑体来达到同样效果。
#
# 注意：CuTeDSL kernel 不支持 early return，所以不能用 if ... : return
#       但可以用 if 把整个逻辑体包起来，效果等价。
#       对于完全越界的 block，if 条件不满足 → 整个 block 的线程什么都不做。

@cute.kernel
def sgemm_fill_zero_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    sA_layout: cute.Layout, sB_layout: cute.Layout,
    M_val: cutlass.Constexpr[int], N_val: cutlass.Constexpr[int],
    K_val: cutlass.Constexpr[int],
):
    """方法 2: 先填 0 再写有效值，完全越界的 block 跳过"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, sA_layout)
    sB = smem.allocate_tensor(cutlass.Float32, sB_layout)

    block_row = bidy * BM
    block_col = bidx * BN
    compute_row = block_row + tidy * TM
    compute_col = block_col + tidx * TN
    load_row = tidy * TM
    load_col = tidx * TN

    c00 = cutlass.Float32(0.0); c01 = cutlass.Float32(0.0)
    c02 = cutlass.Float32(0.0); c03 = cutlass.Float32(0.0)
    c10 = cutlass.Float32(0.0); c11 = cutlass.Float32(0.0)
    c12 = cutlass.Float32(0.0); c13 = cutlass.Float32(0.0)
    c20 = cutlass.Float32(0.0); c21 = cutlass.Float32(0.0)
    c22 = cutlass.Float32(0.0); c23 = cutlass.Float32(0.0)
    c30 = cutlass.Float32(0.0); c31 = cutlass.Float32(0.0)
    c32 = cutlass.Float32(0.0); c33 = cutlass.Float32(0.0)

    # ===== 用 if 包裹整个逻辑体，等价于 early return =====
    # 完全越界的 block（block_row >= M 或 block_col >= N）不进入 if，直接跳过
    if block_row < M_val and block_col < N_val:
        num_k_tiles = (K_val + BK - 1) // BK
        for k0 in range(num_k_tiles):
            k_offset = k0 * BK

            # ===== 加载到 SMEM =====
            # 先整块填 0，再写有效部分
            # 好处：逻辑清晰，SMEM 不会有垃圾值
            for i in range(TM):
                for j in range(TN):
                    sA[load_row + i, load_col + j] = cutlass.Float32(0.0)
                    sB[load_row + i, load_col + j] = cutlass.Float32(0.0)

            for i in range(TM):
                for j in range(TN):
                    a_row = block_row + load_row + i
                    a_col = k_offset + load_col + j
                    if a_row < M_val and a_col < K_val:
                        sA[load_row + i, load_col + j] = gA[a_row, a_col]

                    b_row = k_offset + load_row + i
                    b_col = block_col + load_col + j
                    if b_row < K_val and b_col < N_val:
                        sB[load_row + i, load_col + j] = gB[b_row, b_col]

            cute.arch.sync_threads()

            # ===== 计算（越界位置是 0，乘加不影响结果）=====
            s_row_base = tidy * TM
            s_col_base = tidx * TN

            for p in range(BK):
                a0 = sA[s_row_base, p]; a1 = sA[s_row_base+1, p]
                a2 = sA[s_row_base+2, p]; a3 = sA[s_row_base+3, p]
                b0 = sB[p, s_col_base]; b1 = sB[p, s_col_base+1]
                b2 = sB[p, s_col_base+2]; b3 = sB[p, s_col_base+3]

                c00 = c00 + a0 * b0; c01 = c01 + a0 * b1; c02 = c02 + a0 * b2; c03 = c03 + a0 * b3
                c10 = c10 + a1 * b0; c11 = c11 + a1 * b1; c12 = c12 + a1 * b2; c13 = c13 + a1 * b3
                c20 = c20 + a2 * b0; c21 = c21 + a2 * b1; c22 = c22 + a2 * b2; c23 = c23 + a2 * b3
                c30 = c30 + a3 * b0; c31 = c31 + a3 * b1; c32 = c32 + a3 * b2; c33 = c33 + a3 * b3

            cute.arch.sync_threads()

        # ===== 写回 GMEM（带边界检查）=====
        if compute_row < M_val and compute_col < N_val:
            gC[compute_row, compute_col] = c00
        if compute_row < M_val and compute_col+1 < N_val:
            gC[compute_row, compute_col+1] = c01
        if compute_row < M_val and compute_col+2 < N_val:
            gC[compute_row, compute_col+2] = c02
        if compute_row < M_val and compute_col+3 < N_val:
            gC[compute_row, compute_col+3] = c03

        if compute_row+1 < M_val and compute_col < N_val:
            gC[compute_row+1, compute_col] = c10
        if compute_row+1 < M_val and compute_col+1 < N_val:
            gC[compute_row+1, compute_col+1] = c11
        if compute_row+1 < M_val and compute_col+2 < N_val:
            gC[compute_row+1, compute_col+2] = c12
        if compute_row+1 < M_val and compute_col+3 < N_val:
            gC[compute_row+1, compute_col+3] = c13

        if compute_row+2 < M_val and compute_col < N_val:
            gC[compute_row+2, compute_col] = c20
        if compute_row+2 < M_val and compute_col+1 < N_val:
            gC[compute_row+2, compute_col+1] = c21
        if compute_row+2 < M_val and compute_col+2 < N_val:
            gC[compute_row+2, compute_col+2] = c22
        if compute_row+2 < M_val and compute_col+3 < N_val:
            gC[compute_row+2, compute_col+3] = c23

        if compute_row+3 < M_val and compute_col < N_val:
            gC[compute_row+3, compute_col] = c30
        if compute_row+3 < M_val and compute_col+1 < N_val:
            gC[compute_row+3, compute_col+1] = c31
        if compute_row+3 < M_val and compute_col+2 < N_val:
            gC[compute_row+3, compute_col+2] = c32
        if compute_row+3 < M_val and compute_col+3 < N_val:
            gC[compute_row+3, compute_col+3] = c33


@cute.jit
def sgemm_fill_zero(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
                     M_val: cutlass.Constexpr[int], N_val: cutlass.Constexpr[int],
                     K_val: cutlass.Constexpr[int]):
    """方法 2 的 host 端"""
    grid_x = (N_val + BN - 1) // BN
    grid_y = (M_val + BM - 1) // BM

    sA_layout = cute.make_ordered_layout((BM, BK), order=(1, 0))
    sB_layout = cute.make_ordered_layout((BK, BN), order=(1, 0))

    sgemm_fill_zero_kernel(
        mA, mB, mC, sA_layout, sB_layout, M_val, N_val, K_val
    ).launch(grid=(grid_x, grid_y, 1), block=(THREADS_X, THREADS_Y, 1))


# =============================================================================
# 方法 3: Host 端 Padding — 把矩阵 pad 到 tile 整除，kernel 无需改动
# =============================================================================
# 思路：
#   在调用 kernel 之前，用 PyTorch 把 A、B、C pad 到 tile 大小的整数倍
#   kernel 内部完全不需要边界检查（因为维度一定整除）
#   计算完后裁剪回原始大小
#
# 优点：kernel 代码最简单，逻辑清晰，编译器优化空间最大
# 缺点：浪费显存（多分配 padding 部分），浪费带宽（多加载/存储 0）
#
# 适用场景：
#   - 矩阵不太大、padding 开销可接受
#   - 快速原型验证
#   - 不想在 kernel 里加复杂的边界逻辑

# 这个 kernel 就是 sgemm.py 中最朴素的 tiled kernel，完全不处理边界
@cute.kernel
def sgemm_no_boundary_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    sA_layout: cute.Layout, sB_layout: cute.Layout,
):
    """方法 3 的 kernel: 假设维度一定整除，无任何边界检查"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    K = gA.shape[1]

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float32, sA_layout)
    sB = smem.allocate_tensor(cutlass.Float32, sB_layout)

    block_row = bidy * BM
    block_col = bidx * BN
    compute_row = block_row + tidy * TM
    compute_col = block_col + tidx * TN
    load_row = tidy * TM
    load_col = tidx * TN

    c00 = cutlass.Float32(0.0); c01 = cutlass.Float32(0.0)
    c02 = cutlass.Float32(0.0); c03 = cutlass.Float32(0.0)
    c10 = cutlass.Float32(0.0); c11 = cutlass.Float32(0.0)
    c12 = cutlass.Float32(0.0); c13 = cutlass.Float32(0.0)
    c20 = cutlass.Float32(0.0); c21 = cutlass.Float32(0.0)
    c22 = cutlass.Float32(0.0); c23 = cutlass.Float32(0.0)
    c30 = cutlass.Float32(0.0); c31 = cutlass.Float32(0.0)
    c32 = cutlass.Float32(0.0); c33 = cutlass.Float32(0.0)

    # K 一定整除 BK，所以直接 K // BK
    for k0 in range(K // BK):
        k_offset = k0 * BK

        for i in range(TM):
            for j in range(TN):
                sA[load_row + i, load_col + j] = gA[block_row + load_row + i, k_offset + load_col + j]
                sB[load_row + i, load_col + j] = gB[k_offset + load_row + i, block_col + load_col + j]

        cute.arch.sync_threads()

        s_row_base = tidy * TM
        s_col_base = tidx * TN

        for p in range(BK):
            a0 = sA[s_row_base, p]; a1 = sA[s_row_base+1, p]
            a2 = sA[s_row_base+2, p]; a3 = sA[s_row_base+3, p]
            b0 = sB[p, s_col_base]; b1 = sB[p, s_col_base+1]
            b2 = sB[p, s_col_base+2]; b3 = sB[p, s_col_base+3]

            c00 = c00 + a0 * b0; c01 = c01 + a0 * b1; c02 = c02 + a0 * b2; c03 = c03 + a0 * b3
            c10 = c10 + a1 * b0; c11 = c11 + a1 * b1; c12 = c12 + a1 * b2; c13 = c13 + a1 * b3
            c20 = c20 + a2 * b0; c21 = c21 + a2 * b1; c22 = c22 + a2 * b2; c23 = c23 + a2 * b3
            c30 = c30 + a3 * b0; c31 = c31 + a3 * b1; c32 = c32 + a3 * b2; c33 = c33 + a3 * b3

        cute.arch.sync_threads()

    gC[compute_row, compute_col]     = c00; gC[compute_row, compute_col+1]   = c01
    gC[compute_row, compute_col+2]   = c02; gC[compute_row, compute_col+3]   = c03
    gC[compute_row+1, compute_col]   = c10; gC[compute_row+1, compute_col+1] = c11
    gC[compute_row+1, compute_col+2] = c12; gC[compute_row+1, compute_col+3] = c13
    gC[compute_row+2, compute_col]   = c20; gC[compute_row+2, compute_col+1] = c21
    gC[compute_row+2, compute_col+2] = c22; gC[compute_row+2, compute_col+3] = c23
    gC[compute_row+3, compute_col]   = c30; gC[compute_row+3, compute_col+1] = c31
    gC[compute_row+3, compute_col+2] = c32; gC[compute_row+3, compute_col+3] = c33


@cute.jit
def sgemm_no_boundary(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    """方法 3 的 host 端: 维度保证整除，直接整除除法"""
    M_val, K_val = mA.shape[0], mA.shape[1]
    N_val = mB.shape[1]

    grid_x = N_val // BN
    grid_y = M_val // BM

    sA_layout = cute.make_ordered_layout((BM, BK), order=(1, 0))
    sB_layout = cute.make_ordered_layout((BK, BN), order=(1, 0))

    sgemm_no_boundary_kernel(mA, mB, mC, sA_layout, sB_layout).launch(
        grid=(grid_x, grid_y, 1), block=(THREADS_X, THREADS_Y, 1))


def pad_to_multiple(x: torch.Tensor, multiple_m: int, multiple_n: int) -> torch.Tensor:
    """在 host 端把 2D 矩阵 pad 到 (multiple_m, multiple_n) 的整数倍，多余部分填 0"""
    m, n = x.shape
    pad_m = (multiple_m - m % multiple_m) % multiple_m
    pad_n = (multiple_n - n % multiple_n) % multiple_n
    if pad_m == 0 and pad_n == 0:
        return x
    # F.pad 的参数是 (left, right, top, bottom)
    return torch.nn.functional.pad(x, (0, pad_n, 0, pad_m), value=0.0)


# =============================================================================
# 测试 + 性能对比
# =============================================================================
if __name__ == "__main__":
    # 故意选不整除的尺寸
    M, N, K = 130, 67, 53
    print("=" * 70)
    print(f"GEMM 不整除边界处理 — M={M}, N={N}, K={K}")
    print(f"Tile: BM={BM}, BN={BN}, BK={BK}, TM={TM}, TN={TN}")
    print(f"M%BM={M%BM}, N%BN={N%BN}, K%BK={K%BK}  ← 全部不整除！")
    print("=" * 70)

    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    ref = A @ B  # PyTorch 参考结果

    results = []

    # =========================================================================
    # 方法 1: Grid 向上取整 + 逐元素边界检查
    # =========================================================================
    print("\n--- 方法 1: Grid 向上取整 + 逐元素边界检查 ---")
    C1 = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    A_, B_, C1_ = (
        from_dlpack(A, assumed_align=16),
        from_dlpack(B, assumed_align=16),
        from_dlpack(C1, assumed_align=16),
    )
    compiled_v1 = cute.compile(sgemm_boundary_check, A_, B_, C1_, M, N, K)
    compiled_v1(A_, B_, C1_)
    max_diff_1 = (C1 - ref).abs().max().item()
    assert max_diff_1 < 1e-2, f"方法 1 失败！max_diff={max_diff_1}"
    print(f"✅ 方法 1 正确  max_diff={max_diff_1:.6f}")
    t1 = benchmark(compiled_v1, kernel_arguments=JitArguments(A_, B_, C1_))
    print(f"   耗时: {t1:.2f} µs")
    results.append(("方法1: 逐元素边界检查", t1))

    # =========================================================================
    # 方法 2: Grid 向上取整 + fill-zero + early return
    # =========================================================================
    print("\n--- 方法 2: Grid 向上取整 + fill-zero + early return ---")
    C2 = torch.zeros(M, N, device="cuda", dtype=torch.float32)
    C2_ = from_dlpack(C2, assumed_align=16)
    compiled_v2 = cute.compile(sgemm_fill_zero, A_, B_, C2_, M, N, K)
    compiled_v2(A_, B_, C2_)
    max_diff_2 = (C2 - ref).abs().max().item()
    assert max_diff_2 < 1e-2, f"方法 2 失败！max_diff={max_diff_2}"
    print(f"✅ 方法 2 正确  max_diff={max_diff_2:.6f}")
    t2 = benchmark(compiled_v2, kernel_arguments=JitArguments(A_, B_, C2_))
    print(f"   耗时: {t2:.2f} µs")
    results.append(("方法2: fill-zero + early return", t2))

    # =========================================================================
    # 方法 3: Host 端 Padding
    # =========================================================================
    print("\n--- 方法 3: Host 端 Padding ---")
    A_pad = pad_to_multiple(A, BM, BK)  # (160, 64)
    B_pad = pad_to_multiple(B, BK, BN)  # (64, 96)
    M_pad, K_pad_a = A_pad.shape
    K_pad_b, N_pad = B_pad.shape
    assert K_pad_a == K_pad_b, f"Padding 后 K 不一致: {K_pad_a} vs {K_pad_b}"
    print(f"   原始: A({M},{K}) × B({K},{N}) → C({M},{N})")
    print(f"   Pad后: A({M_pad},{K_pad_a}) × B({K_pad_b},{N_pad}) → C({M_pad},{N_pad})")
    print(f"   显存额外开销: A +{(M_pad*K_pad_a - M*K)*4}B, B +{(K_pad_b*N_pad - K*N)*4}B, C +{(M_pad*N_pad - M*N)*4}B")

    C3_pad = torch.zeros(M_pad, N_pad, device="cuda", dtype=torch.float32)
    A_pad_, B_pad_, C3_pad_ = (
        from_dlpack(A_pad, assumed_align=16),
        from_dlpack(B_pad, assumed_align=16),
        from_dlpack(C3_pad, assumed_align=16),
    )
    compiled_v3 = cute.compile(sgemm_no_boundary, A_pad_, B_pad_, C3_pad_)
    compiled_v3(A_pad_, B_pad_, C3_pad_)
    C3 = C3_pad[:M, :N]  # 裁剪回原始大小
    max_diff_3 = (C3 - ref).abs().max().item()
    assert max_diff_3 < 1e-2, f"方法 3 失败！max_diff={max_diff_3}"
    print(f"✅ 方法 3 正确  max_diff={max_diff_3:.6f}")
    t3 = benchmark(compiled_v3, kernel_arguments=JitArguments(A_pad_, B_pad_, C3_pad_))
    print(f"   耗时: {t3:.2f} µs  (含 padding 部分的多余计算)")
    results.append(("方法3: Host Padding (无边界检查)", t3))

    # =========================================================================
    # 更大尺寸测试（确保非边角情况也正确）
    # =========================================================================
    print("\n" + "=" * 70)
    print("更大尺寸验证: M=257, N=131, K=97")
    print("=" * 70)

    M2, N2, K2 = 257, 131, 97
    A2 = torch.randn(M2, K2, device="cuda", dtype=torch.float32)
    B2 = torch.randn(K2, N2, device="cuda", dtype=torch.float32)
    ref2 = A2 @ B2

    # 方法 1
    C2_v1 = torch.zeros(M2, N2, device="cuda", dtype=torch.float32)
    A2_, B2_, C2_v1_ = (
        from_dlpack(A2, assumed_align=16),
        from_dlpack(B2, assumed_align=16),
        from_dlpack(C2_v1, assumed_align=16),
    )
    compiled = cute.compile(sgemm_boundary_check, A2_, B2_, C2_v1_, M2, N2, K2)
    compiled(A2_, B2_, C2_v1_)
    d1 = (C2_v1 - ref2).abs().max().item()

    # 方法 2
    C2_v2 = torch.zeros(M2, N2, device="cuda", dtype=torch.float32)
    C2_v2_ = from_dlpack(C2_v2, assumed_align=16)
    compiled = cute.compile(sgemm_fill_zero, A2_, B2_, C2_v2_, M2, N2, K2)
    compiled(A2_, B2_, C2_v2_)
    d2 = (C2_v2 - ref2).abs().max().item()

    # 方法 3
    A2_pad = pad_to_multiple(A2, BM, BK)
    B2_pad = pad_to_multiple(B2, BK, BN)
    C2_v3_pad = torch.zeros(A2_pad.shape[0], B2_pad.shape[1], device="cuda", dtype=torch.float32)
    A2_pad_, B2_pad_, C2_v3_pad_ = (
        from_dlpack(A2_pad, assumed_align=16),
        from_dlpack(B2_pad, assumed_align=16),
        from_dlpack(C2_v3_pad, assumed_align=16),
    )
    compiled = cute.compile(sgemm_no_boundary, A2_pad_, B2_pad_, C2_v3_pad_)
    compiled(A2_pad_, B2_pad_, C2_v3_pad_)
    d3 = (C2_v3_pad[:M2, :N2] - ref2).abs().max().item()

    print(f"  方法1 max_diff={d1:.6f}  {'✅' if d1 < 1e-2 else '❌'}")
    print(f"  方法2 max_diff={d2:.6f}  {'✅' if d2 < 1e-2 else '❌'}")
    print(f"  方法3 max_diff={d3:.6f}  {'✅' if d3 < 1e-2 else '❌'}")

    assert d1 < 1e-2 and d2 < 1e-2 and d3 < 1e-2, "大尺寸验证失败！"
    print("✅ 大尺寸验证全部通过")

    # =========================================================================
    # 结果汇总
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"  {'方法':<40} {'耗时(µs)':<12}")
    print(f"  {'-'*52}")
    for name, t in results:
        print(f"  {name:<40} {t:<12.2f}")
    print(f"{'='*70}")
    print()
    print("总结:")
    print("  方法1（逐元素检查）: 最通用，每个 load/store 都有 if 判断")
    print("  方法2（fill-zero）  : 先填 0 再写有效值 + early return 跳过完全越界的 block")
    print("  方法3（Host Padding）: kernel 最简洁，但浪费显存和带宽")
    print()
    print("实际工程中推荐: 方法 1 或 2 的思路 + CuTE 的 predicated copy（见 sgemm.py V3+ 版本）")
    print("CuTE 的 local_tile + ceil_div 自动处理了 grid 的向上取整，")
    print("配合 copy 操作的 predicate tensor，可以优雅地处理边界。")
