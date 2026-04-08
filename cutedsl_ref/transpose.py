"""
=============================================================================
CuTeDSL 复现 origin_cuda_kernel/transpose/transpose.cu
=============================================================================

原生 CUDA 有 4 个版本：
  1. naive_transpose               — 朴素版：每 thread 搬 1 个元素
  2. transpose_4float              — float4 读取，标量写入
  3. mat_transpose_shared_kernel   — shared memory 实现合并写
  4. mat_transpose_shared_kernel_padding — shared memory + padding 消除 bank conflict

CuTeDSL 实现 2 个版本：
  版本 1: 朴素版（标量读写，对应 naive_transpose）
  版本 2: shared memory 版（对应 mat_transpose_shared_kernel_padding）

语义：src[M, N] → dst[N, M]，即 dst[j][i] = src[i][j]
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

TILE = 32

# =============================================================================
# 版本 1: 朴素转置 — 每个 thread 搬 1 个元素
# =============================================================================
# 原生 CUDA：
#   __global__ void naive_transpose(float* dst, const float* src, int M, int N) {
#       int row = blockIdx.x * blockDim.x + threadIdx.x;
#       int col = blockIdx.y * blockDim.y + threadIdx.y;
#       if (row < M && col < N) dst[col * M + row] = src[row * N + col];
#   }

@cute.kernel
def transpose_naive_kernel(gSrc: cute.Tensor, gDst: cute.Tensor):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # 原生 CUDA 等价：
    #   int row = blockIdx.x * blockDim.x + threadIdx.x;
    #   int col = blockIdx.y * blockDim.y + threadIdx.y;
    row = bidx * TILE + tidx
    col = bidy * TILE + tidy

    # 原生 CUDA 等价：dst[col * M + row] = src[row * N + col];
    # gSrc shape = (M, N)，gDst shape = (N, M)
    gDst[col, row] = gSrc[row, col]


@cute.jit
def transpose_naive(mSrc: cute.Tensor, mDst: cute.Tensor):
    M, N = mSrc.shape[0], mSrc.shape[1]
    grid_x = (M + TILE - 1) // TILE
    grid_y = (N + TILE - 1) // TILE
    transpose_naive_kernel(mSrc, mDst).launch(
        grid=(grid_x, grid_y, 1), block=(TILE, TILE, 1))


# =============================================================================
# 版本 2: Shared Memory + Padding 转置
# =============================================================================
# 原生 CUDA：
#   __shared__ float tile[TILE_SIZE][TILE_SIZE+1];  // +1 padding 消除 bank conflict
#   // 1. 合并读：src[global_row][global_col] → tile[local_row][local_col]
#   // 2. __syncthreads()
#   // 3. 合并写：tile[local_col][local_row] → dst[new_row][new_col]
#   //    关键：读 tile 的行列互换了，这样 warp 写 dst 时地址连续
#
# 为什么需要 shared memory？
#   朴素版中，读 src 是合并的（连续列），但写 dst 是非合并的（跳行写）
#   用 shared memory 做中转：先合并读到 tile，再从 tile 的转置方向合并写出

@cute.kernel
def transpose_smem_kernel(gSrc: cute.Tensor, gDst: cute.Tensor, smem_layout: cute.Layout):
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # 分配 shared memory
    smem = cutlass.utils.SmemAllocator()
    tile = smem.allocate_tensor(cutlass.Float32, smem_layout)

    # 原生 CUDA 等价：
    #   int block_row = blockIdx.y * TILE_SIZE;
    #   int block_col = blockIdx.x * TILE_SIZE;
    #   int global_row = block_row + local_row;
    #   int global_col = block_col + local_col;
    block_row = bidy * TILE
    block_col = bidx * TILE
    global_row = block_row + tidy
    global_col = block_col + tidx

    # 步骤 1：合并读 src → shared memory
    # 原生 CUDA 等价：tile[local_row][local_col] = src[global_row * N + global_col];
    # warp 中 tidx 连续 → global_col 连续 → src 地址连续 → 合并读 ✅
    tile[tidy, tidx] = gSrc[global_row, global_col]

    cute.arch.sync_threads()

    # 步骤 2：合并写 shared memory → dst
    # 原生 CUDA 等价：
    #   int new_row = block_col + local_row;
    #   int new_col = block_row + local_col;
    #   dst[new_row * M + new_col] = tile[local_col][local_row];
    #
    # 关键点：我们读 tile[tidx][tidy]（行列互换了！）
    # 这样 warp 中 tidx 连续 → new_row = block_col + tidy 对 warp 内相同
    #                        → new_col = block_row + tidx 对 warp 内连续
    # → dst 地址连续 → 合并写 ✅
    #
    # bank conflict 分析：
    #   tile 声明为 [TILE][TILE+1]，+1 是 padding
    #   读 tile[tidx][tidy] 时：tidx 是行，tidy 是列
    #   warp 内 tidx 连续变化 → 访问不同行的同一列
    #   bank_id = (tidx * (TILE+1) + tidy) % 32 = (tidx * 33 + tidy) % 32
    #   tidx 连续 → bank_id 连续 → 无 bank conflict ✅
    new_row = block_col + tidy
    new_col = block_row + tidx
    gDst[new_row, new_col] = tile[tidx, tidy]


@cute.jit
def transpose_smem(mSrc: cute.Tensor, mDst: cute.Tensor):
    M, N = mSrc.shape[0], mSrc.shape[1]

    # shared memory 布局：(TILE, TILE+1)，+1 是 padding 消除 bank conflict
    # 原生 CUDA 等价：__shared__ float tile[32][33];
    smem_layout = cute.make_ordered_layout((TILE, TILE + 1), order=(1, 0))

    grid_x = (N + TILE - 1) // TILE
    grid_y = (M + TILE - 1) // TILE
    transpose_smem_kernel(mSrc, mDst, smem_layout).launch(
        grid=(grid_x, grid_y, 1), block=(TILE, TILE, 1))


# =============================================================================
# 测试
# =============================================================================
if __name__ == "__main__":
    M, N = 1024, 2048

    print("=" * 60)
    print(f"CuTeDSL Transpose (M={M}, N={N})")
    print("=" * 60)

    src = torch.randn(M, N, device="cuda", dtype=torch.float32)
    ref = src.t().contiguous()

    src_ = from_dlpack(src, assumed_align=16)

    # 版本 1: naive
    dst1 = torch.empty(N, M, device="cuda", dtype=torch.float32)
    dst1_ = from_dlpack(dst1, assumed_align=16)
    c1 = cute.compile(transpose_naive, src_, dst1_)
    c1(src_, dst1_)
    assert torch.allclose(dst1, ref, atol=1e-5), f"Naive 验证失败！max_diff={(dst1-ref).abs().max().item()}"
    print("✅ Naive 转置 正确")
    t1 = benchmark(c1, kernel_arguments=JitArguments(src_, dst1_))
    print(f"   耗时: {t1:.2f} µs")

    # 版本 2: shared memory + padding
    dst2 = torch.empty(N, M, device="cuda", dtype=torch.float32)
    dst2_ = from_dlpack(dst2, assumed_align=16)
    c2 = cute.compile(transpose_smem, src_, dst2_)
    c2(src_, dst2_)
    assert torch.allclose(dst2, ref, atol=1e-5), f"SMEM 验证失败！max_diff={(dst2-ref).abs().max().item()}"
    print("✅ Shared Memory + Padding 转置 正确")
    t2 = benchmark(c2, kernel_arguments=JitArguments(src_, dst2_))
    print(f"   耗时: {t2:.2f} µs")

    print(f"\n{'='*60}")
    print(f"  {'版本':<35} {'耗时(µs)':<12}")
    print(f"  {'-'*47}")
    print(f"  {'Naive (非合并写)':<35} {t1:<12.2f}")
    print(f"  {'SMEM + Padding (合并读写)':<35} {t2:<12.2f}")
    print(f"{'='*60}")
