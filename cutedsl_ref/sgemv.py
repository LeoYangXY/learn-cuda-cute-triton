"""
=============================================================================
CuTeDSL 复现 origin_cuda_kernel/sgemv/sgemv.cu
=============================================================================

原生 CUDA 有 3 个版本：
  1. sgemv_32_kernel  — K 是 32 的倍数，每个 warp 处理一行
  2. sgemv_128_kernel — K 是 128 的倍数，float4 向量化
  3. sgemv_16_kernel  — K=16，每个 warp 处理 2 行

语义：y[M] = A[M, K] × x[K]
  每行做点积：y[i] = sum_j(A[i][j] * x[j])

CuTeDSL 实现 2 个版本：
  版本 1: K 是 32 的倍数（对应 sgemv_32_kernel）
  版本 2: K=16，每 warp 2 行（对应 sgemv_16_kernel）
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch
import math

WARP_SIZE = 32
WARPS_PER_BLOCK = 4

# =============================================================================
# 工具：warp reduce sum
# =============================================================================
@cute.jit
def warp_reduce_sum(val, width=WARP_SIZE):
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


# =============================================================================
# 版本 1: K 是 32 的倍数 — 每个 warp 处理一行
# =============================================================================
# 原生 CUDA：
#   int row = row_start + row_offset;   // warp 负责的行
#   int lane = threadIdx.x % 32;
#   float sum = 0;
#   for (int time = 0; time < K/32; time++) {
#       int col = time * 32 + lane;     // 合并访存：连续 lane 访问连续列
#       sum += a[row * K + col] * x[col];
#   }
#   float row_sum = warp_reduce_sum(sum);
#   if (lane == 0) y[row] = row_sum;

@cute.kernel
def sgemv_k32_kernel(gA: cute.Tensor, gX: cute.Tensor, gY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 原生 CUDA 等价：
    #   int row_start = blockIdx.x * warps_per_block;
    #   int row_offset = threadIdx.x / WARP_SIZE;
    #   int row = row_start + row_offset;
    #   int lane = threadIdx.x % WARP_SIZE;
    row_start = bidx * WARPS_PER_BLOCK
    warp_idx = tidx // WARP_SIZE
    lane = tidx % WARP_SIZE
    row = row_start + warp_idx

    K = gA.shape[1]
    NUM_ITERS = K // WARP_SIZE  # K / 32

    # 原生 CUDA 等价：
    #   for (int time = 0; time < NUM_WARPS; time++) {
    #       int col = time * 32 + lane;
    #       sum += a[row * K + col] * x[col];
    #   }
    # 关键：连续 lane 访问连续列 → 合并访存
    sum_val = cutlass.Float32(0.0)
    for time in range(NUM_ITERS):
        col = time * WARP_SIZE + lane
        sum_val = sum_val + gA[row, col] * gX[col]

    # 原生 CUDA 等价：float row_sum = warp_reduce_sum(sum);
    row_sum = warp_reduce_sum(sum_val)

    # 原生 CUDA 等价：if (lane == 0) y[row] = row_sum;
    if lane == 0:
        gY[row] = row_sum


@cute.jit
def sgemv_k32(mA: cute.Tensor, mX: cute.Tensor, mY: cute.Tensor):
    M = mA.shape[0]
    block_size = WARPS_PER_BLOCK * WARP_SIZE  # 128
    grid_size = (M + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK
    sgemv_k32_kernel(mA, mX, mY).launch(
        grid=(grid_size, 1, 1), block=(block_size, 1, 1))


# =============================================================================
# 版本 2: K=16 — 每个 warp 处理 2 行
# =============================================================================
# 原生 CUDA：
#   一个 block 有 128 个 thread（4 个 warp）
#   每 16 个 thread 负责一行 → 一个 warp 处理 2 行 → 一个 block 处理 8 行
#   col = lane % 16;
#   float val = a[row * K + col] * x[col];
#   float sum = warp_reduce_sum<16>(val);  // 只在 16-lane 子组内规约
#   if (col == 0) y[row] = sum;

K16 = 16

@cute.kernel
def sgemv_k16_kernel(gA: cute.Tensor, gX: cute.Tensor, gY: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 原生 CUDA 等价：
    #   int row_start = blockIdx.x * warps_per_block * 2;
    #   int row_offset = threadIdx.x / 16;
    #   int row = row_start + row_offset;
    rows_per_block = WARPS_PER_BLOCK * 2  # 8
    row_start = bidx * rows_per_block
    row_offset = tidx // K16
    row = row_start + row_offset

    # 原生 CUDA 等价：int col = lane % 16;
    col = tidx % K16

    # 原生 CUDA 等价：float val = a[row * K + col] * x[col];
    val = gA[row, col] * gX[col]

    # 原生 CUDA 等价：float sum = warp_reduce_sum<16>(val);
    # 只在 16-lane 子组内规约
    sum_val = warp_reduce_sum(val, width=K16)

    # 原生 CUDA 等价：if (col == 0) y[row] = sum;
    if col == 0:
        gY[row] = sum_val


@cute.jit
def sgemv_k16(mA: cute.Tensor, mX: cute.Tensor, mY: cute.Tensor):
    M = mA.shape[0]
    block_size = WARPS_PER_BLOCK * WARP_SIZE  # 128
    rows_per_block = WARPS_PER_BLOCK * 2  # 8
    grid_size = (M + rows_per_block - 1) // rows_per_block
    sgemv_k16_kernel(mA, mX, mY).launch(
        grid=(grid_size, 1, 1), block=(block_size, 1, 1))


# =============================================================================
# 测试
# =============================================================================
if __name__ == "__main__":
    M = 128

    print("=" * 60)
    print("CuTeDSL SGEMV")
    print("=" * 60)

    # Test 1: K=96 (multiple of 32)
    K1 = 96
    print(f"\n===== K%32==0 (M={M}, K={K1}) =====")
    A1 = torch.randn(M, K1, device="cuda", dtype=torch.float32)
    x1 = torch.randn(K1, device="cuda", dtype=torch.float32)
    ref1 = A1 @ x1
    y1 = torch.empty(M, device="cuda", dtype=torch.float32)

    A1_, x1_, y1_ = from_dlpack(A1, assumed_align=16), from_dlpack(x1, assumed_align=16), from_dlpack(y1, assumed_align=16)
    c1 = cute.compile(sgemv_k32, A1_, x1_, y1_)
    c1(A1_, x1_, y1_)
    assert torch.allclose(y1, ref1, atol=1e-3, rtol=1e-3), f"K32 验证失败！max_diff={(y1-ref1).abs().max().item()}"
    print("✅ sgemv_k32 正确")
    t1 = benchmark(c1, kernel_arguments=JitArguments(A1_, x1_, y1_))
    print(f"   耗时: {t1:.2f} µs")

    # Test 2: K=16
    K3 = 16
    M3 = 128
    print(f"\n===== K=16 (M={M3}, K={K3}) =====")
    A3 = torch.randn(M3, K3, device="cuda", dtype=torch.float32)
    x3 = torch.randn(K3, device="cuda", dtype=torch.float32)
    ref3 = A3 @ x3
    y3 = torch.empty(M3, device="cuda", dtype=torch.float32)

    A3_, x3_, y3_ = from_dlpack(A3, assumed_align=16), from_dlpack(x3, assumed_align=16), from_dlpack(y3, assumed_align=16)
    c3 = cute.compile(sgemv_k16, A3_, x3_, y3_)
    c3(A3_, x3_, y3_)
    assert torch.allclose(y3, ref3, atol=1e-3, rtol=1e-3), f"K16 验证失败！max_diff={(y3-ref3).abs().max().item()}"
    print("✅ sgemv_k16 正确")
    t3 = benchmark(c3, kernel_arguments=JitArguments(A3_, x3_, y3_))
    print(f"   耗时: {t3:.2f} µs")
