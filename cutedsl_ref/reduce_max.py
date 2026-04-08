"""
=============================================================================
CuTeDSL 复现 origin_cuda_kernel/reduce_max/reduce_max.cu
=============================================================================

原生 CUDA 有 5 个版本：
  1. max_kernel_naive           — 全 atomicMax
  2. max_kernel_shared_only     — shared memory 树形规约 + atomicMax
  3. max_kernel_shuffle          — warp shuffle + atomicMax
  4. max_kernel_shuffle_float4   — float4 + warp shuffle + atomicMax
  5. max_kernel_shuffle_pack_half — fp16 128-bit pack + shuffle + atomicMax

CuTeDSL 没有 atomicMax，所以我们用两阶段策略：
  阶段 1（kernel）：每个 block 规约出一个局部最大值，写到 partial_max[blockIdx.x]
  阶段 2（host）：  对 partial_max 做 torch.max() 得到全局最大值

CuTeDSL 实现 2 个版本：
  版本 1: shared memory 树形规约（对应 max_kernel_shared_only）
  版本 2: warp shuffle 两阶段规约（对应 max_kernel_shuffle）
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch
import math

BLOCK_SIZE = 256
WARP_SIZE = 32
NUM_WARPS = BLOCK_SIZE // WARP_SIZE  # 8

# =============================================================================
# 版本 1: Shared Memory 树形规约
# =============================================================================
# 原生 CUDA：
#   extern __shared__ float sdata[];
#   sdata[tid] = input[global_idx];
#   for (stride = blockDim.x/2; stride > 0; stride >>= 1) {
#       if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
#       __syncthreads();
#   }
#   if (tid == 0) atomicMax(output, sdata[0]);

@cute.kernel
def reduce_max_shared_kernel(gInput: cute.Tensor, gPartial: cute.Tensor, smem_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 分配 shared memory
    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    # 原生 CUDA 等价：
    #   int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    #   sdata[tid] = (global_idx < N) ? input[global_idx] : -FLT_MAX;
    global_idx = bidx * BLOCK_SIZE + tidx
    sdata[tidx] = gInput[global_idx]

    cute.arch.sync_threads()

    # 原生 CUDA 等价：树形规约
    #   for (stride = blockDim.x/2; stride > 0; stride >>= 1) {
    #       if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid+stride]);
    #       __syncthreads();
    #   }
    stride = BLOCK_SIZE // 2  # 128
    while stride > 0:
        if tidx < stride:
            other = sdata[tidx + stride]
            mine = sdata[tidx]
            sdata[tidx] = mine if mine > other else other
        cute.arch.sync_threads()
        stride = stride // 2

    # 原生 CUDA 用 atomicMax(output, sdata[0])，CuTeDSL 没有 atomicMax
    # 改为：每个 block 把自己的最大值写到 partial_max[blockIdx.x]
    if tidx == 0:
        gPartial[bidx] = sdata[0]


@cute.jit
def reduce_max_shared(mInput: cute.Tensor, mPartial: cute.Tensor):
    n = mInput.shape[0]
    num_blocks = n // BLOCK_SIZE
    smem_layout = cute.make_layout((BLOCK_SIZE,))
    reduce_max_shared_kernel(mInput, mPartial, smem_layout).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 版本 2: Warp Shuffle 两阶段规约
# =============================================================================
# 原生 CUDA：
#   float val = input[global_idx];
#   val = warp_reduce_max_f32(val);        // warp 内 shuffle 规约
#   if (lane == 0) sdata[warp_idx] = val;  // warp leader 写 smem
#   __syncthreads();
#   if (warp_idx == 0) {                   // 第一个 warp 做跨 warp 规约
#       val = sdata[lane];
#       val = warp_reduce_max_f32(val);
#       if (lane == 0) atomicMax(output, val);
#   }
#
# CuTeDSL 中 warp shuffle 用 cute.arch.shuffle_sync_bfly(val, offset=...)
# 对应原生 CUDA 的 __shfl_xor_sync(0xffffffff, val, mask)

@cute.kernel
def reduce_max_shuffle_kernel(gInput: cute.Tensor, gPartial: cute.Tensor, smem_layout: cute.Layout):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    warp_idx = tidx // WARP_SIZE
    lane_idx = tidx % WARP_SIZE

    # 分配 shared memory（存每个 warp 的局部最大值）
    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    # 原生 CUDA 等价：float val = input[global_idx];
    global_idx = bidx * BLOCK_SIZE + tidx
    val = gInput[global_idx]

    # ===== 第 1 层：warp 内 shuffle 规约 =====
    # 原生 CUDA 等价：
    #   for (offset = 16; offset > 0; offset >>= 1)
    #       val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    for i in cutlass.range_constexpr(int(math.log2(WARP_SIZE))):
        other = cute.arch.shuffle_sync_bfly(val, offset=1 << i)
        val = val if val > other else other

    # 原生 CUDA 等价：if (lane == 0) sdata[warp_idx] = val;
    if lane_idx == 0:
        sdata[warp_idx] = val

    cute.arch.sync_threads()

    # ===== 第 2 层：跨 warp 规约（只用第一个 warp）=====
    # 原生 CUDA 等价：
    #   if (warp_idx == 0) {
    #       val = sdata[lane];
    #       val = warp_reduce_max_f32<NUM_WARPS>(val);
    #       if (lane == 0) atomicMax(output, val);
    #   }
    if warp_idx == 0:
        val2 = sdata[lane_idx] if lane_idx < NUM_WARPS else cutlass.Float32(-1e38)
        for i in cutlass.range_constexpr(int(math.log2(NUM_WARPS))):
            other2 = cute.arch.shuffle_sync_bfly(val2, offset=1 << i)
            val2 = val2 if val2 > other2 else other2

        if lane_idx == 0:
            gPartial[bidx] = val2


@cute.jit
def reduce_max_shuffle(mInput: cute.Tensor, mPartial: cute.Tensor):
    n = mInput.shape[0]
    num_blocks = n // BLOCK_SIZE
    smem_layout = cute.make_layout((NUM_WARPS,))  # 只需要 NUM_WARPS 个 float
    reduce_max_shuffle_kernel(mInput, mPartial, smem_layout).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 测试
# =============================================================================
if __name__ == "__main__":
    N = 1024 * 1024  # 必须是 BLOCK_SIZE 的倍数

    print("=" * 60)
    print(f"CuTeDSL Reduce Max (N={N})")
    print("=" * 60)

    x = torch.randn(N, device="cuda", dtype=torch.float32)
    ref = x.max().item()
    num_blocks = N // BLOCK_SIZE

    x_ = from_dlpack(x, assumed_align=16)

    # 版本 1: shared memory 树形规约
    partial1 = torch.empty(num_blocks, device="cuda", dtype=torch.float32)
    p1_ = from_dlpack(partial1, assumed_align=16)
    c1 = cute.compile(reduce_max_shared, x_, p1_)
    c1(x_, p1_)
    result1 = partial1.max().item()
    assert abs(result1 - ref) < 1e-5, f"Shared 版验证失败！{result1} vs {ref}"
    print(f"✅ Shared Memory 树形规约 正确 (max={result1:.6f})")
    t1 = benchmark(c1, kernel_arguments=JitArguments(x_, p1_))
    print(f"   耗时: {t1:.2f} µs")

    # 版本 2: warp shuffle
    partial2 = torch.empty(num_blocks, device="cuda", dtype=torch.float32)
    p2_ = from_dlpack(partial2, assumed_align=16)
    c2 = cute.compile(reduce_max_shuffle, x_, p2_)
    c2(x_, p2_)
    result2 = partial2.max().item()
    assert abs(result2 - ref) < 1e-5, f"Shuffle 版验证失败！{result2} vs {ref}"
    print(f"✅ Warp Shuffle 两阶段规约 正确 (max={result2:.6f})")
    t2 = benchmark(c2, kernel_arguments=JitArguments(x_, p2_))
    print(f"   耗时: {t2:.2f} µs")

    print(f"\n{'='*60}")
    print(f"  {'版本':<35} {'耗时(µs)':<12}")
    print(f"  {'-'*47}")
    print(f"  {'Shared Memory 树形规约':<35} {t1:<12.2f}")
    print(f"  {'Warp Shuffle 两阶段规约':<35} {t2:<12.2f}")
    print(f"{'='*60}")
