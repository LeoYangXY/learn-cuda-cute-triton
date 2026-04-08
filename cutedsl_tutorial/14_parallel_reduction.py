"""
=============================================================================
教程 14: 并行归约 —— 树形归约、Warp Shuffle、前缀和
=============================================================================

GPU 上的归约（Reduction）是最基础也最重要的并行算法之一。
本教程讲解三种层次的归约实现，从 SMEM 树形归约到高效的 Warp Shuffle。

1. Shared Memory 树形归约
   - 数据加载到 SMEM，步长减半配对比较/累加
   - 需要 sync_threads 同步
   - 通用但不是最快

2. Warp Shuffle 归约
   - 利用 __shfl_xor_sync 在寄存器间直接交换数据
   - 无需 SMEM，无需 sync_threads（warp 内自动同步）
   - 一个 warp 32 线程，log2(32)=5 次 shuffle 即可归约

3. Block 级别两阶段 Warp Shuffle 归约
   - 第 1 层：每个 warp 内 shuffle 归约
   - warp leader 写 SMEM
   - 第 2 层：第 0 个 warp 从 SMEM 读取所有 warp 的结果，再次 shuffle

关键 API：
  cute.arch.shuffle_sync_bfly(val, offset=N) — Butterfly XOR warp shuffle
  cutlass.range_constexpr(N)                 — 编译期循环展开（shuffle 的 offset 必须是常量）
  cute.arch.sync_threads()                   — block 内同步
=============================================================================
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import math

BLOCK_SIZE = 256
WARP_SIZE = 32
NUM_WARPS = BLOCK_SIZE // WARP_SIZE  # 8


# =============================================================================
# 版本 1: Shared Memory 树形归约（求最大值）
# =============================================================================
# 原理：
#   1. 每个线程把自己的值写入 sdata[tid]
#   2. stride 从 BLOCK_SIZE/2 开始，每次减半
#   3. if (tid < stride) sdata[tid] = max(sdata[tid], sdata[tid+stride])
#   4. 最终 sdata[0] 就是全局最大值

@cute.kernel
def reduce_max_smem_kernel(
    gInput: cute.Tensor,
    gPartial: cute.Tensor,
    smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    global_idx = bidx * BLOCK_SIZE + tidx
    sdata[tidx] = gInput[global_idx]

    cute.arch.sync_threads()

    # 树形归约
    stride = BLOCK_SIZE // 2
    while stride > 0:
        if tidx < stride:
            other = sdata[tidx + stride]
            mine = sdata[tidx]
            sdata[tidx] = mine if mine > other else other
        cute.arch.sync_threads()
        stride = stride // 2

    # block 最大值写到 partial 数组
    if tidx == 0:
        gPartial[bidx] = sdata[0]


@cute.jit
def reduce_max_smem(mInput: cute.Tensor, mPartial: cute.Tensor):
    n = mInput.shape[0]
    num_blocks = n // BLOCK_SIZE
    smem_layout = cute.make_layout((BLOCK_SIZE,))
    reduce_max_smem_kernel(mInput, mPartial, smem_layout).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1)
    )


# =============================================================================
# 版本 2: Warp Shuffle 两阶段归约（求最大值）
# =============================================================================
# 原理：
#   第 1 层：warp 内 butterfly shuffle
#     for offset in [16, 8, 4, 2, 1]:
#         other = __shfl_xor_sync(val, offset)
#         val = max(val, other)
#     → 每个 warp 的 lane 0 拿到了 warp 内最大值
#
#   第 2 层：跨 warp 归约
#     warp leader 写 sdata[warp_idx]
#     第 0 个 warp 从 sdata 读取，再次 shuffle 归约

@cute.kernel
def reduce_max_shuffle_kernel(
    gInput: cute.Tensor,
    gPartial: cute.Tensor,
    smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    warp_idx = tidx // WARP_SIZE
    lane_idx = tidx % WARP_SIZE

    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    global_idx = bidx * BLOCK_SIZE + tidx
    val = gInput[global_idx]

    # ===== 第 1 层：warp 内 shuffle 归约 =====
    # cute.arch.shuffle_sync_bfly(val, offset) 对应 __shfl_xor_sync
    # offset 必须是编译期常量 → 用 range_constexpr
    for i in cutlass.range_constexpr(int(math.log2(WARP_SIZE))):
        other = cute.arch.shuffle_sync_bfly(val, offset=1 << i)
        val = val if val > other else other

    # warp leader 写 SMEM
    if lane_idx == 0:
        sdata[warp_idx] = val

    cute.arch.sync_threads()

    # ===== 第 2 层：跨 warp 归约（只用第 0 个 warp）=====
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
    smem_layout = cute.make_layout((NUM_WARPS,))
    reduce_max_shuffle_kernel(mInput, mPartial, smem_layout).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1)
    )


# =============================================================================
# 版本 3: Warp Shuffle 求和归约（用于 Layer Norm 等场景）
# =============================================================================
# 和 reduce_max 类似，只是把 max 换成 +

@cute.kernel
def reduce_sum_shuffle_kernel(
    gInput: cute.Tensor,
    gPartial: cute.Tensor,
    smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    warp_idx = tidx // WARP_SIZE
    lane_idx = tidx % WARP_SIZE

    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    global_idx = bidx * BLOCK_SIZE + tidx
    val = gInput[global_idx]

    # warp 内 shuffle 求和
    for i in cutlass.range_constexpr(int(math.log2(WARP_SIZE))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)

    if lane_idx == 0:
        sdata[warp_idx] = val

    cute.arch.sync_threads()

    # 跨 warp 求和
    if warp_idx == 0:
        val2 = sdata[lane_idx] if lane_idx < NUM_WARPS else cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(int(math.log2(WARP_SIZE))):
            val2 = val2 + cute.arch.shuffle_sync_bfly(val2, offset=1 << i)

        if lane_idx == 0:
            gPartial[bidx] = val2


@cute.jit
def reduce_sum_shuffle(mInput: cute.Tensor, mPartial: cute.Tensor):
    n = mInput.shape[0]
    num_blocks = n // BLOCK_SIZE
    smem_layout = cute.make_layout((NUM_WARPS,))
    reduce_sum_shuffle_kernel(mInput, mPartial, smem_layout).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1)
    )


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    N_ELEMS = 1024 * 1024  # 必须是 BLOCK_SIZE 的倍数

    print("=" * 60)
    print(f"教程 14: 并行归约 (N={N_ELEMS})")
    print("=" * 60)

    x = torch.randn(N_ELEMS, device="cuda", dtype=torch.float32)
    ref_max = x.max().item()
    ref_sum = x.sum().item()
    num_blocks = N_ELEMS // BLOCK_SIZE

    x_ = from_dlpack(x, assumed_align=16)

    # ---- 版本 1: SMEM 树形归约 (max) ----
    print("\n--- 版本 1: Shared Memory 树形归约 (max) ---")
    partial1 = torch.empty(num_blocks, device="cuda", dtype=torch.float32)
    p1_ = from_dlpack(partial1, assumed_align=16)
    c1 = cute.compile(reduce_max_smem, x_, p1_)
    c1(x_, p1_)
    result1 = partial1.max().item()
    assert abs(result1 - ref_max) < 1e-5, f"SMEM max 失败: {result1} vs {ref_max}"
    print(f"  max={result1:.6f} (参考: {ref_max:.6f})")
    t1 = benchmark(c1, kernel_arguments=JitArguments(x_, p1_))
    print(f"  耗时: {t1:.2f} us")

    # ---- 版本 2: Warp Shuffle 归约 (max) ----
    print("\n--- 版本 2: Warp Shuffle 两阶段归约 (max) ---")
    partial2 = torch.empty(num_blocks, device="cuda", dtype=torch.float32)
    p2_ = from_dlpack(partial2, assumed_align=16)
    c2 = cute.compile(reduce_max_shuffle, x_, p2_)
    c2(x_, p2_)
    result2 = partial2.max().item()
    assert abs(result2 - ref_max) < 1e-5, f"Shuffle max 失败: {result2} vs {ref_max}"
    print(f"  max={result2:.6f} (参考: {ref_max:.6f})")
    t2 = benchmark(c2, kernel_arguments=JitArguments(x_, p2_))
    print(f"  耗时: {t2:.2f} us")

    # ---- 版本 3: Warp Shuffle 归约 (sum) ----
    print("\n--- 版本 3: Warp Shuffle 两阶段归约 (sum) ---")
    partial3 = torch.empty(num_blocks, device="cuda", dtype=torch.float32)
    p3_ = from_dlpack(partial3, assumed_align=16)
    c3 = cute.compile(reduce_sum_shuffle, x_, p3_)
    c3(x_, p3_)
    result3 = partial3.sum().item()
    # fp32 求和有累积误差，放宽容差
    rel_err = abs(result3 - ref_sum) / max(abs(ref_sum), 1e-8)
    assert rel_err < 1e-3, f"Shuffle sum 失败: {result3} vs {ref_sum} (rel_err={rel_err})"
    print(f"  sum={result3:.4f} (参考: {ref_sum:.4f}, 相对误差: {rel_err:.2e})")
    t3 = benchmark(c3, kernel_arguments=JitArguments(x_, p3_))
    print(f"  耗时: {t3:.2f} us")

    # ---- 汇总 ----
    print("\n" + "=" * 60)
    print("  性能对比")
    print(f"  {'版本':<35} {'耗时(us)':<12}")
    print(f"  {'-'*47}")
    print(f"  {'SMEM 树形归约 (max)':<35} {t1:<12.2f}")
    print(f"  {'Warp Shuffle 两阶段 (max)':<35} {t2:<12.2f}")
    print(f"  {'Warp Shuffle 两阶段 (sum)':<35} {t3:<12.2f}")
    print("=" * 60)

    print()
    print("总结:")
    print("  1. SMEM 树形归约：通用，但每步都需要 sync_threads")
    print("  2. Warp Shuffle：寄存器直通，无需 SMEM，5 次 shuffle 归约 32 个值")
    print("  3. 两阶段 = warp 内 shuffle + 跨 warp SMEM 中转 + 再次 shuffle")
    print()
    print("  关键 API: cute.arch.shuffle_sync_bfly(val, offset=1<<i)")
    print("  注意: offset 必须是编译期常量 → 用 cutlass.range_constexpr")
    print()
    print("  教程 14 完成!")
