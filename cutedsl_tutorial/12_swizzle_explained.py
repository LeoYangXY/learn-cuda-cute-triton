"""
=============================================================================
教程 12: Swizzle 详解 —— 消除 Shared Memory Bank Conflict
=============================================================================

什么是 Bank Conflict？
  Shared Memory 有 32 个 bank，每个 bank 4 字节宽。
  同一 warp 的 32 个线程如果访问了同一个 bank → 串行化 → 性能暴跌。

  行优先存储 16 列 fp16 矩阵时，一行 = 16 × 2B = 32B = 8 个 bank。
  连续 4 行恰好覆盖 32 个 bank 后循环回来，于是第 0 行和第 4 行的同一列
  落在同一个 bank → conflict！

什么是 Swizzle？
  Swizzle 用 XOR 操作重新映射地址：
    swizzled_offset[M:M+B] = offset[M:M+B] XOR offset[S:S+B]
  效果：每隔几行就交换列段的 bank 映射，消除 conflict。

CuTeDSL 中使用 Swizzle 的三层 API：

  底层：cute.make_swizzle(B, M, S) + cute.make_composed_layout(...)
  中层：make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, dtype) + tile_to_shape(...)
  高层：sm90_utils.make_smem_layout_a/b(...)  （自动根据 MMA 类型选最优 swizzle）

本教程不依赖 Hopper/Blackwell 硬件，用纯 Python 可视化 + CuTeDSL kernel 演示。

关键 API：
  cute.make_swizzle(bits, base, shift)          — 创建 Swizzle 对象
  cute.make_composed_layout(inner, offset, outer) — 组合 Swizzle + Layout
  cute.composition(layout_a, layout_b)           — 布局组合（Swizzle 的底层机制）
=============================================================================
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# =============================================================================
# 第一部分：纯 Python 可视化 —— 理解 Swizzle 的 XOR 操作
# =============================================================================
# 不需要 GPU，纯 CPU 上理解 Swizzle 的地址映射

def swizzle_demo_cpu():
    """
    用纯 Python 展示 Swizzle 如何重排地址。
    以 fp16 矩阵 (16 行 × 16 列) 为例：
      每行 16 × 2B = 32B
      每 128 bits = 8 个 fp16 元素为一个 "段"
      一行有 2 个段（段 0: col 0~7, 段 1: col 8~15）
    """
    print("=" * 60)
    print("第一部分：纯 Python Swizzle 可视化")
    print("=" * 60)
    print()

    rows = 16
    col_stride = 16  # fp16 列数
    num_elems_per_128b = 8  # 128 bits / 16 bits = 8 个 fp16

    def swizzle_permuted_j(i, j, col_stride=16, num_elems_per_128b=8):
        """
        计算 swizzle 后的列段索引。
        核心公式：(j_segment XOR i_segment) % num_segments × segment_size
        """
        return (
            (int(j / num_elems_per_128b) ^ int(i / 4))
            % (int(col_stride / num_elems_per_128b))
        ) * num_elems_per_128b

    # --- 无 Swizzle ---
    print("无 Swizzle 时，每行的物理列段起始位置：")
    print("  (逻辑列 → 物理列)")
    print(f"  {'row':>5} | col 0~7 → 物理 | col 8~15 → 物理")
    print(f"  {'-'*5}-+-{'-'*16}-+-{'-'*16}")
    for i in range(8):
        print(f"  {i:>5} | col 0~7 → 0~7   | col 8~15 → 8~15")
    print()
    print("  注意：row 0 和 row 4 的 col 0~7 都在 bank 0~3 → conflict!")
    print()

    # --- 有 Swizzle ---
    print("有 Swizzle 时，每行的物理列段起始位置：")
    print(f"  {'row':>5} | 逻辑col 0 → 物理col | 逻辑col 8 → 物理col")
    print(f"  {'-'*5}-+-{'-'*22}-+-{'-'*22}")
    for i in range(8):
        phys_0 = swizzle_permuted_j(i, 0, col_stride, num_elems_per_128b)
        phys_8 = swizzle_permuted_j(i, 8, col_stride, num_elems_per_128b)
        marker_0 = " ← 交换!" if phys_0 != 0 else ""
        marker_8 = " ← 交换!" if phys_8 != 8 else ""
        print(f"  {i:>5} | col 0~7  → {phys_0}~{phys_0+7:<3}{marker_0:10s}"
              f" | col 8~15 → {phys_8}~{phys_8+7:<3}{marker_8}")

    print()
    print("  row 0~3: 不变 (0→0, 8→8)")
    print("  row 4~7: 交换 (0→8, 8→0)")
    print("  → 同一列的 row 0 和 row 4 现在落在不同 bank → 消除 conflict!")
    print()


# =============================================================================
# 第二部分：CuTeDSL 底层 API —— make_swizzle + make_composed_layout
# =============================================================================

@cute.jit
def swizzle_low_level():
    """
    底层 API 演示：手动创建 Swizzle 并与 Layout 组合。
    """
    cute.printf("=" * 60 + "\n")
    cute.printf("  底层 API: make_swizzle + make_composed_layout\n")
    cute.printf("=" * 60 + "\n\n")

    # 1. 创建普通 Layout（无 swizzle）
    bare_layout = cute.make_layout(
        shape=(64, 128),
        stride=(128, 1)
    )
    cute.printf("普通 Layout (64x128 行优先): {}\n", bare_layout)

    # 2. 创建 Swizzle 对象
    # Swizzle<3, 4, 3> 是最常用的配置：
    #   B=3: 3 位 XOR (8 种 pattern)
    #   M=4: 从地址第 4 位开始 (行信息)
    #   S=3: 从地址第 3 位开始 (列信息)
    swizzle = cute.make_swizzle(3, 4, 3)
    cute.printf("Swizzle<3,4,3> created\n")

    # 3. 组合成 ComposedLayout
    # ComposedLayout = Swizzle ∘ offset ∘ Layout
    # 逻辑坐标 → Layout 算出线性偏移 → Swizzle XOR 扰乱偏移 → 物理地址
    composed = cute.make_composed_layout(
        inner=swizzle,    # Swizzle 函数
        offset=0,         # 偏移量（通常为 0）
        outer=bare_layout # 基础 Layout
    )
    cute.printf("ComposedLayout created (Swizzle o Layout)\n\n")

    # 4. 对比有无 Swizzle 的地址映射
    cute.printf("对比 row 0 和 row 4 的列偏移:\n")
    cute.printf("  col    bare_offset  swizzle_offset\n")
    for col in range(0, 16, 8):
        bare_off = cute.crd2idx((0, col), bare_layout)
        swiz_off = cute.crd2idx((0, col), composed)
        cute.printf("  row0,col%d: %d  vs  %d\n", col, bare_off, swiz_off)
    for col in range(0, 16, 8):
        bare_off = cute.crd2idx((4, col), bare_layout)
        swiz_off = cute.crd2idx((4, col), composed)
        cute.printf("  row4,col%d: %d  vs  %d\n", col, bare_off, swiz_off)


# =============================================================================
# 第三部分：实际应用 —— 带 Swizzle 的 SMEM 拷贝
# =============================================================================
# 用 Swizzle 的 ComposedLayout 分配 SMEM，做 GMEM → SMEM → GMEM 拷贝，
# 验证数据正确性（swizzle 对用户透明，不影响逻辑正确性）。

TILE_M = 64
TILE_K = 64
NUM_THREADS = 128
VEC = 4  # 每线程 4 个 fp16 = 64 bits

@cute.kernel
def swizzle_copy_kernel(
    gA: cute.Tensor,
    gC: cute.Tensor,
    smem_layout: cute.Layout,    # 基础 Layout（outer）
    swizzle_obj: cute.Swizzle,   # Swizzle 对象（inner）
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    blkA = cute.local_tile(gA, (TILE_M, TILE_K), (bidx, bidy))
    blkC = cute.local_tile(gC, (TILE_M, TILE_K), (bidx, bidy))

    # 分配带 Swizzle 的 SMEM tensor
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(
        cutlass.Float16, smem_layout, swizzle=swizzle_obj
    )

    # GMEM → SMEM（带 swizzle）
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gA.element_type,
        num_bits_per_copy=gA.element_type.width * VEC,
    )
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    tAgA = thr_copy.partition_S(blkA)
    tAsA = thr_copy.partition_D(sA)
    cute.copy(tiled_copy, tAgA, tAsA)

    cute.arch.sync_threads()

    # SMEM → GMEM（验证）
    tSsA = thr_copy.partition_S(sA)
    tSgC = thr_copy.partition_D(blkC)
    rA = cute.make_fragment_like(tSsA)
    cute.copy(tiled_copy, tSsA, rA)
    cute.copy(tiled_copy, rA, tSgC)


@cute.jit
def swizzle_copy_host(mA: cute.Tensor, mC: cute.Tensor):
    # 基础 Layout（outer part of ComposedLayout）
    smem_layout = cute.make_ordered_layout(
        (TILE_M, TILE_K), order=(1, 0)
    )

    # Swizzle 对象（inner part of ComposedLayout）
    swizzle_obj = cute.make_swizzle(3, 4, 3)

    THR_K = TILE_K // VEC  # 16
    THR_M = NUM_THREADS // THR_K  # 8
    thr_layout = cute.make_layout((THR_M, THR_K), stride=(THR_K, 1))
    val_layout = cute.make_layout((1, VEC), stride=(0, 1))

    M, K = mA.shape
    grid_m = M // TILE_M
    grid_k = K // TILE_K

    swizzle_copy_kernel(mA, mC, smem_layout, swizzle_obj, thr_layout, val_layout).launch(
        grid=(grid_m, grid_k, 1),
        block=(NUM_THREADS, 1, 1),
    )


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()

    # Part 1: 纯 Python 可视化
    swizzle_demo_cpu()

    # Part 2: 底层 API
    print("=" * 60)
    print("第二部分：CuTeDSL 底层 API")
    print("=" * 60)
    swizzle_low_level()
    torch.cuda.synchronize()

    # Part 3: 带 Swizzle 的 SMEM 拷贝验证
    print("\n" + "=" * 60)
    print("第三部分：带 Swizzle 的 SMEM 拷贝验证")
    print("=" * 60)

    M, K = 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, K, device="cuda", dtype=torch.float16)
    A_ = from_dlpack(A, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    compiled = cute.compile(swizzle_copy_host, A_, C_)
    compiled(A_, C_)

    if torch.allclose(C, A, atol=1e-3):
        print("  Swizzle SMEM 拷贝验证通过! (GMEM -> SMEM[swizzle] -> GMEM)")
        print("  Swizzle 对逻辑正确性完全透明。")
    else:
        diff = (C - A).abs().max().item()
        print(f"  验证失败! 最大误差: {diff}")

    print()
    print("=" * 60)
    print("总结：Swizzle 三层 API")
    print("=" * 60)
    print()
    print("  底层 (完全手动):")
    print("    swz = cute.make_swizzle(3, 4, 3)")
    print("    composed = cute.make_composed_layout(inner=swz, offset=0, outer=layout)")
    print()
    print("  中层 (半自动):")
    print("    atom = make_smem_layout_atom(SmemLayoutAtomKind.K_SW128, Float16)")
    print("    layout = cute.tile_to_shape(atom, (64, 128), order=(0, 1))")
    print("    sA = smem.allocate_tensor(Float16, layout.outer, swizzle=layout.inner)")
    print()
    print("  高层 (全自动):")
    print("    layout = sm90_utils.make_smem_layout_a(a_layout, tile_mnk, dtype, stages)")
    print("    sA = storage.sA.get_tensor(layout.outer, swizzle=layout.inner)")
    print()
    print("  教程 12 完成!")
