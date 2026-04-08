"""
=============================================================================
教程 11: Layout Composition 与 Identity Tensor
=============================================================================

Layout Composition 是 CuTE 中最强大的代数操作之一。
理解 composition 就能理解 CuTE 如何实现 Swizzle、转置、子 tile 索引。

什么是 Composition？
  给定两个 Layout：A 和 B
  R = composition(A, B) 表示 R(c) = A(B(c))
  即：先用 B 把坐标 c 映射成一个"中间索引"，再用 A 把这个中间索引映射成最终偏移。

  用人话说：B 是"坐标变换器"，A 是"地址映射器"。

为什么 Composition 重要？
  1. Swizzle 就是 composition：swizzle ∘ layout
  2. 子 tile 选择就是 composition：layout ∘ tile_selector
  3. V 矩阵从 K 的 SMEM 空间复用就是 composition：k_layout ∘ transpose_layout
     (funny_cute/flashmla_dsl.py 中就是这样做的)

Identity Tensor 是什么？
  make_identity_tensor(shape) 创建一个张量，每个位置存储的值就是它自己的坐标。
  identity[(2, 3)] 的值就是 (2, 3)。
  
  用途：验证分块/分区策略是否正确 —— 分块后检查 tile 内的坐标是否符合预期。

关键 API：
  cute.composition(base_layout, tiler_layout) — 布局组合
  cute.make_identity_tensor(shape)            — 创建 Identity Tensor

=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


# =============================================================================
# 第一部分：Composition 基础 — A ∘ B
# =============================================================================
# 通过一个具体例子来理解 composition

@cute.jit
def composition_basics():
    """
    演示 composition(A, B) = R，其中 R(c) = A(B(c))
    """

    cute.printf("=" * 60 + "\n")
    cute.printf("  Composition 基础：R = A ∘ B\n")
    cute.printf("=" * 60 + "\n\n")

    # ---- Layout A (基础布局) ----
    # A: (6, 2) : (8, 2)
    # 表示 6 行 2 列的数据，行步长=8，列步长=2
    # A(i, j) = i * 8 + j * 2
    A = cute.make_layout(shape=(6, 2), stride=(8, 2))
    cute.printf("Layout A (基础布局): {}\n", A)
    cute.printf("  A(0,0)=%d  A(0,1)=%d\n", cute.crd2idx((0, 0), A), cute.crd2idx((0, 1), A))
    cute.printf("  A(1,0)=%d  A(1,1)=%d\n", cute.crd2idx((1, 0), A), cute.crd2idx((1, 1), A))
    cute.printf("  A(2,0)=%d  A(2,1)=%d\n", cute.crd2idx((2, 0), A), cute.crd2idx((2, 1), A))

    # ---- Layout B (坐标变换器 / Tiler) ----
    # B: (4, 3) : (3, 1)  行优先
    # B(i, j) = i * 3 + j * 1
    # B 将 (4, 3) 空间映射到 0~11 的线性索引
    B = cute.make_layout(shape=(4, 3), stride=(3, 1))
    cute.printf("\nLayout B (坐标变换器): {}\n", B)
    cute.printf("  B(0,0)=%d  B(0,1)=%d  B(0,2)=%d\n",
                cute.crd2idx((0, 0), B), cute.crd2idx((0, 1), B), cute.crd2idx((0, 2), B))
    cute.printf("  B(1,0)=%d  B(1,1)=%d  B(1,2)=%d\n",
                cute.crd2idx((1, 0), B), cute.crd2idx((1, 1), B), cute.crd2idx((1, 2), B))

    # ---- Composition: R = A ∘ B ----
    # R(c) = A(B(c))
    # B 先把坐标 c 映射成线性索引 k
    # 然后 A 把 k 拆成 (k // 2, k % 2) 再映射成偏移
    R = cute.composition(A, B)
    cute.printf("\nR = composition(A, B): {}\n", R)
    cute.printf("R 的 shape 和 B 一样: {}\n", R.shape)

    # 验证：R(i, j) = A(B(i, j) // 2, B(i, j) % 2) = A(idx_in_A) = ...
    cute.printf("\n验证 R(c) = A(B(c)):\n")
    for i in cutlass.range_constexpr(4):
        for j in cutlass.range_constexpr(3):
            b_val = cute.crd2idx((i, j), B)
            # B(i,j) 是一个线性索引，CuTE 自动将其拆成 A 的多维坐标
            a_row = b_val // 2
            a_col = b_val % 2
            a_val = cute.crd2idx((a_row, a_col), A)
            r_val = cute.crd2idx((i, j), R)
            cute.printf("  R(%d,%d)=%d  [B(%d,%d)=%d -> A(%d,%d)=%d]\n",
                        i, j, r_val, i, j, b_val, a_row, a_col, a_val)


# =============================================================================
# 第二部分：Composition 的实际用途 — V 矩阵转置复用
# =============================================================================
# 在 FlashMLA (funny_cute/flashmla_dsl.py) 中，V 矩阵存储在 K 的 SMEM 空间里。
# 但 V 需要转置后使用（K 是 (page_size, head_dim)，V 需要 (head_dim, page_size)）。
#
# 解决方案：v_layout = composition(k_layout, transpose_layout)
# 不需要额外的 SMEM 空间，直接换个"看法"来访问同一块内存！

@cute.jit
def composition_transpose_demo():
    """
    演示用 composition 实现零拷贝转置
    """
    cute.printf("\n" + "=" * 60 + "\n")
    cute.printf("  用 Composition 实现零拷贝转置\n")
    cute.printf("=" * 60 + "\n\n")

    # K 矩阵的 SMEM layout: (4, 8) 行优先
    k_layout = cute.make_layout(shape=(4, 8), stride=(8, 1))
    cute.printf("K layout (4x8 行优先): {}\n", k_layout)

    # 转置 layout: (8, 4) 列优先
    # 这个 layout 把 (col, row) 坐标映射成 (row, col) 的线性索引
    transpose_layout = cute.make_ordered_layout((8, 4), order=(1, 0))
    cute.printf("Transpose layout (8x4 列优先): {}\n", transpose_layout)

    # V layout = composition(K layout, transpose layout)
    # V(i, j) = K_layout(transpose_layout(i, j))
    # = K_layout(j, i)  （效果上就是转置）
    v_layout = cute.composition(k_layout, transpose_layout)
    cute.printf("V layout = composition(K, transpose): {}\n\n", v_layout)

    # 验证：V(col, row) 应该等于 K(row, col) = row * 8 + col
    cute.printf("验证 V[col, row] == K[row, col]:\n")
    for col in cutlass.range_constexpr(4):  # V 的行（对应 K 的列）
        for row in cutlass.range_constexpr(4):  # V 的列（对应 K 的行）
            v_val = cute.crd2idx((col, row), v_layout)
            k_val = cute.crd2idx((row, col), k_layout)
            cute.printf("  V(%d,%d)=%2d  K(%d,%d)=%2d\n",
                        col, row, v_val, row, col, k_val)


# =============================================================================
# 第三部分：Identity Tensor — 坐标调试利器
# =============================================================================

@cute.jit
def identity_tensor_demo():
    """
    演示 make_identity_tensor 及其用途
    """
    cute.printf("\n" + "=" * 60 + "\n")
    cute.printf("  Identity Tensor — 每个位置存储自己的坐标\n")
    cute.printf("=" * 60 + "\n\n")

    # 创建 4×8 的 Identity Tensor
    shape = (4, 8)
    identity = cute.make_identity_tensor(shape)

    # 打印 identity tensor 的内容
    cute.printf("Identity Tensor (%d x %d):\n", shape[0], shape[1])
    for i in range(shape[0]):
        cute.printf("  row %d: ", i)
        for j in range(shape[1]):
            coord = identity[i, j]
            cute.printf("(%d,%d) ", coord[0], coord[1])
        cute.printf("\n")

    # ---- 用 Identity Tensor 验证 Tiling ----
    cute.printf("\n用 Identity Tensor 验证 2x2 Tiling:\n")
    cute.printf("将 4×8 矩阵分成 2×2 的 tile\n\n")

    tile_size = 2
    for tile_r in range(shape[0] // tile_size):
        for tile_c in range(shape[1] // tile_size):
            cute.printf("Tile (%d,%d):\n", tile_r, tile_c)
            for i in range(tile_size):
                cute.printf("  ")
                for j in range(tile_size):
                    row = tile_r * tile_size + i
                    col = tile_c * tile_size + j
                    coord = identity[row, col]
                    cute.printf("(%d,%d) ", coord[0], coord[1])
                cute.printf("\n")
            if tile_c < shape[1] // tile_size - 1:
                cute.printf("")
        cute.printf("\n")


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()

    # Part 1: Composition 基础
    composition_basics()
    torch.cuda.synchronize()

    # Part 2: Composition 实现转置
    composition_transpose_demo()
    torch.cuda.synchronize()

    # Part 3: Identity Tensor
    identity_tensor_demo()
    torch.cuda.synchronize()

    print("\n🎉 教程 11 全部完成！")
    print()
    print("总结：")
    print("  composition(A, B) = R, 其中 R(c) = A(B(c))")
    print("  - B 是坐标变换器，A 是地址映射器")
    print("  - Swizzle 就是 composition：swizzle ∘ layout")
    print("  - 零拷贝转置：composition(k_layout, transpose_layout)")
    print()
    print("  make_identity_tensor(shape):")
    print("  - 每个位置存储自己的坐标")
    print("  - 用于验证 tiling/partition 策略是否正确")
