"""
=============================================================================
教程 02: Layout（布局）—— CuTE 的灵魂抽象
=============================================================================

Layout 是 CuTE 中最核心的概念。理解 Layout 就理解了 CuTE 的一半。

什么是 Layout？
  Layout = (Shape, Stride)
  它定义了"逻辑坐标"到"线性内存偏移"的映射关系。

  offset = coord[0] * stride[0] + coord[1] * stride[1] + ...

举例：一个 4×3 的矩阵
  行优先（Row-major, C order）: shape=(4,3), stride=(3,1)
    → 同一行的元素在内存中连续
    → offset(i,j) = i*3 + j*1

  列优先（Col-major, Fortran order）: shape=(4,3), stride=(1,4)
    → 同一列的元素在内存中连续
    → offset(i,j) = i*1 + j*4

为什么 Layout 如此重要？
  1. 它统一了"数据在内存中怎么排列"的描述
  2. 它让 Tiling（分块）、Partition（分区）变成纯代数操作
  3. 它让同一份代码可以处理不同的内存布局（行优先/列优先/swizzle）
  4. 它是 CuTE 实现零拷贝转置、高效 Tensor Core 访问的基础

关键 API：
  cute.make_layout(shape, stride)     — 创建 Layout
  cute.make_ordered_layout(shape, order) — 按指定顺序创建 Layout
  cute.crd2idx(coord, layout)         — 坐标 → 线性偏移
  cute.idx2crd(idx, shape)            — 线性偏移 → 坐标
  cute.make_tensor(ptr, layout)       — 用 Layout 包装内存指针为 Tensor
  cute.make_fragment(layout, dtype)   — 在寄存器中创建 Tensor（用于 kernel 内部）
  cute.print_tensor(tensor)           — 打印 Tensor 内容（调试利器）
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch


# =============================================================================
# 第一部分：Layout 基础 —— Shape 和 Stride
# =============================================================================

@cute.jit
def layout_basics():
    """演示 Layout 的基本创建和坐标映射"""

    # --- 1. 行优先布局 ---
    # 3 行 4 列的矩阵，行优先存储
    # stride=(4,1) 表示：行间距为 4（跳过一整行），列间距为 1（连续）
    row_major = cute.make_layout(
        shape=(3, 4),       # 3 行 4 列
        stride=(4, 1)       # 行优先：行步长=列数，列步长=1
    )
    cute.printf("行优先 Layout: {}\n", row_major)

    # --- 2. 列优先布局 ---
    # 同样 3×4，但列优先存储
    # stride=(1,3) 表示：行间距为 1（连续），列间距为 3（跳过一整列）
    col_major = cute.make_layout(
        shape=(3, 4),       # 3 行 4 列
        stride=(1, 3)       # 列优先：行步长=1，列步长=行数
    )
    cute.printf("列优先 Layout: {}\n", col_major)

    # --- 3. 坐标到偏移的映射 ---
    # 对于坐标 (1, 2)（第 2 行第 3 列，0-indexed）：
    coord = (1, 2)
    row_offset = cute.crd2idx(coord, row_major)  # 1*4 + 2*1 = 6
    col_offset = cute.crd2idx(coord, col_major)  # 1*1 + 2*3 = 7
    cute.printf("坐标 (1,2) → 行优先偏移=%d, 列优先偏移=%d\n", row_offset, col_offset)

    # --- 4. 使用 make_ordered_layout 简化创建 ---
    # order 指定哪个维度在内存中变化最快
    # order=(1,0) → 维度 1（列）变化最快 → 行优先
    # order=(0,1) → 维度 0（行）变化最快 → 列优先
    row_major_v2 = cute.make_ordered_layout((3, 4), order=(1, 0))
    col_major_v2 = cute.make_ordered_layout((3, 4), order=(0, 1))
    cute.printf("make_ordered_layout 行优先: {}\n", row_major_v2)
    cute.printf("make_ordered_layout 列优先: {}\n", col_major_v2)


# =============================================================================
# 第二部分：用 Layout 创建 Tensor 并操作
# =============================================================================

@cute.jit
def layout_tensor_demo():
    """演示如何用 Layout 创建寄存器 Tensor 并进行读写"""

    # 创建一个 4×4 的行优先寄存器 Tensor
    layout = cute.make_layout((4, 4), stride=(4, 1))
    tensor = cute.make_fragment(layout, cutlass.Float32)

    # 按行优先顺序填充 0~15
    for i in range(4):
        for j in range(4):
            tensor[i, j] = cutlass.Float32(i * 4 + j)

    cute.printf("4x4 行优先 Tensor:\n")
    cute.print_tensor(tensor)

    # 用 idx2crd 做反向映射：线性索引 → 多维坐标
    # 例如线性索引 7 在 (4,4) shape 中对应坐标 (1,3)
    flat_idx = 7
    coord = cute.idx2crd(flat_idx, layout.shape)
    cute.printf("线性索引 %d → 坐标 (%d, %d)\n", flat_idx, coord[0], coord[1])


# =============================================================================
# 第三部分：Layout 实战 —— 矩阵转置
# =============================================================================
# 转置的本质：读用行优先 Layout，写用列优先 Layout（或反过来）
# 这就是 Layout 的威力 —— 转置不需要额外的数据搬运逻辑

M, K = 4, 8

@cute.kernel
def transpose_kernel(
    gA: cute.Tensor,    # 输入：M×K
    gB: cute.Tensor,    # 输出：K×M
    m: int,
    k: int,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()
    gid = bidx * bdimx + tidx

    # 将一维全局 ID 映射回二维坐标
    # 对于 M×K 的行优先矩阵：row = gid / K, col = gid % K
    if gid < m * k:
        row = gid // k
        col = gid % k
        # 转置：B[col, row] = A[row, col]
        gB[col * m + row] = gA[row * k + col]


@cute.jit
def transpose_host(mA: cute.Tensor, mB: cute.Tensor):
    m = M
    k = K

    total = m * k
    threads = 32
    blocks = (total + threads - 1) // threads

    transpose_kernel(mA, mB, m, k).launch(
        grid=(blocks, 1, 1),
        block=(threads, 1, 1)
    )


# =============================================================================
# 第四部分：层次化 Layout —— CuTE 的独特之处
# =============================================================================
# CuTE 的 Layout 支持嵌套（hierarchical）结构
# 例如 shape=((2,4), (3,2)) 表示一个 2 级层次的布局
# 这在 Tiling 和 MMA 分区中非常常见

@cute.jit
def hierarchical_layout_demo():
    """演示层次化 Layout"""

    # 一个简单的层次化 Layout
    # shape=((2,4), 3) 表示第一个维度被分成 2 组，每组 4 个元素
    # 总共 2*4*3 = 24 个元素
    hier_layout = cute.make_layout(
        shape=((2, 4), 3),
        stride=((1, 6), 2)
    )
    cute.printf("层次化 Layout: {}\n", hier_layout)

    # 3D Layout：常用于多 stage 的 SMEM buffer
    # shape=(num_stages, rows, cols)
    staged_layout = cute.make_layout(
        shape=(3, 8, 16),       # 3 个 stage，每个 8×16
        stride=(128, 16, 1)     # stage 间距=128，行优先
    )
    cute.printf("多 Stage Layout: {}\n", staged_layout)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()

    # ---- Part 1 ----
    print("=" * 60)
    print("第一部分：Layout 基础")
    print("=" * 60)
    layout_basics()
    torch.cuda.synchronize()

    # ---- Part 2 ----
    print("\n" + "=" * 60)
    print("第二部分：Layout + Tensor 操作")
    print("=" * 60)
    layout_tensor_demo()
    torch.cuda.synchronize()

    # ---- Part 3 ----
    print("\n" + "=" * 60)
    print("第三部分：矩阵转置")
    print("=" * 60)
    A_2d = torch.arange(M * K, dtype=torch.float16, device="cuda").reshape(M, K)
    print(f"输入 A ({M}×{K}):\n{A_2d}")

    # 关键：将 2D 张量 flatten 为 1D 传入 kernel
    # 因为我们的 kernel 用的是手动计算的 1D 索引
    A_flat = A_2d.contiguous().view(-1)
    B_flat = torch.empty(K * M, dtype=torch.float16, device="cuda")
    A_ = from_dlpack(A_flat, assumed_align=16)
    B_ = from_dlpack(B_flat, assumed_align=16)
    transpose_host(A_, B_)
    B_2d = B_flat.view(K, M)
    print(f"输出 B ({K}×{M}):\n{B_2d}")

    assert torch.allclose(B_2d, A_2d.T.contiguous()), "转置验证失败！"
    print("✅ 转置验证通过！")

    # ---- Part 4 ----
    print("\n" + "=" * 60)
    print("第四部分：层次化 Layout")
    print("=" * 60)
    hierarchical_layout_demo()
    torch.cuda.synchronize()

    print("\n🎉 教程 02 全部完成！")
