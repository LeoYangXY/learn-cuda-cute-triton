"""
=============================================================================
教程 01: CuTeDSL 基础 —— 线程模型与向量加法（向量化版 API 详解）
=============================================================================

核心概念：
  1. @cute.kernel — 标记一个 GPU kernel 函数（等价于 CUDA 的 __global__）
  2. @cute.jit    — 标记一个 host 函数（在 CPU 上运行，负责启动 kernel）
  3. cute.compile  — 将 @cute.jit 函数编译为可执行对象（JIT 编译）

本文件重点讲解 CuTE 的 TiledCopy 向量化拷贝机制中涉及的所有 API。

=============================================================================
API 总览（你已经懂的 vs 需要讲的）：
  ✅ 你已懂：make_layout, make_shape
  📖 本文详解：
    - local_tile      — CTA 级分块：从大张量中切出当前 Block 负责的那一块
    - make_copy_atom   — 定义最小拷贝操作（一个线程一次搬多少数据、用什么指令）
    - make_tiled_copy  — 将 copy_atom 扩展到整个 CTA（所有线程如何分工搬运）
    - get_slice        — 从 TiledCopy 中切出当前线程的视图
    - partition_S/D    — 将 tile 按 TiledCopy 模式分配给当前线程（S=源, D=目标）
    - make_fragment_like — 在寄存器中创建与某个 tensor 形状相同的 fragment
    - cute.copy        — 按 TiledCopy 的配置执行向量化拷贝（生成 LDG.128 等指令）
    - cute.size        — 返回 tensor/fragment 的元素总数

  📖 关于 "等号赋值" vs "cute.copy" 的区别：
    - gC[i] = gA[i] + gB[i]  →  编译器生成 LDG.32 + STG.32（每次 32 bits）
      这是标量操作，每条指令只搬 1 个 Float32（4 字节）
    - cute.copy(tiled_copy, src, dst)  →  编译器根据 copy_atom 的配置生成
      LDG.128 + STG.128（每次 128 bits = 4 个 Float32 = 16 字节）
      这是向量化操作，一条指令搬 4 个元素

    它们走的是同一个硬件单元（LSU，Load/Store Unit），但：
    - 标量版：发射 N 条 LDG.32 指令 → 指令发射带宽成为瓶颈
    - 向量版：发射 N/4 条 LDG.128 指令 → 指令数减少 4 倍，带宽利用率更高

    类比：搬砖，一次搬 1 块 vs 一次搬 4 块，走的是同一条路（LSU），
    但后者效率高 4 倍，因为"走路的次数"（指令数）少了。

    在 CUDA C++ 中你需要手动用 float4 来实现向量化：
      float4 a = *reinterpret_cast<float4*>(&A[i]);  // LDG.128
    在 CuTE 中，TiledCopy 帮你自动完成了这件事。
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

N = 1 << 20  # 向量长度

# =============================================================================
# 向量化向量加法（优化版）—— 每个 API 逐行详解
# =============================================================================
#
# 优化思路：使用 CuTE 的 TiledCopy 进行 128-bit 向量化加载/存储。
# 每个线程一次处理 4 个 Float32 元素（128 bits）。
# 这样减少了指令数，提高了内存带宽利用率。

VEC_SIZE = 4       # 每个线程每次拷贝处理 4 个元素
VEC_BLOCK_SIZE = 256  # 每个 Block 256 个线程


@cute.kernel
def vector_add_vec_kernel(
    gA: cute.Tensor,          # 输入 A：GMEM 中的 1D 张量，shape = (N,)
    gB: cute.Tensor,          # 输入 B：GMEM 中的 1D 张量，shape = (N,)
    gC: cute.Tensor,          # 输出 C：GMEM 中的 1D 张量，shape = (N,)
    tv_layout: cute.Layout,   # 线程-值 布局：描述线程如何映射到数据
    tiler_n: cute.Shape,      # CTA tile 大小：每个 Block 处理多少个元素
):
    tidx, _, _ = cute.arch.thread_idx()  # 线程在 Block 内的索引 (0~255)
    bidx, _, _ = cute.arch.block_idx()   # Block 在 Grid 内的索引

    # =========================================================================
    # API 1: cute.make_copy_atom — 定义最小拷贝操作（原子操作）
    # =========================================================================
    # "Atom"（原子）= 拷贝操作中最小的、不可分割的单元。
    # 它描述的是：一个线程在一条拷贝指令中做什么。
    #
    # 参数说明：
    #   - CopyUniversalOp()：拷贝指令类型。"Universal" = "通用"，
    #     表示最基础的 SIMT 拷贝方式——每个线程独立发射 load/store 指令，
    #     走 LSU（Load/Store Unit）硬件路径。
    #     根据 num_bits_per_copy 的值，编译器选择对应位宽的指令：
    #       32 bits → LDG.32 / STG.32
    #       64 bits → LDG.64 / STG.64
    #       128 bits → LDG.128 / STG.128
    #     注意：这不是 TMA！TMA 是另一条硬件路径（见下方说明）。
    #   - gA.element_type：数据类型 = Float32
    #   - num_bits_per_copy：每条指令拷贝多少 bit
    #     = 32（Float32 位宽）× 4（VEC_SIZE）= 128 bits
    #
    # 结果：这个 atom 表示"每个线程每条指令拷贝 128 bits（4 个 Float32）"
    # 编译器会为此生成一条 LDG.128 指令。
    #
    # 类比：Copy Atom 就像说"每个工人一次搬 4 块砖"
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),   # SIMT 通用拷贝（走 LSU，不是 TMA）
        gA.element_type,                 # Float32
        num_bits_per_copy=gA.element_type.width * VEC_SIZE,  # 32 × 4 = 128 bits
    )

    # =========================================================================
    # API 2: cute.make_tiled_copy — 将 Atom 扩展到整个 CTA
    # =========================================================================
    # Copy Atom 只描述了一个线程的行为。
    # make_tiled_copy 将 Atom 与线程布局组合，描述 CTA 内所有线程如何协作拷贝。
    #
    # 参数说明：
    #   - copy_atom：每个线程的拷贝行为（每次 128-bit）
    #   - tv_layout：线程-值 布局，shape=(256, 4)，stride=(4, 1)
    #     这是一个 2D 布局：
    #       第 0 维 = 线程维度（256 个线程）
    #       第 1 维 = 值维度（每个线程 4 个值）
    #     stride=(4, 1) 的含义：
    #       线程 0 → 元素 [0, 1, 2, 3]   （偏移 0，线程间步长为 4）
    #       线程 1 → 元素 [4, 5, 6, 7]   （偏移 4）
    #       线程 2 → 元素 [8, 9, 10, 11] （偏移 8）
    #       ...
    #       线程 255 → 元素 [1020, 1021, 1022, 1023]
    #     每个线程的 4 个元素在内存中是连续的（stride=1），
    #     所以可以用一条 128-bit 指令加载。
    #   - tiler_n：总 tile 大小 = (1024,)
    #     256 个线程 × 4 个元素 = 每个 CTA 处理 1024 个元素
    #
    # 结果：TiledCopy = "256 个线程，每个加载 4 个连续的 Float32"
    #
    # 类比：Copy Atom = "一个工人搬 4 块砖"
    #        TiledCopy = "256 个工人，每人搬 4 块砖，总共覆盖 1024 块砖"
    #
    # 在 GEMM 中，等价的操作是 make_tiled_mma（Atom → CTA 级 MMA 模式）。
    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_n)

    # =========================================================================
    # API 3: tiled_copy.get_slice(tidx) — 获取当前线程的拷贝视图
    # =========================================================================
    # TiledCopy 描述的是整个 CTA 的拷贝模式。
    # get_slice(tidx) 从中提取出当前这个线程负责的部分。
    #
    # 例如，如果 tidx=5：
    #   thr_copy 知道"我负责元素 [20, 21, 22, 23]"
    #
    # 这类似于 GEMM 中的 tiled_mma.get_slice(tidx)，
    # 它告诉每个线程自己负责 MMA 的哪一部分。
    thr_copy = tiled_copy.get_slice(tidx)

    # =========================================================================
    # API 4: cute.local_tile — CTA 级分块（Block 级别的切分）
    # =========================================================================
    # 这是 CuTE 层次化分解的第一层：
    #   全局张量 → CTA tile → 线程分区 → 寄存器 fragment
    #
    # local_tile(tensor, tiler, coord) 将全局张量按 tiler 大小切分，
    # 返回位于 coord 位置的那个 tile。
    #
    # 这里：gA 的 shape 是 (N,)，tiler_n = (1024,)，coord = (bidx,)
    #
    # 可视化（N=4096, tiler_n=(1024,)）：
    #   gA: [0 .................. 4095]
    #        |--- Block 0 ---|--- Block 1 ---|--- Block 2 ---|--- Block 3 ---|
    #        |  1024 个元素   |  1024 个元素   |  1024 个元素   |  1024 个元素   |
    #
    #   blkA = gA[bidx*1024 : (bidx+1)*1024]  （逻辑上等价）
    #
    # 对于 2D 张量（如 GEMM 中），local_tile 的工作方式类似：
    #   gA_tile = local_tile(gA, (BM, BN, BK), (bidx, bidy, None), proj=(1,None,1))
    #   这将 A[M,K] 切成 (BM, BK) 大小的 tile，选择第 bidx 个 M-tile，
    #   K 维度留为 None（在循环中遍历）。
    #
    # proj 参数（这里没用到，但 GEMM 中会用到）：
    #   proj=(1, None, 1) 表示"将 tiler 的第 0 和第 2 维应用到这个张量，
    #   跳过第 1 维"。这是因为 A[M,K] 没有 N 维度。
    #
    # 你画的图是正确的：
    #   对于 shape(M,N) 的 2D 张量，local_tile 用 tile(2,2) 和 coord(1,1)
    #   选出第 (1,1) 个 2×2 的子块。
    #   结果是一个视图（view），不是拷贝，指针直接指向那块内存区域。
    blkA = cute.local_tile(gA, tiler_n, (bidx,))  # shape: (1024,)
    blkB = cute.local_tile(gB, tiler_n, (bidx,))  # shape: (1024,)
    blkC = cute.local_tile(gC, tiler_n, (bidx,))  # shape: (1024,)

    # =========================================================================
    # API 5: thr_copy.partition_S / partition_D — 线程级分区
    # =========================================================================
    # 这是层次化分解的第二层：
    #   local_tile 给了我们 CTA 的 tile（1024 个元素）之后，
    #   partition 进一步将它分配给各个线程。
    #
    # partition_S(tensor)：按源（Source）分区（从这里读取）
    # partition_D(tensor)：按目标（Destination）分区（往这里写入）
    #
    # 对于线程 5，tv_layout stride=(4,1)：
    #   partition_S(blkA) → blkA 中元素 [20, 21, 22, 23] 的视图
    #   （因为线程 5 × 每线程 4 个值 = 偏移 20）
    #
    # 可视化：
    #   blkA（1024 个元素）：
    #     线程 0:   [0,   1,   2,   3  ]  ← partition_S 返回这个视图
    #     线程 1:   [4,   5,   6,   7  ]
    #     线程 2:   [8,   9,   10,  11 ]
    #     ...
    #     线程 255: [1020, 1021, 1022, 1023]
    #
    # 为什么 S 和 D 要分开？
    #   在这个简单例子中，S 和 D 的布局相同。
    #   但在 GEMM 中使用 SMEM 时，它们可能不同：
    #     partition_S(gA)  → 源是 GMEM（行优先布局）
    #     partition_D(sA)  → 目标是 SMEM（可能是 swizzle 布局）
    #   TiledCopy 会自动处理布局转换。
    #
    # 与 local_partition 的对比（教程 03 中使用）：
    #   local_partition(tensor, thread_layout, thread_idx) 做的事情一样，
    #   但使用独立的线程布局，不绑定到 TiledCopy。
    #   partition_S/D 是 TiledCopy 感知的版本，确保分区方式
    #   与拷贝指令的要求匹配。
    tAsA = thr_copy.partition_S(blkA)  # A 的源分区（GMEM → 将被读取）
    tBsB = thr_copy.partition_S(blkB)  # B 的源分区（GMEM → 将被读取）
    tCsC = thr_copy.partition_D(blkC)  # C 的目标分区（GMEM → 将被写入）

    # =========================================================================
    # API 6: cute.make_fragment_like — 创建寄存器级张量（Fragment）
    # =========================================================================
    # "fragment" 是存在于寄存器（REGISTER）中的张量（不在 GMEM，也不在 SMEM）。
    # 寄存器是 GPU 上最快的存储（~0 周期延迟）。
    #
    # make_fragment_like(tensor) 创建一个与输入张量形状相同的寄存器张量。
    # 对于线程 0，tAsA 有 4 个元素，所以 rA 是 4 个 Float32 寄存器。
    #
    # 数据流：
    #   GMEM (tAsA) --拷贝--> 寄存器 (rA) --计算--> 寄存器 (rC) --拷贝--> GMEM (tCsC)
    #
    # 为什么不直接在 GMEM 上计算？
    #   因为 GMEM 访问延迟约 400 个时钟周期。先加载到寄存器，
    #   ALU 就可以在快速的本地数据上运算。
    rA = cute.make_fragment_like(tAsA)  # A 的 4 个 Float32 寄存器
    rB = cute.make_fragment_like(tBsB)  # B 的 4 个 Float32 寄存器

    # =========================================================================
    # API 7: cute.copy(tiled_copy, src, dst) — 向量化拷贝操作
    # =========================================================================
    # 这是与简单赋值（=）的关键区别。
    #
    # cute.copy 使用 TiledCopy 的配置来生成向量化硬件指令。
    # 因为我们的 copy_atom 指定了每次拷贝 128 bits，编译器会生成：
    #   LDG.128（一条指令加载 128 bits = 4 个 Float32）
    # 而不是：
    #   4 × LDG.32（每条加载 32 bits，共 4 条独立指令）
    #
    # ┌─────────────────────────────────────────────────────────────┐
    # │  "=" 赋值（标量）：                                          │
    # │    gC[i] = gA[i]                                            │
    # │    → LDG.32 r1, [addr]     （1 条指令，搬 4 字节）           │
    # │    → STG.32 [addr], r1     （1 条指令，搬 4 字节）           │
    # │    合计：1 个元素需要 2 条指令                                │
    # │                                                             │
    # │  cute.copy（向量化）：                                       │
    # │    cute.copy(tiled_copy, src, dst)                          │
    # │    → LDG.128 r1, [addr]   （1 条指令，搬 16 字节）           │
    # │    → STG.128 [addr], r1   （1 条指令，搬 16 字节）           │
    # │    合计：4 个元素只需 2 条指令                                │
    # │                                                             │
    # │  同一个硬件单元（LSU），但指令数减少 4 倍！                    │
    # │  这很重要，因为 GPU 的指令吞吐量是有限的。                    │
    # └─────────────────────────────────────────────────────────────┘
    #
    # 在 CUDA C++ 中，你需要这样写：
    #   float4 val = *reinterpret_cast<float4*>(&A[offset]);  // LDG.128
    # CuTE 的 cute.copy 根据 copy_atom 的配置自动完成了这件事。
    #
    # 注意：cute.copy 也可以只传 copy_atom（不传 tiled_copy）：
    #   cute.copy(atom_store, src, dst)
    # 这在 GEMM 的 epilogue（写回阶段）中用于将结果写回 GMEM。
    cute.copy(tiled_copy, tAsA, rA)  # GMEM → 寄存器（LDG.128）
    cute.copy(tiled_copy, tBsB, rB)  # GMEM → 寄存器（LDG.128）

    # =========================================================================
    # 计算：在寄存器中做逐元素加法
    # =========================================================================
    # cute.size(tensor) 返回 tensor/fragment 的元素总数。
    # 在我们的例子中，cute.size(rC) = 4（每个线程有 4 个元素）。
    #
    # 这个循环完全在寄存器中运行 —— 没有内存访问，纯 ALU 运算。
    # 编译器很可能会将其展开为 4 条 FADD 指令。
    rC = cute.make_fragment_like(rA)   # 结果的 4 个 Float32 寄存器
    for i in range(cute.size(rC)):     # cute.size(rC) = 4
        rC[i] = rA[i] + rB[i]         # 寄存器到寄存器的加法（FADD）

    # =========================================================================
    # API 7（再次）：cute.copy 用于存储 — 寄存器 → GMEM（STG.128）
    # =========================================================================
    # 同样的 API，但方向反过来：寄存器 fragment → GMEM。
    # tiled_copy 知道布局映射关系，所以会生成 STG.128。
    cute.copy(tiled_copy, rC, tCsC)    # 寄存器 → GMEM（STG.128）


@cute.jit
def vector_add_vectorized(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    """
    Host 函数：配置并启动向量化 kernel。

    这个函数在 CPU 上运行，负责设置 TiledCopy 的参数。
    """
    n = mA.shape[0]
    elems_per_block = VEC_BLOCK_SIZE * VEC_SIZE  # 256 × 4 = 每个 Block 处理 1024 个元素
    num_blocks = n // elems_per_block             # N / 1024 个 Block

    # =========================================================================
    # tv_layout：线程-值 布局 — 线程到数据映射的"蓝图"
    # =========================================================================
    # 这个 2D 布局的 shape=(256, 4)，stride=(4, 1)：
    #   - 第 0 维（线程）：256 个线程，stride=4
    #     → 线程 0 从偏移 0 开始，线程 1 从偏移 4 开始，线程 2 从偏移 8 开始...
    #   - 第 1 维（值）：每个线程 4 个值，stride=1
    #     → 每个线程内部的元素是连续的：[0,1,2,3], [4,5,6,7], ...
    #
    # 这个布局保证了每个线程的 4 个元素在内存中是连续的，
    # 这是 128-bit 向量化加载（LDG.128 需要对齐的连续数据）的必要条件。
    #
    # 反面例子（不利于向量化的布局）：
    #   stride=(1, 256) 会让线程 0 → [0, 256, 512, 768]
    #   这些元素不连续，所以无法使用 LDG.128！
    #   （这种交错布局在其他场景有用，比如避免 SMEM bank conflict）
    tv_layout = cute.make_layout(
        (VEC_BLOCK_SIZE, VEC_SIZE),   # shape = (256, 4)
        stride=(VEC_SIZE, 1)          # stride = (4, 1) → 每个线程内连续
    )

    # tiler_n：CTA tile 大小 = 每个 Block 处理的总元素数
    tiler_n = (elems_per_block,)  # (1024,)

    vector_add_vec_kernel(mA, mB, mC, tv_layout, tiler_n).launch(
        grid=(num_blocks, 1, 1),       # N/1024 个 Block
        block=(VEC_BLOCK_SIZE, 1, 1)   # 每个 Block 256 个线程
    )


# =============================================================================
# 完整数据流图：
# =============================================================================
#
#  ┌──────────────────────────────────────────────────────────────────────┐
#  │  GMEM: gA[N]  （例如 N = 1M 个元素）                                │
#  │  ┌────────────────────────────────┐                                  │
#  │  │ Block 0: 1024 个元素            │  ← cute.local_tile(gA, (1024,)) │
#  │  │ ┌──────┬──────┬──────┬───────┐ │                                  │
#  │  │ │ T0:4 │ T1:4 │ T2:4 │ ...  │ │  ← thr_copy.partition_S         │
#  │  │ └──┬───┴──┬───┴──────┴───────┘ │                                  │
#  │  └────┼──────┼────────────────────┘                                  │
#  │       │      │  cute.copy（LDG.128，每个线程一条指令）                │
#  │       ▼      ▼                                                       │
#  │  ┌──────────────────┐                                                │
#  │  │ 寄存器            │  rA[4], rB[4]  ← make_fragment_like           │
#  │  │ rC[i] = rA[i]    │  逐元素加法（FADD，纯 ALU 运算）              │
#  │  │       + rB[i]    │                                                │
#  │  └────────┬─────────┘                                                │
#  │           │  cute.copy（STG.128，每个线程一条指令）                   │
#  │           ▼                                                          │
#  │  GMEM: gC[N]                                                         │
#  └──────────────────────────────────────────────────────────────────────┘
#
# =============================================================================
# API 层次总结（从粗到细）：
# =============================================================================
#
#  第 1 层：Grid 启动
#    grid=(num_blocks,) → 每个 Block 处理 1024 个元素
#
#  第 2 层：local_tile（CTA 级）
#    blkA = local_tile(gA, (1024,), (bidx,))
#    → 将全局张量切成 1024 元素的 tile，选择第 bidx 个 Block 的 tile
#
#  第 3 层：TiledCopy + partition_S/D（线程级）
#    thr_copy = tiled_copy.get_slice(tidx)
#    tAsA = thr_copy.partition_S(blkA)
#    → 将 1024 元素的 tile 进一步分配给 256 个线程（每个 4 个）
#
#  第 4 层：make_fragment_like（寄存器级）
#    rA = make_fragment_like(tAsA)
#    → 创建与线程分区形状匹配的寄存器存储
#
#  第 5 层：cute.copy（硬件指令级）
#    cute.copy(tiled_copy, tAsA, rA)
#    → 根据 copy_atom 的配置生成 LDG.128 / STG.128 指令
#
# =============================================================================


# =============================================================================
# 主函数：运行向量化向量加法
# =============================================================================

# =============================================================================
# TMA 版本的向量加法 —— 体验 TMA 硬件搬运
# =============================================================================
#
# 数据流对比：
#   LSU 版（上面的 vector_add_vec_kernel）：
#     GMEM --[LDG.128, 每个线程]--> 寄存器 --[计算]--> 寄存器 --[STG.128]--> GMEM
#
#   TMA 版（下面的 vector_add_tma_kernel）：
#     GMEM --[TMA硬件, 1条指令]--> SMEM --[LDS.128, 每个线程]--> 寄存器 --[计算]--> 寄存器 --[STG.128]--> GMEM
#
# 注意：对于 vector_add 这种简单场景，TMA 版本不会更快（反而多了 SMEM 中转）。
# 这里纯粹是为了体验 TMA 的写法和工作原理。
# TMA 的真正优势在 GEMM 等有数据复用的场景。
#
# TMA 的限制：
#   1. TMA 只能搬到 SMEM，不能直接到寄存器
#   2. TMA 需要 tensor descriptor（在 host 端创建）
#   3. TMA 需要 mbarrier 来同步（知道数据什么时候搬完）
#   4. make_tiled_tma_atom 需要 2D 的 SMEM layout，所以我们把 1D 向量 reshape 成 2D
# =============================================================================

# TMA tile 参数
# 我们把 1D 向量 reshape 成 2D: (N_ROWS, N_COLS)
# 每个 Block 处理一个 (TMA_TILE_ROWS, TMA_TILE_COLS) 的 tile
TMA_TILE_ROWS = 1        # 每个 tile 1 行（简化：每个 Block 处理 1 行）
TMA_TILE_COLS = 1024     # 每个 tile 1024 列
TMA_BLOCK_SIZE = 256     # 每个 Block 256 个线程
TMA_VEC = 4              # 每个线程从 SMEM 加载 4 个 Float32（128 bits）
# 每个 Block 处理的元素数 = TMA_TILE_ROWS * TMA_TILE_COLS = 1024
# 256 个线程 × 4 个元素 = 1024 个元素


@cute.kernel
def vector_add_tma_kernel(
    # ---- TMA 相关参数（在 host 端创建，传入 kernel）----
    tma_atom_a: cute.CopyAtom,    # A 的 TMA 描述符（包含 GMEM 地址、形状、SMEM 布局等信息）
    mA_tma: cute.Tensor,          # A 的 TMA 坐标张量（不是数据本身，是 TMA 用来定位 tile 的坐标）
    tma_atom_b: cute.CopyAtom,    # B 的 TMA 描述符
    mB_tma: cute.Tensor,          # B 的 TMA 坐标张量
    # ---- 普通参数 ----
    mC: cute.Tensor,              # 输出 C（2D 视图）
    a_smem_layout: cute.Layout,   # SMEM 布局（2D）
    shared_storage: cutlass.Constexpr,  # SMEM 分配结构体的类型
    tv_layout: cute.Layout,       # 线程-值 布局（用于 SMEM → 寄存器 和 寄存器 → GMEM 的拷贝）
    tiler_mn: cute.Shape,         # CTA tile 大小 = (TMA_TILE_ROWS, TMA_TILE_COLS)
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # =========================================================================
    # 步骤 1: TMA 描述符预取
    # =========================================================================
    # TMA 描述符存在 L2 cache 中。预取可以减少首次 TMA 拷贝的延迟。
    # 只需要一个 warp 做这件事。
    if warp_idx == 0:
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

    # =========================================================================
    # 步骤 2: SMEM 分配
    # =========================================================================
    # 分配两块 SMEM：sA 和 sB，用于存放 TMA 搬运过来的数据。
    # storage 是一个结构体，包含 mbarrier 和两块 SMEM buffer。
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage)
    sA = storage.sA.get_tensor(a_smem_layout)  # SMEM 中的 A tile，shape = (1, 1024)
    sB = storage.sB.get_tensor(a_smem_layout)  # SMEM 中的 B tile，shape = (1, 1024)

    # =========================================================================
    # 步骤 3: CTA 级分块（与 LSU 版本相同的概念）
    # =========================================================================
    # 将全局 2D 张量按 tile 大小切分，选择当前 Block 负责的 tile。
    # bidx → 行方向的 tile 索引（每行 = 1024 个元素 = 1 个 Block）
    # bidy = 0（列方向只有 1 个 tile，因为 n_cols == TMA_TILE_COLS）
    gA = cute.local_tile(mA_tma, (TMA_TILE_ROWS, TMA_TILE_COLS), (bidx, bidy))  # shape: (1, 1024)
    gB = cute.local_tile(mB_tma, (TMA_TILE_ROWS, TMA_TILE_COLS), (bidx, bidy))  # shape: (1, 1024)
    gC = cute.local_tile(mC, (TMA_TILE_ROWS, TMA_TILE_COLS), (bidx, bidy))       # shape: (1, 1024)

    # =========================================================================
    # 步骤 4: TMA 分区 —— 告诉 TMA 硬件"搬到 SMEM 的哪里"
    # =========================================================================
    # tma_partition 将 SMEM tensor 和 GMEM tensor 按 TMA 的要求进行分区。
    #
    # group_modes(sA, 0, 2)：将 sA 的前两个维度合并成一个维度。
    #   sA 原本是 (1, 1024)，合并后变成 (1024,)。
    #   这是因为 TMA 把整个 tile 当作一个整体来搬运。
    #
    # 参数说明：
    #   - tma_atom_a：TMA 描述符
    #   - cta_coord=0：当前 CTA 在 cluster 中的坐标（我们不用 cluster，所以是 0）
    #   - cta_layout=cute.make_layout(1)：cluster 布局（单个 CTA）
    #   - smem_tensor：目标 SMEM（合并维度后）
    #   - gmem_tensor：源 GMEM（合并维度后）
    #
    # 返回：
    #   - tAsA：SMEM 端的分区视图（TMA 写入的目标）
    #   - tAgA：GMEM 端的分区视图（TMA 读取的源）
    sA_tma = cute.group_modes(sA, 0, 2)
    gA_tma = cute.group_modes(gA, 0, 2)
    tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        cta_coord=0,
        cta_layout=cute.make_layout(1),
        smem_tensor=sA_tma,
        gmem_tensor=gA_tma,
    )

    sB_tma = cute.group_modes(sB, 0, 2)
    gB_tma = cute.group_modes(gB, 0, 2)
    tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        cta_coord=0,
        cta_layout=cute.make_layout(1),
        smem_tensor=sB_tma,
        gmem_tensor=gB_tma,
    )

    # =========================================================================
    # 步骤 5: 初始化 mbarrier —— TMA 的同步机制
    # =========================================================================
    # mbarrier（Memory Barrier）是 Hopper/Blackwell 的硬件同步原语。
    # 它有两个计数器：
    #   1. 到达计数（arrival count）：有多少个"参与者"已经到达
    #   2. 期望字节数（expected TX bytes）：TMA 还需要搬运多少字节
    #
    # 当两个计数器都归零时，barrier 被释放，等待的线程可以继续。
    #
    # cnt=1 表示只有 1 个"参与者"（我们只用 1 个线程发起 TMA）。
    # elect_one() 确保只有 warp 中的一个线程执行初始化。
    barrier_ptr = storage.barrier_ptr.data_ptr()
    with cute.arch.elect_one():
        cute.arch.mbarrier_init(barrier_ptr, cnt=1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()  # 确保所有线程看到 mbarrier 初始化完成

    # =========================================================================
    # 步骤 6: TMA 拷贝 —— 硬件自动搬运 GMEM → SMEM
    # =========================================================================
    # 这是 TMA 的核心！
    #
    # 与 LSU 版本的关键区别：
    #   LSU：256 个线程各自发射 LDG.128 → 256 条指令
    #   TMA：1 个 warp 发射 1 条 cp.async.bulk.tensor → 1 条指令搬整个 tile
    #
    # 注意事项：
    #   1. cute.copy 必须由整个 warp（32 个线程）调用，不能只用 1 个线程
    #      （CuTeDSL 内部会选择 warp 中的某个线程来实际发射 PTX 指令）
    #   2. mbarrier_arrive_and_expect_tx 只需要 1 个线程调用
    #      它告诉 mbarrier："还有 tma_copy_bytes 字节需要搬运"
    #   3. TMA 搬运是异步的 —— cute.copy 返回后数据还没到 SMEM！
    #      必须用 mbarrier_wait 等待。
    tma_copy_bytes = cute.size_in_bytes(cutlass.Float32, a_smem_layout)

    # 搬运 A
    if warp_idx == 0:
        with cute.arch.elect_one():
            # 告诉 mbarrier：期望收到 2 * tma_copy_bytes 字节（A 和 B 各一份）
            cute.arch.mbarrier_arrive_and_expect_tx(
                barrier_ptr,
                tma_copy_bytes * 2,  # A + B 两次 TMA 的总字节数
            )
        # 发射 TMA 拷贝指令（异步，立即返回）
        cute.copy(tma_atom_a, tAgA, tAsA, tma_bar_ptr=barrier_ptr)
        cute.copy(tma_atom_b, tBgB, tBsB, tma_bar_ptr=barrier_ptr)

    # =========================================================================
    # 步骤 7: 等待 TMA 完成
    # =========================================================================
    # 所有线程在这里阻塞，直到 TMA 硬件搬完数据。
    # phase=0 是 mbarrier 的相位位（用于多 stage 流水线轮转，这里只有 1 个 stage）。
    cute.arch.mbarrier_wait(barrier_ptr, 0)

    # =========================================================================
    # 步骤 8: 从 SMEM 读到寄存器，计算，写回 GMEM
    # =========================================================================
    # 现在 sA 和 sB 中已经有数据了（TMA 搬过来的）。
    # 接下来用 LSU（CopyUniversalOp）从 SMEM 读到寄存器，做加法，再写回 GMEM。
    #
    # 这部分和 LSU 版本类似，只是源从 GMEM 变成了 SMEM。
    copy_atom_s2r = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        cutlass.Float32,
        num_bits_per_copy=128,  # 128 bits = 4 个 Float32
    )
    tiled_copy_s2r = cute.make_tiled_copy(copy_atom_s2r, tv_layout, tiler_mn)
    thr_copy = tiled_copy_s2r.get_slice(tidx)

    # 从 SMEM 分区 —— 注意这里要用 1D 视图
    # sA 是 2D 的 (1, 1024)，我们需要把它展平成 1D 的 (1024,)
    # 因为 tiled_copy_s2r 的 tiler 是 1D 的 (1024,)
    sA_flat = cute.make_tensor(sA.iterator, cute.make_layout((TMA_TILE_ROWS * TMA_TILE_COLS,)))
    sB_flat = cute.make_tensor(sB.iterator, cute.make_layout((TMA_TILE_ROWS * TMA_TILE_COLS,)))
    gC_flat = cute.make_tensor(gC.iterator, cute.make_layout((TMA_TILE_ROWS * TMA_TILE_COLS,)))
    tAsA_s2r = thr_copy.partition_S(sA_flat)  # 当前线程从 sA 读取的部分
    tBsB_s2r = thr_copy.partition_S(sB_flat)  # 当前线程从 sB 读取的部分
    tCgC_r2g = thr_copy.partition_D(gC_flat)  # 当前线程写入 gC 的部分

    # SMEM → 寄存器（LDS.128）
    rA = cute.make_fragment_like(tAsA_s2r)
    rB = cute.make_fragment_like(tBsB_s2r)
    cute.copy(tiled_copy_s2r, tAsA_s2r, rA)
    cute.copy(tiled_copy_s2r, tBsB_s2r, rB)

    # 寄存器中计算
    rC = cute.make_fragment_like(rA)
    for i in range(cute.size(rC)):
        rC[i] = rA[i] + rB[i]

    # 寄存器 → GMEM（STG.128）
    cute.copy(tiled_copy_s2r, rC, tCgC_r2g)


@cute.jit
def vector_add_tma(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    """
    Host 函数：配置 TMA 描述符并启动 TMA 版 kernel。

    TMA 的设置分两步：
      1. Host 端：创建 TMA 描述符（cuTensorMap），编码 GMEM tensor 的形状、步长、SMEM 布局
      2. Kernel 端：用描述符发射 TMA 拷贝指令
    """
    n = mA.shape[0]

    # ---- 将 1D 向量 reshape 成 2D ----
    # TMA 的 make_tiled_tma_atom 需要 2D 的 SMEM layout 和 tile shape。
    # 所以我们把 (N,) 的向量看作 (N_ROWS, N_COLS) 的 2D 张量。
    n_cols = TMA_TILE_COLS  # 1024
    n_rows = n // n_cols    # N / 1024

    # 将 1D tensor reshape 成 2D（行优先）
    # mA 原本是 (N,) stride=(1,)
    # reshape 后是 (n_rows, n_cols) stride=(n_cols, 1)
    mA_2d = cute.make_tensor(
        mA.iterator,
        cute.make_layout((n_rows, n_cols), stride=(n_cols, 1))
    )
    mB_2d = cute.make_tensor(
        mB.iterator,
        cute.make_layout((n_rows, n_cols), stride=(n_cols, 1))
    )
    mC_2d = cute.make_tensor(
        mC.iterator,
        cute.make_layout((n_rows, n_cols), stride=(n_cols, 1))
    )

    # ---- SMEM 布局 ----
    # 简单的行优先 2D 布局，与 GMEM 一致
    # 在 GEMM 中通常会用 swizzle 来消除 bank conflict，
    # 但 vector_add 不需要（每个元素只读一次，没有 bank conflict 问题）
    a_smem_layout = cute.make_ordered_layout(
        (TMA_TILE_ROWS, TMA_TILE_COLS),  # (1, 1024)
        order=(1, 0)  # 行优先（列是连续的）
    )

    # ---- 创建 TMA 描述符 ----
    # make_tiled_tma_atom 在 host 端创建 cuTensorMap 描述符。
    # 这个描述符编码了：
    #   - GMEM tensor 的基地址、形状、步长
    #   - SMEM 的布局（包括 swizzle 模式）
    #   - 每次搬运的 tile 大小
    #
    # 参数说明：
    #   - CopyBulkTensorTileG2SOp()：TMA 方向 = GMEM → SMEM
    #   - mA_2d：源 GMEM tensor（2D 视图）
    #   - a_smem_layout：目标 SMEM 布局
    #   - (TMA_TILE_ROWS, TMA_TILE_COLS)：每次搬运的 tile 大小
    #
    # 返回：
    #   - tma_atom_a：TMA CopyAtom（传入 kernel）
    #   - tma_tensor_a：TMA 坐标张量（不是数据，是 TMA 用来定位 tile 的坐标系统）
    tma_atom_a, tma_tensor_a = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),  # GMEM → SMEM
        mA_2d,              # 源 GMEM tensor
        a_smem_layout,      # 目标 SMEM 布局
        (TMA_TILE_ROWS, TMA_TILE_COLS),  # tile 大小
    )

    tma_atom_b, tma_tensor_b = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
        mB_2d,
        a_smem_layout,
        (TMA_TILE_ROWS, TMA_TILE_COLS),
    )

    # ---- SMEM 结构体 ----
    # 定义 kernel 需要的共享内存布局：
    #   - barrier_ptr：mbarrier 用的内存（8 字节 × 2）
    #   - sA：A 的 SMEM buffer
    #   - sB：B 的 SMEM buffer
    buffer_align_bytes = 128  # TMA 要求 128 字节对齐

    @cute.struct
    class SharedStorage:
        barrier_ptr: cute.struct.MemRange[cutlass.Int64, 2]
        sA: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, cute.cosize(a_smem_layout)],
            buffer_align_bytes,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, cute.cosize(a_smem_layout)],
            buffer_align_bytes,
        ]

    # ---- 线程-值 布局（用于 SMEM → 寄存器 的拷贝）----
    # 256 个线程，每个线程处理 4 个元素（128 bits）
    # tv_layout 是 1D 的，因为我们的 tile 实际上是 1 行 1024 列
    tv_layout = cute.make_layout(
        (TMA_BLOCK_SIZE, TMA_VEC),  # (256, 4)
        stride=(TMA_VEC, 1)         # 每个线程内连续
    )
    tiler_mn = (TMA_TILE_COLS,)  # 1D tiler = (1024,)

    # ---- Grid 配置 ----
    grid_rows = n_rows // TMA_TILE_ROWS  # 行方向的 Block 数
    grid_cols = n_cols // TMA_TILE_COLS  # 列方向的 Block 数（= 1，因为 n_cols == TMA_TILE_COLS）

    vector_add_tma_kernel(
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        mC_2d,
        a_smem_layout,
        SharedStorage,
        tv_layout,
        tiler_mn,
    ).launch(
        grid=(grid_rows, grid_cols, 1),
        block=(TMA_BLOCK_SIZE, 1, 1),
    )


# =============================================================================
# TMA 版数据流图：
# =============================================================================
#
#  ┌──────────────────────────────────────────────────────────────────────────┐
    #  Host 端（CPU）：                                                        │
    #  │    1. 将 1D 向量 reshape 成 2D: (N,) → (N/1024, 1024)                   │#  │    2. 创建 TMA 描述符: make_tiled_tma_atom(G2SOp, tensor, smem_layout)  │
#  │    3. 定义 SharedStorage 结构体（mbarrier + sA + sB）                    │
#  │    4. 启动 kernel                                                        │
#  └──────────────────────────────────────────────────────────────────────────┘
#                                    │
#                                    ▼
#  ┌──────────────────────────────────────────────────────────────────────────┐
#  │  Kernel 端（GPU）：                                                      │
#  │                                                                          │
#  │  1. 预取 TMA 描述符（减少首次延迟）                                       │
#  │  2. 分配 SMEM（sA, sB）                                                  │
#  │  3. 初始化 mbarrier                                                      │
#  │                                                                          │
#  │  4. TMA 拷贝（只需 1 个 warp）：                                         │
#  │     ┌─────────┐  cp.async.bulk.tensor  ┌──────────┐                     │
#  │     │  GMEM A │ ─────────────────────→ │  SMEM sA │  1 条指令搬整个 tile │
#  │     │  GMEM B │ ─────────────────────→ │  SMEM sB │  1 条指令搬整个 tile │
#  │     └─────────┘                        └──────────┘                     │
#  │                                                                          │
#  │  5. mbarrier_wait（所有线程等 TMA 搬完）                                  │
#  │                                                                          │
#  │  6. SMEM → 寄存器（每个线程用 LDS.128 从 SMEM 读）                       │
#  │     ┌──────────┐  LDS.128   ┌──────────┐                               │
#  │     │  SMEM sA │ ────────→  │ 寄存器 rA │  128 个线程各读各的            │
#  │     │  SMEM sB │ ────────→  │ 寄存器 rB │                               │
#  │     └──────────┘            └──────────┘                               │
#  │                                                                          │
#  │  7. 寄存器中计算：rC[i] = rA[i] + rB[i]                                 │
#  │                                                                          │
#  │  8. 寄存器 → GMEM（每个线程用 STG.128 写回）                             │
#  │     ┌──────────┐  STG.128   ┌──────────┐                               │
#  │     │ 寄存器 rC │ ────────→  │  GMEM C  │                               │
#  │     └──────────┘            └──────────┘                               │
#  └──────────────────────────────────────────────────────────────────────────┘
#
# =============================================================================


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("向量化向量加法（优化版）—— API 详解版")
    print("=" * 60)

    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)

    C3 = torch.empty(N, device="cuda", dtype=torch.float32)
    C3_ = from_dlpack(C3, assumed_align=16)

    compiled_vec = cute.compile(vector_add_vectorized, A_, B_, C3_)
    compiled_vec(A_, B_, C3_)

    assert torch.allclose(C3, A + B, atol=1e-5, rtol=1e-5), "向量化版本：正确性验证失败！"
    print("✅ 向量化向量加法正确性验证通过！")

    time_vec = benchmark(compiled_vec, kernel_arguments=JitArguments(A_, B_, C3_))
    bw_vec = (3 * N * 4) / (time_vec * 1e3)
    print(f"⏱  向量化版 耗时: {time_vec:.2f} µs | 带宽: {bw_vec:.2f} GB/s")

    # ---- TMA 版本 ----
    print("\n" + "-" * 60)
    print("TMA 版向量加法 —— 体验 TMA 硬件搬运")
    print("-" * 60)

    C_tma = torch.empty(N, device="cuda", dtype=torch.float32)
    C_tma_ = from_dlpack(C_tma, assumed_align=16)

    compiled_tma = cute.compile(vector_add_tma, A_, B_, C_tma_)
    compiled_tma(A_, B_, C_tma_)

    assert torch.allclose(C_tma, A + B, atol=1e-5, rtol=1e-5), "TMA 版本：正确性验证失败！"
    print("✅ TMA 版向量加法正确性验证通过！")

    time_tma = benchmark(compiled_tma, kernel_arguments=JitArguments(A_, B_, C_tma_))
    bw_tma = (3 * N * 4) / (time_tma * 1e3)
    print(f"⏱  TMA 版   耗时: {time_tma:.2f} µs | 带宽: {bw_tma:.2f} GB/s")

    # ---- PyTorch 基准对比 ----
    print("\n" + "-" * 60)
    print("PyTorch 基准对比")
    print("-" * 60)

    C_pt = torch.empty_like(A)
    for _ in range(10):
        torch.add(A, B, out=C_pt)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    num_iter = 100
    for _ in range(num_iter):
        torch.add(A, B, out=C_pt)
    end.record()
    torch.cuda.synchronize()
    pt_time_us = start.elapsed_time(end) * 1000 / num_iter
    pt_bw = (3 * N * 4) / (pt_time_us * 1e3)
    print(f"⏱  PyTorch   耗时: {pt_time_us:.2f} µs | 带宽: {pt_bw:.2f} GB/s")

    # ---- 汇总对比 ----
    print("\n" + "=" * 60)
    print("性能汇总")
    print("=" * 60)
    print(f"  {'方法':<15} {'耗时(µs)':<12} {'带宽(GB/s)':<12} {'vs PyTorch':<12}")
    print(f"  {'-'*51}")
    print(f"  {'LSU 向量化':<15} {time_vec:<12.2f} {bw_vec:<12.2f} {pt_time_us/time_vec:<12.2f}x")
    print(f"  {'TMA 版':<15} {time_tma:<12.2f} {bw_tma:<12.2f} {pt_time_us/time_tma:<12.2f}x")
    print(f"  {'PyTorch':<15} {pt_time_us:<12.2f} {pt_bw:<12.2f} {'1.00x':<12}")
    print()
    print("💡 注意：TMA 版在 vector_add 场景下不会更快（多了 SMEM 中转）。")
    print("   TMA 的优势在 GEMM 等有数据复用的场景才能体现。")
    print("   这里只是为了体验 TMA 的写法。")

    print("\n🎉 教程 01 全部完成！")
