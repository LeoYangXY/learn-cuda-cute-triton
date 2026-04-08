"""
=============================================================================
教程 10: Tiled Copy 详解 —— 原生 CUDA vs CuTE 一一对照
=============================================================================

这篇教程只讲一件事：一个 Block 里有 N 个线程，要搬一块 (M, K) 的数据，怎么分工？

我们用一个最小的例子来讲：
  - 4 个线程
  - 4×4 的 float 数据（16 个元素）
  - 每个线程一次搬 2 个连续的 float（float2 = 64 bits）

=============================================================================
第一部分：原生 CUDA —— 你自己怎么写
=============================================================================

数据 tile (4行 × 4列):
  a  b  c  d
  e  f  g  h
  i  j  k  l
  m  n  o  p

你脑子里想 3 件事：

  ① 每次搬多少？
     每个线程一次搬 2 个连续 float（float2 = 64 bits）

  ② 线程怎么排？
     每行 4 个元素 ÷ 每线程 2 个 = 每行 2 个线程
     4 个线程 ÷ 每行 2 个 = 2 行
     → 排成 2行 × 2列

  ③ 搬几轮？
     一轮 = 4 线程 × 2 个值 = 8 个元素
     总共 16 个元素 → 需要 2 轮

把分工画出来：

  第 1 轮：
    T0:[a,b]  T1:[c,d]     ← 线程 0 和 1 负责第 0 行
    T2:[e,f]  T3:[g,h]     ← 线程 2 和 3 负责第 1 行

  第 2 轮：
    T0:[i,j]  T1:[k,l]     ← 线程 0 和 1 负责第 2 行
    T2:[m,n]  T3:[o,p]     ← 线程 2 和 3 负责第 3 行

原生 CUDA 代码（伪代码）：

  int tid = threadIdx.x;                      // 0, 1, 2, 3
  int threads_per_row = kTileN / 2;           // 4 / 2 = 2
  int i_row = tid / threads_per_row;          // T0→0, T1→0, T2→1, T3→1
  int i_col = tid % threads_per_row;          // T0→0, T1→1, T2→0, T3→1
  int elems_per_loop = 4 * 2;                 // 8 个元素/轮
  int n_loops = 16 / elems_per_loop;          // 2 轮

  for (int i = 0; i < n_loops; i++) {
      float2* src = (float2*)&global[i*8 + i_row*4 + i_col*2];
      float2* dst = (float2*)&shared[i*8 + i_row*4 + i_col*2];
      *dst = *src;
  }

=============================================================================
第二部分：CuTE —— 同样 3 件事，换成 3 个声明
=============================================================================

对应 ①：copy_atom —— "每次搬多少"
  原生 CUDA：(float2*)&src[...]  手动强转指针
  CuTE：    copy_atom(bits=64)   声明 64 bits

对应 ②：thr_layout + val_layout —— "线程怎么排"
  原生 CUDA：i_row = tid / 2, i_col = tid % 2, offset = i_row*4 + i_col*2
  CuTE：    thr_layout = (2, 2)   → 4 个线程排成 2行×2列
            val_layout = (1, 2)   → 每个线程搬 1行×2列 = 2 个连续值

  画出来：

    thr_layout = (2行, 2列)        val_layout = (1行, 2列)
    ┌─────┬─────┐                  每个线程负责的形状：
    │ T0  │ T1  │  ← 第 0 行       ┌───┬───┐
    ├─────┼─────┤                  │ v0│ v1│  ← 2 个连续值
    │ T2  │ T3  │  ← 第 1 行       └───┴───┘
    └─────┴─────┘

    合在一起，一轮覆盖 2行 × 4列 = 8 个元素：

      T0:[v0,v1]  T1:[v0,v1]
      T2:[v0,v1]  T3:[v0,v1]

    对应到数据（第 1 轮）：

      T0:[a,b]  T1:[c,d]
      T2:[e,f]  T3:[g,h]

对应 ③：CuTE 自动算"搬几轮"
  tile = 4×4 = 16 个元素
  一轮 = 2×4 = 8 个元素
  → 2 轮，CuTE 的 partition 自动算出来，cute.copy 自动循环

完整对照表：
  ┌──────────────┬──────────────────────┬───────────────────────────┐
  │   决策       │  原生 CUDA            │  CuTE                     │
  ├──────────────┼──────────────────────┼───────────────────────────┤
  │ ① 每次搬多少 │ (float2*)强制转换     │ copy_atom(bits=64)        │
  ├──────────────┼──────────────────────┼───────────────────────────┤
  │ ② 线程怎么排 │ i_row = tid / 2      │ thr_layout = (2, 2)       │
  │              │ i_col = tid % 2      │ val_layout = (1, 2)        │
  │              │ offset = i_row*4     │ get_slice(tid) 自动算       │
  │              │        + i_col*2     │                            │
  ├──────────────┼──────────────────────┼───────────────────────────┤
  │ ③ 搬几轮    │ n_loops = 16/8 = 2   │ 自动！                     │
  │              │ for (i=0; i<2; ...) │ cute.copy 内部循环          │
  │              │   src += 8           │ partition 自动分配          │
  ├──────────────┼──────────────────────┼───────────────────────────┤
  │ 地址计算     │ i_row*4 + i_col*2    │ 隐藏在 Layout 代数里       │
  │              │ + i_loop*8           │ 你只需要声明形状和步长       │
  └──────────────┴──────────────────────┴───────────────────────────┘

=============================================================================
第三部分：partition 做了什么
=============================================================================

partition = 从整个 tile 中，切出当前线程负责的那些元素。

原始 tile (4×4):                    partition 后，线程 0 拿到的：
┌───┬───┬───┬───┐
│ a │ b │ c │ d │  row 0            第 1 轮           第 2 轮
├───┼───┼───┼───┤                  ┌───┬───┐        ┌───┬───┐
│ e │ f │ g │ h │  row 1            │ a │ b │        │ i │ j │
├───┼───┼───┼───┤                  └───┴───┘        └───┴───┘
│ i │ j │ k │ l │  row 2
├───┼───┼───┼───┤                  shape = (2,  2,  1)
│ m │ n │ o │ p │  row 3                    ↑   ↑   ↑
└───┴───┴───┘───|                           │   │   └── 列方向不需要循环
                                            │   └── 行方向循环 2 轮
                                            └── 每轮搬 2 个值（atom 大小）

  shape 中第一个维度 = atom 内的值（一轮搬几个）
  后面的维度 = 需要循环几轮（行方向 × 列方向）

  "atom 之外，皆是循环"
  → atom 是一轮搬运的模板
  → tile 比 atom 大就多循环几轮

每个线程 partition 后拿到的数据：
  T0: 第 1 轮 [a, b]  第 2 轮 [i, j]
  T1: 第 1 轮 [c, d]  第 2 轮 [k, l]
  T2: 第 1 轮 [e, f]  第 2 轮 [m, n]
  T3: 第 1 轮 [g, h]  第 2 轮 [o, p]

=============================================================================
第四部分：partition_S 和 partition_D 为什么要分开
=============================================================================

简单场景（普通 load/store，比如本例的 g2s）：
  源端和目标端的线程-数据映射完全一样，S 和 D 没区别。

  T0 从 GMEM 读 [a,b] → 存到 SMEM 的 [a,b] 位置
  映射相同，partition_S 和 partition_D 切出来一样。

复杂场景（ldmatrix 指令，SMEM → 寄存器）：
  T0 从 SMEM 读了一整行 [a,b,c,d]，但硬件会把数据重新分发：
    a → T0 的寄存器
    b → T1 的寄存器
    c → T2 的寄存器
    d → T3 的寄存器

  源端（SMEM 视角）：T0 读了 [a,b,c,d]
  目标端（寄存器视角）：T0 只拿到了 [a]

  源和目标的映射不一样！
  所以 partition_S 切出的数据 ≠ partition_D 切出的数据。

  CuTE 的 Copy Atom 内部记录了源端和目标端各自的映射关系，
  partition_S/D 会分别用对应的映射来切分。

=============================================================================
第五部分：完整流程图
=============================================================================

步骤 1: make_tiled_copy — 定义"搬运计划"
  ┌─────────────────────────────────────────────────┐
  │  copy_atom:  每次 64 bits（2 个 float）          │
  │  thr_layout: 4 个线程排成 2×2                    │
  │  val_layout: 每个线程搬 1×2 个连续值             │
  │                                                  │
  │  → 内部自动算出：一轮覆盖 2行×4列 = 8 个元素      │
  └─────────────────────────────────────────────────┘
                        │
                        ▼
步骤 2: get_slice(tid) — 绑定到具体线程
  ┌─────────────────────────────────────────────────┐
  │  "我是线程 0，在 2×2 网格中位置 (0,0)"            │
  │  "我每轮负责第 0 行的前 2 个元素"                  │
  └─────────────────────────────────────────────────┘
                        │
                        ▼
步骤 3: partition_S / partition_D — 切出当前线程的数据
  ┌─────────────────────────────────────────────────┐
  │  tile 是 4×4 = 16 个元素                         │
  │  一轮覆盖 8 个 → 需要 2 轮                       │
  │                                                  │
  │  线程 0 拿到：                                    │
  │    第 1 轮: [a, b]  (row 0, col 0~1)             │
  │    第 2 轮: [i, j]  (row 2, col 0~1)             │
  │                                                  │
  │  shape = (2, 2, 1) = (atom值数, 行循环, 列循环)   │
  └─────────────────────────────────────────────────┘
                        │
                        ▼
步骤 4: cute.copy — 自动循环执行搬运
  ┌─────────────────────────────────────────────────┐
  │  第 1 轮：                                       │
  │    T0:[a,b]  T1:[c,d]                           │
  │    T2:[e,f]  T3:[g,h]                           │
  │                                                  │
  │  第 2 轮：                                       │
  │    T0:[i,j]  T1:[k,l]                           │
  │    T2:[m,n]  T3:[o,p]                           │
  │                                                  │
  │  → 和你手写 CUDA for 循环的结果完全一样！         │
  └─────────────────────────────────────────────────┘

=============================================================================
第六部分：实际代码 —— 2D Tile 的 g2s 拷贝
=============================================================================

下面我们写一个真实的 CuTeDSL 代码，完成 2D tile 的 GMEM → SMEM 拷贝，
然后再从 SMEM → 寄存器，最后寄存器 → GMEM 写回。

场景：
  - 矩阵 A 大小 (M, K)
  - 每个 Block 负责一个 (TILE_M, TILE_K) 的 tile
  - 128 个线程，每线程每次搬 4 个 float16（64 bits）
  - GMEM → SMEM → 寄存器 → GMEM（只做拷贝，不做计算，验证拷贝正确性）

对应到我们的 3 件事：
  ① 每次搬多少？ 64 bits = 4 个 float16
  ② 线程怎么排？ 128 个线程排成 (thr_M, thr_K) 的网格
  ③ 搬几轮？     CuTE 自动算

原生 CUDA 你得这么写：
  int threads_per_row = TILE_K / 4;          // 列方向几个线程
  int i_row = tid / threads_per_row;
  int i_col = tid % threads_per_row;
  int elems_per_loop = 128 * 4;              // 512 元素/轮
  int n_loops = (TILE_M * TILE_K) / elems_per_loop;

  for (int i = 0; i < n_loops; i++) {
      half4* src = (half4*)&gmem[...复杂地址计算...];
      half4* dst = (half4*)&smem[...复杂地址计算...];
      *dst = *src;
  }

CuTE 你只需要：
  copy_atom = make_copy_atom(CopyUniversalOp(), float16, bits=64)
  thr_layout = (thr_M, thr_K)   # 线程排布
  val_layout = (1, 4)            # 每线程 4 个连续值
  tiled_copy = make_tiled_copy(copy_atom, thr_layout, val_layout)
  thr_copy = tiled_copy.get_slice(tid)
  my_src = thr_copy.partition_S(gmem_tile)
  my_dst = thr_copy.partition_D(smem_tile)
  cute.copy(tiled_copy, my_src, my_dst)    # 完事！

=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

# =============================================================================
# 参数配置
# =============================================================================
M = 256           # 矩阵行数
K = 256           # 矩阵列数
TILE_M = 64       # 每个 Block 负责的行数
TILE_K = 64       # 每个 Block 负责的列数
NUM_THREADS = 128  # 每个 Block 的线程数
VEC = 4           # 每个线程每次搬 4 个 float16（= 64 bits）

# 线程排布计算（对应原生 CUDA 里你手动算的那些东西）
# 每行 TILE_K 个元素，每线程搬 VEC 个 → 每行需要 TILE_K/VEC 个线程
THR_K = TILE_K // VEC   # = 64 / 4 = 16 个线程负责列方向
THR_M = NUM_THREADS // THR_K  # = 128 / 16 = 8 个线程负责行方向
# 128 个线程排成 8行 × 16列

# 一轮覆盖 = 8行 × 64列 = 512 个元素
# 总共 64×64 = 4096 元素 → 4096/512 = 8 轮
LOOPS = (TILE_M * TILE_K) // (NUM_THREADS * VEC)

SMEM_PAD = 0  # 不加 padding，保持 GMEM 和 SMEM 列数一致
              # 实际 GEMM 中会加 padding 消除 bank conflict（见教程 05）

"""
=============================================================================
对应到我们的 4 线程例子：

  4 线程例子：             实际代码：
  4 个线程                 128 个线程
  4×4 tile                 64×64 tile
  每线程搬 2 个 float      每线程搬 4 个 float16
  排成 2×2                 排成 8×16
  2 轮搬完                 8 轮搬完

  核心思路完全一样！只是数字变大了。

  线程排布图（实际代码）：

  TILE_K = 64 列
  ├──────────────────────────────────────────────────────────────────┤

  T0:[v0~v3]  T1:[v0~v3]  T2:[v0~v3]  ...  T15:[v0~v3]   ← row 0     ┐
  T16:[v0~v3] T17:[v0~v3] T18:[v0~v3] ...  T31:[v0~v3]   ← row 1     │
  T32:[v0~v3] T33:[v0~v3] T34:[v0~v3] ...  T47:[v0~v3]   ← row 2     │ 一轮
  T48:[v0~v3] T49:[v0~v3] T50:[v0~v3] ...  T63:[v0~v3]   ← row 3     │ 覆盖
  T64:[v0~v3] T65:[v0~v3] T66:[v0~v3] ...  T79:[v0~v3]   ← row 4     │ 8 行
  T80:[v0~v3] T81:[v0~v3] T82:[v0~v3] ...  T95:[v0~v3]   ← row 5     │
  T96:[v0~v3] T97:[v0~v3] T98:[v0~v3] ...  T111:[v0~v3]  ← row 6     │
  T112:[v0~v3]T113:[v0~v3]T114:[v0~v3]...  T127:[v0~v3]  ← row 7     ┘

  一轮覆盖 8 行 × 64 列 = 512 个元素
  总共 64 行 → 需要 64/8 = 8 轮

  第 1 轮：row 0~7
  第 2 轮：row 8~15
  第 3 轮：row 16~23
  ...
  第 8 轮：row 56~63
=============================================================================
"""


# =============================================================================
# Kernel：2D Tile 拷贝（GMEM → SMEM → 寄存器 → GMEM）
# =============================================================================
#
# 数据流：
#   GMEM tile --[g2s tiled_copy]--> SMEM --[s2r tiled_copy]--> 寄存器 --[r2g]--> GMEM
#
# 这就是 GEMM kernel 里每个 k-loop 迭代做的事情，
# 只是这里我们不做 MMA 计算，直接把数据写回去验证正确性。

@cute.kernel
def tiled_copy_kernel(
    gA: cute.Tensor,                # GMEM 输入，shape = (M, K)
    gC: cute.Tensor,                # GMEM 输出，shape = (M, K)
    smem_layout: cute.Layout,       # SMEM 布局，shape = (TILE_M, TILE_K + PAD)
    thr_layout: cute.Layout,        # 线程排布 2D，shape = (THR_M, THR_K) = (8, 16)
    val_layout: cute.Layout,        # 值排布 2D，shape = (1, VEC) = (1, 4)
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # ===================== 第 1 步：CTA 级分块 =====================
    # 从大矩阵 (M, K) 中切出当前 Block 负责的 (TILE_M, TILE_K) 子块
    #
    # 原生 CUDA 你得手算指针：
    #   half* tile_ptr = A + bidx * TILE_M * K + bidy * TILE_K;
    #   // 注意：tile_ptr 每行的 stride 是 K（不是 TILE_K），你得自己记住
    #
    # CuTE 自动算好偏移，stride 也自动保留正确的值
    blkA = cute.local_tile(gA, (TILE_M, TILE_K), (bidx, bidy))  # shape: (64, 64)
    blkC = cute.local_tile(gC, (TILE_M, TILE_K), (bidx, bidy))  # shape: (64, 64)

    # ===================== 第 2 步：分配 SMEM =====================
    # 原生 CUDA 等价：__shared__ half sA[TILE_M][TILE_K];
    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(cutlass.Float16, smem_layout)  # shape: (64, 64)

    # ===================== 第 3 步：make_copy_atom =====================
    # ① 告诉 CuTE "每次搬多少"
    #
    # 原生 CUDA 等价：*(uint2*)&dst = *(uint2*)&src;  // 64-bit load/store
    # 这里声明每条指令搬 64 bits = 4 个 float16
    # 这一步只是"选工具"，还没搬任何数据
    copy_atom_g2s = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),               # 用普通 load/store 指令，走LSU这个硬件资源
        gA.element_type,                             # 数据类型 = Float16
        num_bits_per_copy=gA.element_type.width * VEC,  # 16 × 4 = 64 bits
    )

    # ===================== 第 4 步：make_tiled_copy_tv =====================
    # 把 ① copy_atom + ② 线程排布 + ③ 值排布 组合成完整的"搬运计划"
    #
    # thr_layout = (8, 16) stride=(16, 1)：128 个线程排成 8行×16列
    # val_layout = (1, 4) stride=(0, 1)：每个线程搬 1行×4列 = 4 个连续值
    #
    # 合在一起，一轮覆盖：8行 × (16线程×4值) = 8行 × 64列 = 512 个元素
    # tile 是 64×64 = 4096 元素 → 需要 4096/512 = 8 轮
    #
    # 原生 CUDA 等价：这些计算你得自己手动完成：
    #   int threads_per_row = TILE_K / 4;    // = 16
    #   int rows_per_round = 128 / 16;       // = 8
    #   int n_rounds = TILE_M / 8;           // = 8
    #
    # 这一步只是"制定计划"，还没搬数据
    tiled_copy_g2s = cute.make_tiled_copy_tv(
        copy_atom_g2s,     # ① 每次搬 64 bits
        thr_layout,        # ② 线程排成 (8, 16)
        val_layout,        # ③ 每线程搬 (1, 4) 个值
    )

    # ===================== 第 5 步：get_slice =====================
    # 把"搬运计划"绑定到当前线程
    #
    # thr_g2s 不是 tensor！它是一个"绑定了线程 ID 的切分器"。
    # 它本身不持有任何数据，只记住了"我是线程 X"这个信息。
    # 后面你拿它调 partition_S/D，它才根据线程 X 的身份去切数据。
    #
    # 内部做了什么？查 thr_layout 的映射表：
    #   thr_layout = (8, 16), stride=(16, 1)
    #
    #   tid=0  → (row=0, col=0)    即 0/16=0,  0%16=0
    #   tid=1  → (row=0, col=1)    即 1/16=0,  1%16=1
    #   tid=5  → (row=0, col=5)    即 5/16=0,  5%16=5
    #   tid=16 → (row=1, col=0)    即 16/16=1, 16%16=0
    #   tid=127→ (row=7, col=15)   即 127/16=7,127%16=15
    #
    # get_slice(5) 就是查这张表："线程 5 在网格中是 (row=0, col=5)"
    # 然后把这个位置信息存起来，给后面的 partition 用。注意这个查表的理解是关键
    #
    # 后面 partition_S(blkA) 用这个位置算出线程 5 每轮搬哪些数据：
    #   线程 5 在 (row=0, col=5)，每线程搬 4 个连续值
    #   第 1 轮：row=0,  col=5*4=20 → blkA[0,  20:24]
    #   第 2 轮：row=8,  col=20     → blkA[8,  20:24]
    #   ...
    #   第 8 轮：row=56, col=20     → blkA[56, 20:24]
    #
    # 原生 CUDA 等价：
    #   int i_row = tid / 16;   // 我在第几行
    #   int i_col = tid % 16;   // 我在第几列
    #   // get_slice 就是帮你做了这个除法和取模，存起来给 partition 用
    thr_g2s = tiled_copy_g2s.get_slice(tidx)

    # ===================== 第 6 步：partition_S / partition_D =====================
    # 从源和目标 tensor 中切出当前线程负责的数据
    #
    # partition_S(blkA)：从 GMEM 的 blkA 中切出 "我要读的元素"
    # partition_D(sA)：  从 SMEM 的 sA 中切出 "我要写的位置"
    #
    # 对于线程 0（i_row=0, i_col=0）：
    #   partition 返回 8 轮 × 每轮 4 个值 的视图
    #   第 1 轮：row 0, col 0~3
    #   第 2 轮：row 8, col 0~3
    #   ...
    #   第 8 轮：row 56, col 0~3
    #
    # 原生 CUDA 等价：你在 for 循环里手算每轮的地址偏移
    #   for (int round = 0; round < 8; round++) {
    #       int row = round * 8 + i_row;
    #       int col = i_col * 4;
    #       // src = &gmem[row * K + col];
    #       // dst = &smem[row * (TILE_K + PAD) + col];
    #   }
    tAgA = thr_g2s.partition_S(blkA)   # 从 GMEM 读：当前线程的所有轮次的源数据
    tAsA = thr_g2s.partition_D(sA)     # 往 SMEM 写：当前线程的所有轮次的目标位置

    # partition 后的 shape（实际运行打印出来的）：
    #
    #   blkA shape:  (64, 64)           ← 整个 tile
    #   tAgA shape:  ((4, 1), 8, 1)     ← partition 后，线程 0 拿到的
    #   tAsA shape:  ((4, 1), 8, 1)     ← 同上（简单场景 S 和 D 一样）
    #
    # 怎么读这个 shape？
    #
    #   ((4, 1),  8,  1)
    #     ↑       ↑   ↑
    #     │       │   └── 列方向循环 1 次（不用循环，一轮就覆盖了 64 列）
    #     │       └── 行方向循环 8 次（8 轮才能覆盖 64 行）
    #     └── atom 内：这个线程一轮搬 4×1 = 4 个值
    #
    # 怎么算出来的？
    #   tile = (64, 64)
    #   一轮 atom 覆盖 = thr_layout × val_layout = (8,16) × (1,4) = (8, 64)
    #   行方向：64 行 ÷ 8 行/轮 = 8 轮
    #   列方向：64 列 ÷ 64 列/轮 = 1 轮
    #   每个线程每轮搬 4 个值
    #   → shape = ((4, 1),  8,  1)
    #
    # 这就是 "atom 之外，皆是循环"：
    #   shape 第一个维度 = atom 内的值（一轮搬几个）
    #   后面的维度 = 循环次数（行方向 × 列方向）
    #   cute.copy 看到 (4, 8, 1)，就知道：每轮搬 4 个值，循环 8×1 = 8 轮

    # ===================== 第 7 步：cute.copy —— 执行！ =====================
    # 到这里才真正搬数据！
    # cute.copy 读取 tAgA/tAsA 的 shape，发现后面是 (8, 1)
    # → 自动生成 8 轮循环，每轮每线程用一条 64-bit load/store 搬 4 个 fp16
    #
    # 原生 CUDA 等价：
    #   for (int round = 0; round < 8; round++) {
    #       *(uint2*)&smem[dst_offset] = *(uint2*)&gmem[src_offset];
    #   }
    #
    # 上面 6 步都是准备工作，这一行才是真正干活
    cute.copy(tiled_copy_g2s, tAgA, tAsA)

    cute.arch.sync_threads()  # 等所有线程写完 SMEM

    # ===================== 第 8 步：SMEM → 寄存器 → GMEM（验证正确性） =====================
    # 用同样的套路再建一个 tiled_copy，从 SMEM 读到寄存器，再从寄存器写到 GMEM
    # 如果最后 C == A，说明 g2s 拷贝是正确的
    #
    # 注意：这里的 4 步（make_copy_atom → make_tiled_copy → get_slice → partition → copy）
    # 和上面 g2s 完全一样的套路，只是源和目标换了

    # ① 每次搬多少：和 g2s 一样，64 bits
    copy_atom_s2r = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        sA.element_type,
        num_bits_per_copy=sA.element_type.width * VEC,
    )
    # ② 制定计划：和 g2s 用相同的线程排布
    tiled_copy_s2r = cute.make_tiled_copy_tv(copy_atom_s2r, thr_layout, val_layout)
    # ③ 绑定线程
    thr_s2r = tiled_copy_s2r.get_slice(tidx)
    # ④ 切分数据
    tSsA = thr_s2r.partition_S(sA)     # 从 SMEM 读
    tSgC = thr_s2r.partition_D(blkC)   # 往 GMEM 写
    # ⑤ 在寄存器中开空间
    rA = cute.make_fragment_like(tSsA)
    # ⑥ 执行：SMEM → 寄存器 → GMEM
    cute.copy(tiled_copy_s2r, tSsA, rA)    # SMEM → 寄存器（LDS 指令）
    cute.copy(tiled_copy_s2r, rA, tSgC)    # 寄存器 → GMEM（STG 指令）


# =============================================================================
# Host 函数：配置参数并启动 kernel
# =============================================================================
@cute.jit
def tiled_copy_host(mA: cute.Tensor, mC: cute.Tensor):

    # SMEM 布局：(TILE_M, TILE_K + PAD) = (64, 64)，行优先
    # 原生 CUDA 等价：__shared__ half sA[64][64];
    # （没加 padding，实际 GEMM 中会加 padding 消除 bank conflict）
    smem_layout = cute.make_ordered_layout(
        (TILE_M, TILE_K + SMEM_PAD),   # (64, 72)
        order=(1, 0)                    # 行优先：列方向连续
    )

    # 线程排布（2D）—— 对应"② 线程怎么排"
    #
    # 128 个线程排成 THR_M × THR_K = 8行 × 16列
    # stride = (THR_K, 1) = (16, 1) → 行优先
    #   T0 在 (0,0)，T1 在 (0,1)，...，T15 在 (0,15)
    #   T16 在 (1,0)，T17 在 (1,1)，...
    #
    # 原生 CUDA 等价：
    #   int i_col = tid % 16;
    #   int i_row = tid / 16;
    thr_layout = cute.make_layout(
        (THR_M, THR_K),         # (8, 16)
        stride=(THR_K, 1)       # (16, 1)
    )

    # 值排布（2D）—— 每个线程搬 1行×4列 = 4 个连续值
    val_layout = cute.make_layout(
        (1, VEC),               # (1, 4)
        stride=(0, 1)           # 列方向连续
    )

    grid_m = M // TILE_M   # = 256/64 = 4 个 Block（行方向）
    grid_k = K // TILE_K   # = 256/64 = 4 个 Block（列方向）

    tiled_copy_kernel(mA, mC, smem_layout, thr_layout, val_layout).launch(
        grid=(grid_m, grid_k, 1),
        block=(NUM_THREADS, 1, 1),
    )


"""
=============================================================================
第七部分：为什么 CuTE 的方式更好
=============================================================================

对于这个例子，原生 CUDA 也不难写。但想想 GEMM 里你要面对的：

1. Swizzle（消除 SMEM bank conflict）：
   原生 CUDA：smem 地址要异或一个 pattern，每次访问都得手算
   CuTE：换一个 smem_layout（带 swizzle 的），其他代码一行不改

2. ldmatrix（高效 s2r 拷贝）：
   原生 CUDA：ptx 内联汇编，地址计算更复杂（每个线程读一行分给不同线程）
   CuTE：换一个 copy_atom（LdMatrix），其他代码一行不改

3. 跨架构兼容：
   原生 CUDA：Ampere 用 cp.async，Hopper 用 TMA，每个都得重写
   CuTE：换一个 copy_atom，其他代码一行不改

看出规律了吗？不管底层怎么变，你只需要换 copy_atom。
thr_layout、val_layout、partition、cute.copy 这些上层逻辑都不用动。

这就是 CuTE 的核心价值：
  把"地址计算"和"搬运逻辑"解耦。
  你声明"搬多少、怎么排"，CuTE 负责算地址和循环。
=============================================================================
"""


# =============================================================================
# 主函数：运行 2D tiled copy 并验证正确性
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("教程 10: Tiled Copy 详解 —— 2D Tile GMEM→SMEM→寄存器→GMEM")
    print("=" * 60)
    print()
    print(f"矩阵大小: ({M}, {K})")
    print(f"Tile 大小: ({TILE_M}, {TILE_K})")
    print(f"线程数: {NUM_THREADS}")
    print(f"每线程搬: {VEC} 个 float16 = {VEC * 16} bits")
    print(f"线程排布: ({THR_M}, {THR_K}) = {THR_M}行 × {THR_K}列")
    print(f"一轮覆盖: {THR_M}行 × {TILE_K}列 = {THR_M * TILE_K} 个元素")
    print(f"需要轮数: {LOOPS}")
    print()

    # 创建测试数据
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    C = torch.zeros(M, K, device="cuda", dtype=torch.float16)

    A_ = from_dlpack(A, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    # 编译 + 运行
    compiled = cute.compile(tiled_copy_host, A_, C_)
    compiled(A_, C_)

    # 验证：C 应该和 A 完全相同
    if torch.allclose(C, A, atol=1e-3):
        print("✅ Tiled Copy 正确性验证通过！")
        print("   GMEM → SMEM → 寄存器 → GMEM，数据完全一致。")
    else:
        diff = (C - A).abs().max().item()
        print(f"❌ 验证失败！最大误差: {diff}")

    # 性能测试
    time_us = benchmark(compiled, kernel_arguments=JitArguments(A_, C_))
    bytes_moved = 2 * M * K * 2  # 读 A + 写 C，每个 float16 = 2 字节
    bw = bytes_moved / (time_us * 1e3)  # GB/s
    print(f"⏱  耗时: {time_us:.2f} µs | 带宽: {bw:.2f} GB/s")

    print()
    print("=" * 60)
    print("回顾：原生 CUDA vs CuTE，你要决定的 3 件事")
    print("=" * 60)
    print()
    print("  ① 每次搬多少？")
    print(f"     原生 CUDA: (half4*)&src[...]")
    print(f"     CuTE:      copy_atom(bits={VEC * 16})")
    print()
    print("  ② 线程怎么排？")
    print(f"     原生 CUDA: i_row = tid / {THR_K}, i_col = tid % {THR_K}")
    print(f"     CuTE:      thr_layout = ({THR_M}, {THR_K})")
    print(f"                val_layout = (1, {VEC})")
    print()
    print("  ③ 搬几轮？")
    print(f"     原生 CUDA: for (i=0; i<{LOOPS}; i++) {{ ... }}")
    print(f"     CuTE:      cute.copy(tiled_copy, src, dst)  // 自动！")
    print()
    print("🎉 教程 10 完成！")
