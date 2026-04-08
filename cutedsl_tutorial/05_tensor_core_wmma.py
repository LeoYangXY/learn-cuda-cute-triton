"""
=============================================================================
教程 05: Tensor Core（WMMA）与向量化拷贝
=============================================================================

Tensor Core 是 NVIDIA GPU 的矩阵乘法专用硬件单元。
从 Volta (SM70) 开始引入，每代都在增强。

架构演进：
  SM70 (Volta/Turing)  : WMMA — Warp-level Matrix Multiply Accumulate
  SM80 (Ampere)        : WMMA 增强 + TF32/BF16 支持
  SM90 (Hopper)        : WGMMA — Warp Group MMA（4个warp协作）+ TMA
  SM100 (Blackwell)    : UMMA/tcgen05 — 统一 MMA + TMEM + 2CTA 协作

=============================================================================
原生 CUDA 怎么用 Tensor Core
=============================================================================

在原生 CUDA 中，你要用 Tensor Core 做 GEMM，大致流程是：

  1. 手动写 PTX 内联汇编调用 mma.sync 指令
     asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ..."
                  : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)   // D 的 4 个 float
                  :  "r"(a0), "r"(a1), "r"(a2), "r"(a3),      // A 的 4 个寄存器
                     "r"(b0), "r"(b1),                          // B 的 2 个寄存器
                     "f"(c0), "f"(c1), "f"(c2), "f"(c3));      // C 的 4 个 float

  2. 这条指令的含义：
     - 一个 Warp（32 线程）协作完成 D = A × B + C
     - A 是 16×16 的 fp16 矩阵（行优先）
     - B 是 16×8 的 fp16 矩阵（列优先）
     - C/D 是 16×8 的 fp32 矩阵
     - 每个线程手里拿着 A 的 8 个元素、B 的 4 个元素、C/D 的 4 个元素
     - 哪个线程拿哪些元素 → 硬件规定死的，你得查手册

  3. 要做大矩阵（比如 4096×4096），你还得：
     - 手动分 tile：每个 Block 负责 C 的一个 (128, 128) 子块
     - 手动 K 循环：沿 K 方向每次取 (128, 64) 的 A 和 (128, 64) 的 B
     - 手动搬数据：GMEM → SMEM → 寄存器，每一步都手算地址
     - 手动多 Warp 协作：4 个 Warp 分别负责不同的子块
     - 手动拼结果：MMA 指令只算 16×8，要覆盖 128×128 得循环很多次

  4. 总之，原生 CUDA 你得自己管：
     a. 选指令（mma.sync.aligned.m16n8k16...）
     b. 查手册知道每个线程拿哪些数据
     c. 手动把大矩阵切成 tile
     d. 手动安排多个 Warp 分工
     e. 手动循环调用 MMA 指令覆盖整个 tile
     f. 手动搬数据（GMEM→SMEM 用 cp.async / LDG.128，SMEM→寄存器 用 ldmatrix）

=============================================================================
CuTE 怎么做 —— 和原生 CUDA 一一对应
=============================================================================

  原生 CUDA 你手动做的事        CuTE 对应的 API
  ─────────────────────────    ─────────────────────────────────
  a. 选 MMA 指令                MmaF16BF16Op(shape_mnk=(16,8,16))
     mma.sync.m16n8k16...      → 封装了 PTX 指令，你不用写汇编

  b. 查手册知道线程-数据映射     MMA_Traits（内部自动处理）
     "线程 5 拿 A 的第几个元素"  → 你完全不用关心，CuTE 帮你查好了

  c. 切 tile                   local_tile(mA, cta_tiler, (bidx, bidy, None))
     A_tile = A + bidx*128*K   → 自动算指针偏移和 stride

  d. 安排多 Warp 分工            make_tiled_mma(op, atom_layout_mnk=(2,2,1))
     "4 个 Warp 排成 2×2"      → atom_layout 决定 Warp 怎么排

  e. 循环调用 MMA 覆盖 tile     cute.gemm(tiled_mma, D, A, B, C)
     for (...) mma.sync(...)   → 内部自动循环所有子块

  f. 搬数据                     cute.copy(tiled_copy, src, dst)
     cp.async / ldmatrix        → copy_atom 决定用什么指令

  整体流程：
  ┌────────────────────────────────────────────────────────────┐
  │  原生 CUDA                     CuTE                        │
  │                                                            │
  │  选 PTX 指令           →       MmaF16BF16Op                │
  │  查线程-数据映射表      →       MMA_Traits（内部）          │
  │  ↓ 组合成                       ↓                          │
  │  "一个 Warp 做一次     →       MMA_Atom                    │
  │   16×8×16 的乘法"              （最小矩阵乘法单元）        │
  │  ↓ 多个 Warp 拼起来             ↓                          │
  │  手动安排 4 个 Warp    →       TiledMMA                    │
  │  分别负责不同子块               make_tiled_mma(op,          │
  │                                  atom_layout=(2,2,1))      │
  │  ↓ 分给每个线程                  ↓                          │
  │  tid 算自己负责哪些数据 →       ThrMMA                      │
  │                                 tiled_mma.get_slice(tid)   │
  │  ↓ 切分 A/B/C                   ↓                          │
  │  手动算 A/B/C 的偏移    →      partition_A/B/C              │
  │                                 thr_mma.partition_A(sA)    │
  │  ↓ 执行                         ↓                          │
  │  for (...) {                    cute.gemm(tiled_mma,       │
  │    asm("mma.sync...");            tCrC, tCrA, tCrB, tCrC)  │
  │  }                              → 一行搞定                  │
  └────────────────────────────────────────────────────────────┘

本教程在 SM80+ 上使用 WMMA Tensor Core。
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from typing import Tuple
import torch


class GemmWMMA:
    """
    使用 WMMA Tensor Core 的 GEMM 实现。

    数据流：
      GMEM → (向量化拷贝) → SMEM → (LdMatrix) → Register → (WMMA) → Register → GMEM

    关键组件：
      1. TiledCopy (GMEM→SMEM): 向量化全局内存到共享内存的拷贝
      2. TiledCopy (SMEM→RMEM): LdMatrix 指令，高效加载到寄存器
      3. TiledMma: WMMA 矩阵乘加操作
    """

    def __init__(self, cta_tiler=(128, 128, 64)):
        # cta_tiler = (bM, bN, bK) = 每个 Block 负责的 C 子块大小 + K 方向分块大小
        #
        # 原生 CUDA 等价：
        #   #define BM 128   // 每个 Block 算 C 的 128 行
        #   #define BN 128   // 每个 Block 算 C 的 128 列
        #   #define BK 64    // 每次 K 循环取 64 列的 A 和 64 行的 B
        self._bM, self._bN, self._bK = cta_tiler
        self._cta_tiler = cta_tiler

        # mma_inst_shape = (16, 8, 16)
        # 这是 Ampere (SM80) 的 Tensor Core 硬件提供的最小矩阵乘法：
        #   A (16×16, fp16) × B (16×8, fp16) + C (16×8, fp32) = D (16×8, fp32)
        # 一个 Warp（32 线程）执行一条 mma.sync.aligned.m16n8k16 指令完成。
        #
        # 原生 CUDA 等价：
        #   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...");
        #   这条指令由 32 个线程协作执行，每个线程各持有 A/B/C/D 的一部分寄存器。
        self.mma_inst_shape = (16, 8, 16)

        # atom_layout_mnk = (2, 2, 1)
        # 意思：在 M 方向放 2 个 Warp，N 方向放 2 个 Warp，K 方向 1 个
        # → 总共 2×2×1 = 4 个 Warp 并行算 4 个小矩阵乘法
        #
        # 画出来：
        #   一个 Atom（1 个 Warp）算 16×8 的 C 子块
        #   4 个 Warp 排成 2×2，覆盖更大的 C 子块：
        #
        #             N 方向
        #         ┌────────┬────────┐
        #         │ Warp0  │ Warp1  │  ← M 方向第 0 行
        #  M 方向 │ 16×8   │ 16×8   │
        #         ├────────┼────────┤
        #         │ Warp2  │ Warp3  │  ← M 方向第 1 行
        #         │ 16×8   │ 16×8   │
        #         └────────┴────────┘
        #
        #   4 个 Warp 一起覆盖 32×16 的 C 子块
        #
        # 原生 CUDA 等价：
        #   int warp_id = threadIdx.x / 32;
        #   int warp_m = warp_id / 2;   // 0 或 1
        #   int warp_n = warp_id % 2;   // 0 或 1
        #   // Warp (warp_m, warp_n) 负责 C 的 [warp_m*16:(warp_m+1)*16, warp_n*8:(warp_n+1)*8]
        self.atom_layout_mnk = (2, 2, 1)

        # 总线程数 = 32 × 2 × 2 = 128（4 个 Warp）
        self._num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self._smem_padding = 8   # SMEM 每行多加 8 个元素避免 bank conflict
        self._num_vectorized = 4  # GMEM→SMEM 拷贝时每线程一次搬 4 个 fp16 = 64 bits

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        # ==================================================================
        # 1. 创建 MMA 操作 —— 选择 Tensor Core 指令
        # ==================================================================
        # 原生 CUDA 等价：你选择用哪条 PTX 指令
        #   asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ...");
        #
        # CuTE 里你不用写汇编，只需要声明：
        #   - 输入类型：Float16
        #   - 累加类型：Float32
        #   - 指令形状：(16, 8, 16) → M=16, N=8, K=16
        #
        # 这一步只是"选工具"，还没有做任何计算。
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16,      # A 和 B 的数据类型
            acc_dtype=cutlass.Float32,      # C/D 的累加类型
            shape_mnk=self.mma_inst_shape   # (16, 8, 16)
        )

        # ==================================================================
        # permutation_mnk：4 个 Warp 合在一起能覆盖多大的 C 子块
        # ==================================================================
        # 一个 Atom（1 Warp）算 C 的 16×8
        # atom_layout = (2, 2, 1) → 2×2 = 4 个 Warp
        #
        # 4 个 Warp 覆盖的 C 子块大小：
        #   M 方向：2 × 16 = 32
        #   N 方向：2 × 8 × 2 = 32  （×2 是因为 Ampere 的 permutation 会让每个
        #                              Warp 在 N 方向算两个 8 列的子块）
        #   K 方向：1 × 16 = 16
        #
        # 所以 4 个 Warp 一起覆盖 C 的 32×32 子块，每次消耗 K=16 列
        #
        # 要覆盖整个 CTA tile 的 C (128×128)：
        #   M 方向需要 128/32 = 4 轮
        #   N 方向需要 128/32 = 4 轮
        #   → cute.gemm 内部自动循环 4×4 = 16 次 MMA 指令组
        #
        # 原生 CUDA 等价：
        #   for (int mma_m = 0; mma_m < 4; mma_m++)
        #     for (int mma_n = 0; mma_n < 4; mma_n++)
        #       asm("mma.sync...");   // 每组 4 个 Warp 各自执行
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],       # 2×16 = 32
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,   # 2×8×2 = 32
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],       # 1×16 = 16
        )

        # ==================================================================
        # make_tiled_mma：把 MMA 指令 + Warp 排布组合成完整的计算计划
        # ==================================================================
        # 这一步做的事情：
        #   1. 封装 PTX 指令 → MMA_Atom（最小矩阵乘法单元）
        #   2. 按 atom_layout=(2,2,1) 安排 4 个 Warp 并行
        #   3. 按 permutation 确定一次能覆盖多大的 C 子块
        #   4. 记录每个线程负责 A/B/C 的哪些元素（硬件定死的映射）
        #
        # 就像 make_tiled_copy 制定"搬运计划"一样，
        # make_tiled_mma 制定"计算计划"，但还没有真正计算。
        #
        # 原生 CUDA 等价：你得自己在脑子里（或纸上）规划好上面这些东西，
        #   然后手动写代码去实现。CuTE 帮你把规划过程自动化了。
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,                    # 选哪条 MMA 指令
            atom_layout_mnk=self.atom_layout_mnk, # 4 个 Warp 怎么排：(2,2,1)
            permutation_mnk=permutation_mnk       # 覆盖多大的 C 子块
        )

        # ==================================================================
        # 2. SMEM Layout —— 共享内存的数据排布
        # ==================================================================
        # A 在 SMEM 中的大小：(bM, bK) = (128, 64)
        # B 在 SMEM 中的大小：(bN, bK) = (128, 64)
        #
        # stride = (bK + padding, 1)：行优先，每行末尾加 padding 个元素
        #
        # 为什么加 padding？
        #   SMEM 有 32 个 bank，每个 bank 4 字节宽。
        #   如果 bK=64，每行 64 个 fp16 = 128 字节 = 恰好 32 个 bank
        #   → 同一列的元素落在同一个 bank → 列方向访问会 bank conflict
        #   加 8 个 fp16 = 16 字节，错开 4 个 bank，打破对齐 → 消除 conflict
        #
        # 原生 CUDA 等价：
        #   __shared__ half sA[128][64 + 8];   // 每行多 8 个元素
        #   __shared__ half sB[128][64 + 8];
        padding = self._smem_padding
        sA_layout = cute.make_layout((self._bM, self._bK), stride=(self._bK + padding, 1))
        sB_layout = cute.make_layout((self._bN, self._bK), stride=(self._bK + padding, 1))

        # ==================================================================
        # 3. GMEM → SMEM 的向量化拷贝配置
        # ==================================================================
        # 这部分和教程 10 讲的 tiled copy 完全一样的思路：
        #   ① 每次搬多少？ 4 个 fp16 = 64 bits
        #   ② 线程怎么排？ 128 个线程排成 (?, ?) 的网格
        #   ③ 搬几轮？     自动
        num_vec = self._num_vectorized

        # ① copy_atom：每线程每次搬 64 bits = 4 个 fp16
        # 原生 CUDA 等价：*(uint2*)&smem[...] = *(uint2*)&gmem[...];  // 64-bit load
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vec  # 16 × 4 = 64 bits
        )

        # ② 线程排布
        # bK=64, num_vec=4 → 每行需要 64/4 = 16 个线程
        # 128 个线程 → 128/16 = 8 行
        # → tA = (8, 16) stride=(16, 1)：8 行 16 列，行优先
        #
        # 画出来（以 A 的 SMEM 为例，128 行 × 64 列）：
        #
        #   bK = 64 列
        #   ├───────────────────────────────────────────────────────────────┤
        #   T0:[4个] T1:[4个] T2:[4个] ... T15:[4个]   ← row 0    ┐
        #   T16:[4个]T17:[4个]T18:[4个]... T31:[4个]   ← row 1    │ 一轮
        #   ...                                                     │ 覆盖
        #   T112:[4个] ...             ... T127:[4个]   ← row 7    ┘ 8 行
        #
        #   一轮覆盖 8 行 × 64 列 = 512 个 fp16
        #   总共 128 行 → 128/8 = 16 轮
        #
        # 原生 CUDA 等价：
        #   int col_tid = tid % 16;        // 列方向的线程 id
        #   int row_tid = tid / 16;        // 行方向的线程 id
        #   for (int round = 0; round < 16; round++) {
        #       int row = round * 8 + row_tid;
        #       int col = col_tid * 4;
        #       *(uint2*)&sA[row][col] = *(uint2*)&gA[row][col];
        #   }
        major_mode_size = self._bK // num_vec   # 64 / 4 = 16
        tA = cute.make_layout(
            shape=(self._num_threads // major_mode_size, major_mode_size),  # (8, 16)
            stride=(major_mode_size, 1)   # (16, 1) → 行优先
        )
        # vA：每个线程搬 1×4 个连续值
        vA = cute.make_layout(shape=(1, num_vec), stride=(0, 1))   # (1, 4)

        # 组装成 tiled_copy（搬运计划）
        # make_tiled_copy_tv：和 make_tiled_copy 功能一样
        #   tv 后缀表示你直接传入 Thread-Value 的 layout
        tiled_copy_A = cute.make_tiled_copy_tv(atom_copy, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_copy, tA, vA)

        # ====== 4. 启动 Kernel ======
        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B,
            tiled_mma, permutation_mnk
        ).launch(grid=grid_dim, block=(self._num_threads, 1, 1))

    @cute.kernel
    def kernel(
        self,
        mA, mB, mC,
        sA_layout, sB_layout,
        tiled_copy_A, tiled_copy_B,
        tiled_mma, permutation_mnk,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        # ==================================================================
        # SMEM 分配
        # ==================================================================
        # 原生 CUDA 等价：
        #   __shared__ half sA[128][64+8];
        #   __shared__ half sB[128][64+8];
        allocator = cutlass.utils.SmemAllocator()
        sA = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
        sB = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)

        # ==================================================================
        # CTA 级分块 —— 从大矩阵切出当前 Block 的 tile
        # ==================================================================
        # GEMM: C(M,N) = A(M,K) × B(N,K)^T （注意 B 是 (N,K) 即行存储的转置形式）
        #
        # cta_tiler = (128, 128, 64) 分别对应 M、N、K 方向
        #
        # 对于 A(M, K)：
        #   proj=(1, None, 1) → 取 M 和 K 维度，跳过 N
        #   bidx 选第几个 M-tile，K 方向留 None（后面 K 循环里遍历）
        #   → gA 的 shape: (128, 64, num_k_tiles)
        #
        # 对于 B(N, K)：
        #   proj=(None, 1, 1) → 取 N 和 K 维度，跳过 M
        #   bidy 选第几个 N-tile
        #   → gB 的 shape: (128, 64, num_k_tiles)
        #
        # 对于 C(M, N)：
        #   proj=(1, 1, None) → 取 M 和 N 维度，跳过 K
        #   → gC 的 shape: (128, 128)
        #
        # 原生 CUDA 等价：
        #   half* gA_ptr = A + bidx * 128 * K;   // A 的起始行
        #   half* gB_ptr = B + bidy * 128 * K;   // B 的起始行
        #   float* gC_ptr = C + bidx * 128 * N + bidy * 128;  // C 的子块
        #   // 然后 K 循环中每次偏移 gA_ptr += 64, gB_ptr += 64
        gA = cute.local_tile(mA, self._cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self._cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self._cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

        # ==================================================================
        # GMEM→SMEM 拷贝的线程分工（和教程 10 的 tiled copy 完全一样）
        # ==================================================================
        # get_slice(tid)：绑定到当前线程
        # partition_S：从源（GMEM）切出当前线程负责读的元素
        # partition_D：从目标（SMEM）切出当前线程负责写的位置
        #
        # 原生 CUDA 等价：
        #   int col_tid = tid % 16;
        #   int row_tid = tid / 16;
        #   // 线程 tid 负责：第 row_tid 行，第 col_tid*4 ~ col_tid*4+3 列
        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)
        tAgA = thr_copyA.partition_S(gA)  # 源：GMEM 中 A 的这个线程的数据
        tAsA = thr_copyA.partition_D(sA)  # 目标：SMEM 中 A 的这个线程的位置
        tBgB = thr_copyB.partition_S(gB)
        tBsB = thr_copyB.partition_D(sB)

        # ==================================================================
        # MMA 分区 —— 告诉每个线程它在矩阵乘法中负责哪些数据
        # ==================================================================
        # get_slice(tid)：和 tiled_copy 的 get_slice 一模一样的用法
        #   绑定到当前线程，返回 ThrMMA 对象
        #
        # partition_A(sA)：从 SMEM 中的 A 切出当前线程做 MMA 时需要的那些元素
        # partition_B(sB)：同理
        # partition_C(gC)：从 C 切出当前线程负责计算的那些元素
        #
        # 这里和 tiled_copy 的关键区别：
        #   tiled_copy 的线程-数据映射是你自己定的（thr_layout）
        #   MMA 的线程-数据映射是硬件定的（哪个线程拿 A/B/C 的哪几个元素，
        #   由 mma.sync 指令的规范决定，你改不了）
        #   CuTE 在 MMA_Traits 内部帮你查好了这个映射，partition 自动用它。
        #
        # partition_C 返回的 shape: (MMA, MMA_M, MMA_N)
        #   MMA   = 一次 MMA 指令中，这个线程负责 C 的几个元素
        #   MMA_M = M 方向要循环几次 MMA 指令才能覆盖整个 tile
        #   MMA_N = N 方向要循环几次
        #   （和 tiled_copy 的 (CPY_ATOM, CPY_M, CPY_N) 完全对应）
        #
        # 具体数字：
        #   permutation 覆盖 C 的 32×32
        #   CTA tile 是 128×128
        #   → MMA_M = 128/32 = 4, MMA_N = 128/32 = 4
        #   → cute.gemm 内部会循环 4×4 = 16 次
        #
        # 原生 CUDA 等价（非常痛苦）：
        #   int warp_id = tid / 32;
        #   int lane_id = tid % 32;
        #   int warp_m = warp_id / 2;
        #   int warp_n = warp_id % 2;
        #   // 然后查 mma.sync 手册，根据 lane_id 算出：
        #   //   这个线程持有 A 的哪 8 个元素
        #   //   这个线程持有 B 的哪 4 个元素
        #   //   这个线程持有 C 的哪 4 个元素
        #   // 这些映射关系非常复杂，手算极易出错
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)    # 这个线程做 MMA 时从 sA 读哪些
        tCsB = thr_mma.partition_B(sB)    # 这个线程做 MMA 时从 sB 读哪些
        tCgC = thr_mma.partition_C(gC)    # 这个线程负责 C 的哪些元素

        # ==================================================================
        # 创建寄存器 fragment —— 在寄存器中给 A/B/C 开空间
        # ==================================================================
        # make_fragment_A/B/C：创建和 partition 结果形状匹配的寄存器存储
        #
        # 原生 CUDA 等价：
        #   // 查手册知道：每个线程持有 A 的 8 个 fp16, B 的 4 个, C 的 4 个 fp32
        #   uint32_t a_frag[4];  // 8 个 fp16 = 4 个 uint32
        #   uint32_t b_frag[2];  // 4 个 fp16 = 2 个 uint32
        #   float c_frag[4];     // 4 个 fp32
        #
        # CuTE 自动算好大小，你不用查手册。
        tCrA = tiled_mma.make_fragment_A(tCsA)   # A 的寄存器
        tCrB = tiled_mma.make_fragment_B(tCsB)   # B 的寄存器
        tCrC = tiled_mma.make_fragment_C(tCgC)   # C/D 的寄存器（fp32 累加器）

        # ==================================================================
        # SMEM → Register 的拷贝：LdMatrix 指令
        # ==================================================================
        # Tensor Core 要求输入数据在寄存器中有特定的排布方式。
        # 普通的 LDS（共享内存读取）可以把数据搬过来，但排布不对。
        # ldmatrix 是专门设计的指令：
        #   一个 Warp 的 32 个线程各自从 SMEM 读一行（8 个 fp16 = 16 bytes），
        #   然后硬件自动把数据重新分发到各线程的寄存器里，
        #   使得排布正好符合 mma.sync 指令的要求。
        #
        # 这就是教程 10 里说的"partition_S 和 partition_D 切出来不一样"的真实例子！
        #   partition_S（SMEM 端）：线程 0 读 SMEM 的第 0 行
        #   partition_D（寄存器端）：线程 0 拿到的不是第 0 行的全部，
        #     而是从各行中挑选出来的元素（硬件重新分发了）
        #
        # 原生 CUDA 等价：
        #   asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 ..."
        #                : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        #                : "r"(smem_ptr));
        #   // 4 个 8×8 的矩阵一次从 SMEM 加载到寄存器
        atom_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mA.element_type,
        )
        atom_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            mB.element_type,
        )

        # make_tiled_copy_A / make_tiled_copy_B：
        # 这里和 GMEM→SMEM 的 make_tiled_copy 不同！
        # 那个是你自己定义线程排布（thr_layout, val_layout）
        # 这个是让 CuTE 根据 tiled_mma 自动生成线程排布：
        #   "MMA 指令要求数据怎么排，我就怎么从 SMEM 搬"
        # 所以参数只需要 copy_atom + tiled_mma，不需要手动传 thr/val layout。
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

        thr_s2r_A = tiled_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_s2r_B.get_slice(tid)
        tCsA_view = thr_s2r_A.partition_S(sA)        # SMEM 端的切分
        tCrA_view = thr_s2r_A.retile(tCrA)           # 寄存器端的重排列
        tCsB_view = thr_s2r_B.partition_S(sB)
        tCrB_view = thr_s2r_B.retile(tCrB)
        # retile：把 MMA 的 fragment 按 ldmatrix 的视角重新排列
        # 因为 MMA partition 和 copy partition 看数据的方式不同，
        # retile 在它们之间做翻译。数据还是同一块寄存器，只是换了个看法。

        # ==================================================================
        # 主循环：沿 K 方向迭代
        # ==================================================================
        # 外层循环沿 K 方向走，每次取 bK=64 列的 A 和 B
        #
        # 原生 CUDA 等价：
        #   for (int k = 0; k < K; k += 64) {
        #       // 1. 搬数据：GMEM → SMEM
        #       // 2. 同步
        #       // 3. 搬数据：SMEM → 寄存器（ldmatrix）
        #       // 4. 计算：mma.sync
        #       // 5. 同步
        #   }
        tCrC.fill(0.0)   # C 累加器清零

        for kidx in range(mA.shape[1] // self._bK):
            # 第一步：GMEM → SMEM（向量化拷贝，每线程 64-bit load）
            # tAgA[None, None, None, kidx] 中的 kidx 选择第几个 K-tile
            # None 表示取所有元素（类似 numpy 的 :）
            cute.copy(tiled_copy_A, tAgA[None, None, None, kidx], tAsA[None, None, None])
            cute.copy(tiled_copy_B, tBgB[None, None, None, kidx], tBsB[None, None, None])
            cute.arch.sync_threads()   # 等所有线程写完 SMEM

            # 第二步：SMEM → Register（ldmatrix）
            cute.copy(tiled_s2r_A, tCsA_view, tCrA_view)
            cute.copy(tiled_s2r_B, tCsB_view, tCrB_view)

            # 第三步：MMA 计算 —— D = A × B + C
            # 一行搞定！内部自动循环 4×4 = 16 次 mma.sync 指令
            #
            # 原生 CUDA 等价（循环 + 内联汇编）：
            #   for (int mm = 0; mm < 4; mm++)
            #     for (int nn = 0; nn < 4; nn++)
            #       for (int kk = 0; kk < 64/16; kk++)
            #         asm("mma.sync.aligned.m16n8k16 ...");
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

        # ==================================================================
        # 写回 GMEM
        # ==================================================================
        # 把 fp32 累加器转成 fp16，然后写回 GMEM
        #
        # 这里用 atom_store（不是 tiled_copy）：
        #   因为 C 的寄存器排布是 MMA 定义的，不需要额外的线程排布信息，
        #   每个线程直接把自己的 C fragment 写到 GMEM 对应位置即可。
        #
        # 原生 CUDA 等价：
        #   for (int i = 0; i < 4; i++) {
        #       half val = __float2half(c_frag[i]);
        #       C_ptr[...] = val;   // 地址根据 lane_id 和 warp_id 手动算
        #   }
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for i in range(cute.size(tCrC_out)):
            tCrC_out[i] = cutlass.Float16(tCrC[i])   # fp32 → fp16
        cute.copy(atom_store, tCrC_out, tCgC)         # 寄存器 → GMEM


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((N, K), device="cuda", dtype=torch.float16)
    ref = torch.matmul(A, B.T)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)

    print("=" * 60)
    print("WMMA Tensor Core GEMM（向量化拷贝 + LdMatrix + WMMA）")
    print("=" * 60)

    C = torch.empty((M, N), device="cuda", dtype=torch.float16)
    C_ = from_dlpack(C, assumed_align=16)

    gemm = GemmWMMA()
    compiled = cute.compile(gemm, A_, B_, C_)
    compiled(A_, B_, C_)

    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "WMMA GEMM 验证失败！"
    print("✅ WMMA GEMM 正确性验证通过！")

    time_us = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
    tflops = (2 * M * N * K) / (time_us * 1e6)
    print(f"⏱  WMMA GEMM 耗时: {time_us:.2f} µs | TFLOPS: {tflops:.4f}")

    # ---- PyTorch 性能对比 ----
    print("\n" + "=" * 60)
    print("📊 性能对比：CuTeDSL WMMA vs PyTorch")
    print("=" * 60)

    C_pt = torch.empty_like(ref)
    for _ in range(10):
        torch.matmul(A, B.T, out=C_pt)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    num_iter = 100
    for _ in range(num_iter):
        torch.matmul(A, B.T, out=C_pt)
    end.record()
    torch.cuda.synchronize()
    pt_time_us = start.elapsed_time(end) * 1000 / num_iter
    pt_tflops = (2 * M * N * K) / (pt_time_us * 1e6)
    print(f"⏱  PyTorch   耗时: {pt_time_us:.2f} µs | TFLOPS: {pt_tflops:.4f}")
    print(f"⏱  WMMA GEMM 耗时: {time_us:.2f} µs | TFLOPS: {tflops:.4f}")
    ratio = pt_time_us / time_us
    print(f"📊 CuTeDSL / PyTorch 速度比: {ratio:.2f}x {'(更快 ✅)' if ratio > 1 else '(更慢 ⚠️)'}")

    print("\n🎉 教程 05 全部完成！")
