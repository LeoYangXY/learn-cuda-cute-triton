"""
=============================================================================
GPU 数据搬运硬件路径全面测试 (RTX 5050 / SM120)
=============================================================================

对于 RTX 5050 (SM120)，数据搬运只有两种硬件资源：
  - LSU (Load/Store Unit)：每个 SM 有 32 个
  - TMA (Tensor Memory Accelerator)：每个 SM 有 1 个

下面展示四种软件写法，它们走不同的硬件资源：

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  方法 1: Load/Store (LSU 同步)                                          │
│    走 LSU 硬件单元                                                       │
│    PTX: ld.global / st.global / ld.shared / st.shared                   │
│    SASS: LDG / STG / LDS / STS                                         │
│    路径: GMEM → Reg → (可选 SMEM) → 计算                                │
│    注意: GMEM→SMEM 必须经过 Register 中转 (LDG + STS 两条指令)          │
│                                                                         │
│  方法 2: cp.async (LSU 异步)                                            │
│    走 LSU 硬件单元 (异步队列)                                             │
│    PTX: cp.async.cg.shared.global                                       │
│    SASS: LDGSTS                                                         │
│    路径: GMEM → SMEM (一步完成, 绕过寄存器, 异步)                        │
│                                                                         │
│  方法 3: ldmatrix (LSU 矩阵搬运)                                        │
│    走 LSU 硬件单元 (warp 协作 + 数据重排)                                 │
│    PTX: ldmatrix.sync.aligned.m8n8.x4.shared.b16                       │
│    SASS: LDSM                                                           │
│    路径: SMEM → Reg (warp 内 32 线程协作, 自动排成 Tensor Core layout)   │
│                                                                         │
│  方法 4: TMA (独立 TMA 硬件单元)                                         │
│    走 TMA 硬件单元 (不占 LSU)                                             │
│    PTX: cp.async.bulk.tensor.Xd.shared::cluster.global                  │
│    SASS: UTMALDG                                                        │
│    路径: GMEM → SMEM (1 个线程发指令, 硬件自动算地址 + swizzle)           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

四种方法都做 C = A + B (FP16)，对比正确性和性能。
=============================================================================
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float16
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.testing import benchmark, JitArguments

N = 1024 * 1024  # 1M 元素
BLOCK_SIZE = 256
VEC = 8  # 每线程 8 个 FP16 = 128 bits


# =============================================================================
# 方法 1: Load/Store —— LSU 同步 (CopyUniversalOp)
# =============================================================================
# 硬件: LSU 管道 (同步), 每个 SM 有 32 个 LSU
# 等价原生 CUDA:
#   标量版: c[idx] = a[idx] + b[idx];                    → LDG.32 / STG.32
#   向量版: float4 a = *(float4*)&gA[idx];               → LDG.128
# 数据路径: GMEM --LDG--> Register --计算--> Register --STG--> GMEM
#           如果要到 SMEM: GMEM --LDG--> Reg --STS--> SMEM (两步, 占寄存器)

ELEMS_PER_BLOCK_1 = BLOCK_SIZE * VEC  # 256 × 8 = 2048

@cute.kernel
def add_load_store_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    tv_layout: cute.Layout, block_shape: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blkA = cute.local_tile(gA, block_shape, (bidx,))
    blkB = cute.local_tile(gB, block_shape, (bidx,))
    blkC = cute.local_tile(gC, block_shape, (bidx,))

    # 128-bit 向量化: LDG.E.128 / STG.E.128
    # make_copy_atom: 描述一次原子拷贝操作的"性质" + "每次搬多少"
    #   - 第 1 个参数 (Op): 用什么硬件指令 (CopyUniversalOp = 普通 LDG/STG)
    #   - 第 2 个参数: 元素数据类型
    #   - num_bits_per_copy: 每次搬运的 bit 数 (决定向量化宽度)
    # 注意: copy_atom 本身不知道有多少线程、tile 多大,
    #       这些信息在 make_tiled_copy 时才传入。
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gA.element_type,
        num_bits_per_copy=gA.element_type.width * VEC,  # 16 × 8 = 128 bits
    )
    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, block_shape)
    thr_copy = tiled_copy.get_slice(tidx)

    tA_src = thr_copy.partition_S(blkA)
    tB_src = thr_copy.partition_S(blkB)
    tC_dst = thr_copy.partition_D(blkC)

    # GMEM → Register (LDG.128)
    rA = cute.make_fragment_like(tA_src)
    rB = cute.make_fragment_like(tB_src)
    cute.copy(tiled_copy, tA_src, rA)
    cute.copy(tiled_copy, tB_src, rB)

    # 计算 (在 Register 中)
    rC = cute.make_fragment_like(rA)
    for i in range(cute.size(rC)):
        rC[i] = rA[i] + rB[i]

    # Register → GMEM (STG.128)
    cute.copy(tiled_copy, rC, tC_dst)


@cute.jit
def add_load_store(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    n = a.shape[0]
    tv_layout = cute.make_layout((BLOCK_SIZE, VEC), stride=(VEC, 1))
    block_shape = (ELEMS_PER_BLOCK_1,)
    grid = (n // ELEMS_PER_BLOCK_1, 1, 1)
    add_load_store_kernel(a, b, c, tv_layout, block_shape).launch(
        grid=grid, block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 方法 2: cp.async —— LSU 异步队列 (CopyG2SOp / LDGSTS)
# =============================================================================
# 硬件: LSU 管道的异步队列 (Ampere sm80 引入)
# 等价原生 CUDA:
#   __pipeline_memcpy_async(&sA[tid*8], &gA[tid*8], 16);
#   __pipeline_commit();
#   __pipeline_wait_prior(0);
# 数据路径:
#   GMEM → SMEM (一步 LDGSTS, 绕过寄存器, 异步!)
#   对比 Load/Store: GMEM --LDG--> Reg --STS--> SMEM (两步, 占寄存器)

ELEMS_PER_BLOCK_2 = BLOCK_SIZE * VEC

@cute.kernel
def add_cpasync_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    tv_layout: cute.Layout, block_shape: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blkA = cute.local_tile(gA, block_shape, (bidx,))
    blkB = cute.local_tile(gB, block_shape, (bidx,))
    blkC = cute.local_tile(gC, block_shape, (bidx,))

    # ---- SMEM 分配 ----
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_layout(block_shape)
    sA = smem.allocate_tensor(gA.element_type, smem_layout, 128)
    sB = smem.allocate_tensor(gA.element_type, smem_layout, 128)

    # ---- Step 1: cp.async GMEM → SMEM (LDGSTS, 绕过寄存器) ----
    # cp.async 和直接 a[idx] = b[idx] 在 CuTE DSL 层面的API 调用模式几乎一模一样，只有make_copy_atom中的op的区别
    # make_copy_atom 只描述: 用什么指令 + 每次搬多少 bit
    #   - CopyG2SOp() = cp.async 异步指令 (GMEM→SMEM, 绕过寄存器)
    #   - num_bits_per_copy = 128 → 每次搬 128 bit = 8 个 FP16
    # 它不知道有多少线程、tile 多大, 这些在 make_tiled_copy 时传入。
    cpasync_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(), gA.element_type,
        num_bits_per_copy=gA.element_type.width * VEC,
    )
    cpasync_tiled = cute.make_tiled_copy(cpasync_atom, tv_layout, block_shape)
    thr_cpasync = cpasync_tiled.get_slice(tidx)

    tA_g = thr_cpasync.partition_S(blkA)
    tA_s = thr_cpasync.partition_D(sA)
    tB_g = thr_cpasync.partition_S(blkB)
    tB_s = thr_cpasync.partition_D(sB)

    cute.copy(cpasync_tiled, tA_g, tA_s)
    cute.copy(cpasync_tiled, tB_g, tB_s)

    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # ---- Step 2: SMEM → Register (LDS) + 计算 + STG ----
    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), gA.element_type,
        num_bits_per_copy=gA.element_type.width * VEC,
    )
    simt_tiled = cute.make_tiled_copy(simt_atom, tv_layout, block_shape)
    thr_simt = simt_tiled.get_slice(tidx)

    tsA = thr_simt.partition_S(sA)
    tsB = thr_simt.partition_S(sB)

    rA = cute.make_fragment_like(tsA)
    rB = cute.make_fragment_like(tsB)
    cute.copy(simt_tiled, tsA, rA)
    cute.copy(simt_tiled, tsB, rB)

    rC = cute.make_fragment_like(rA)
    for i in range(cute.size(rC)):
        rC[i] = rA[i] + rB[i]

    tC_dst = thr_simt.partition_D(blkC)
    cute.copy(simt_tiled, rC, tC_dst)


@cute.jit
def add_cpasync(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    n = a.shape[0]
    tv_layout = cute.make_layout((BLOCK_SIZE, VEC), stride=(VEC, 1))
    block_shape = (ELEMS_PER_BLOCK_2,)
    grid = (n // ELEMS_PER_BLOCK_2, 1, 1)
    add_cpasync_kernel(a, b, c, tv_layout, block_shape).launch(
        grid=grid, block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 方法 3: ldmatrix —— LSU 矩阵搬运 (LdMatrix8x8x16bOp / LDSM)
# =============================================================================
# 硬件: LSU 管道 (warp 协作模式, Turing sm75 引入)
# 等价原生 CUDA (只能内联 PTX):
#   asm volatile(
#       "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
#       : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
#
# 数据路径: SMEM → Register (warp 内 32 线程协作, 自动重排成 Tensor Core layout)
# 关键: 32 个线程各提供 1 个 SMEM 地址, 但拿到的数据经过 warp 内 shuffle
#       输出寄存器布局直接匹配 Tensor Core 输入要求
#
# 注意: ldmatrix 只做 SMEM→Reg! 本 kernel 组合使用:
#   autovec_copy (GMEM→SMEM) + ldmatrix (SMEM→Reg) + 加法 + 逐元素写回
#
# ldmatrix 是为 Tensor Core (MMA) 服务的, 需要配合 tiled_mma 使用
# 这里用 WMMA (MmaF16BF16Op) 的分区来驱动 ldmatrix
#
# =============================================================================
# 对比 MMA 和 Copy
# =============================================================================
#
#   MMA 侧                                     Copy 侧
#   ──────────────────────────────────          ──────────────────────────────────
#   MMA_Op (一条 PTX 硬件指令)                   Copy_Op (一条 PTX 硬件指令)
#     MmaF16BF16Op                                CopyUniversalOp (LDG/STG)
#     MmaUniversalOp                              CopyG2SOp (cp.async/LDGSTS)
#                                                 LdMatrix8x8x16bOp (LDSM)
#                                                            ↓
#   (没有显式的 Atom 层,                         CopyAtom (显式创建)
#    直接传 Op 给 make_tiled_mma,                  = Op + 数据类型 + 每次搬多少 bit
#    内部自动处理线程-数据映射)                     cute.make_copy_atom(op, dtype, ...)
#                          ↓                                     ↓
#   TiledMMA                                     TiledCopy
#     = 多个 Warp 排成网格, 协作覆盖更大 tile      = 多个线程排成网格, 协作搬运整个 tile
#     cute.make_tiled_mma(op, atom_layout, ...)   cute.make_tiled_copy(atom, tv_layout, block_shape)
#                          ↓                                     ↓
#   ThrMMA                                       ThrCopy
#     = 绑定到具体线程 tid 后的视图                = 绑定到具体线程 tid 后的视图
#     tiled_mma.get_slice(tid)                    tiled_copy.get_slice(tid)
#     → partition_A / partition_B / partition_C    → partition_S / partition_D / retile
#
# 联系: 
#   从 TiledMMA 自动推导出 TiledCopy (用于 ldmatrix / stmatrix)
#   cute.make_tiled_copy_A(atom, tiled_mma)  → 生成 TiledCopy, 线程排布从 MMA 的 A 操作数推导
#   cute.make_tiled_copy_B(atom, tiled_mma)  → 同理, 从 B 操作数推导
#   cute.make_tiled_copy_C(atom, tiled_mma)  → 同理, 从 C 操作数推导 (用于写回)
#
# =============================================================================
# ldmatrix vs cp.async: 都是数据搬运，为什么写法不同?
# =============================================================================
#
# 核心问题: 创建 TiledCopy 时, 线程排布 (tv_layout) 从哪来?
#
# 拿到 tiled_mma 之后, 我们知道了这些 Warp 要负责的一块区域 (比如 32×32 的 C)。
# 但实际数据搬运的时候, 要落到 thread 维度 —— 每个线程具体从 SMEM 的哪里读、
# 写到寄存器的哪个位置, 这就是 tv_layout 要描述的东西。
#
# 这里有两种情况:
#
# 【cp.async (GMEM→SMEM)】数据搬进 SMEM 怎么排都行, tv_layout 你自己定:
#
#   atom = cute.make_copy_atom(CpAsyncG2SOp(), dtype, num_bits_per_copy=128)
#   tiled_copy = cute.make_tiled_copy(atom, tv_layout, block_shape)
#                                           ^^^^^^^^^ 你自己传，可以自己指定任意的
#
# 【ldmatrix (SMEM→Register)】寄存器排布由 MMA 硬件指令定死, 没有自由度。
#   用 make_tiled_copy 的话, 你得自己查 PTX 手册算出正确的 tv_layout。
#   但 tiled_mma 里已经记录了这个线程-数据映射!
#   所以 CuTE 提供了 make_tiled_copy_A/B/C, 直接从 tiled_mma 自动推导 tv_layout:
#
#   atom = cute.make_copy_atom(LdMatrix8x8x16bOp(False, 4), Float16)
#   s2r_copy = cute.make_tiled_copy_A(atom, tiled_mma)
#                                           ^^^^^^^^^ 从 tiled_mma 自动推导!
#
# 总结:
#   make_tiled_copy(atom, tv_layout, block_shape)  → 你手动指定 tv_layout
#   make_tiled_copy_A/B/C(atom, tiled_mma)         → 从 tiled_mma 自动推导 tv_layout


M_TILE = 64
K_TILE = 64
THREADS_LDMATRIX = 128  # 1 warpgroup (4 warps)

@cute.kernel
def add_ldmatrix_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 2D 分块
    tile_shape = (M_TILE, K_TILE)
    blkA = cute.local_tile(gA, tile_shape, (bidx, 0))
    blkB = cute.local_tile(gB, tile_shape, (bidx, 0))
    blkC = cute.local_tile(gC, tile_shape, (bidx, 0))

    # ---- SMEM 分配 (使用 swizzle layout 避免 bank conflict) ----

    # 这是一个共享内存分配器对象，类似于 CUDA 中声明 __shared__ 变量。
    # 它不是真的 malloc，而是在编译期静态地计算每个 tensor 在 SMEM 中的偏移量。
    # 一个 kernel 中所有的 smem.allocate_tensor() 调用会被编译器汇总，最终生成一个总的 extern __shared__ 声明
    smem = cutlass.utils.SmemAllocator()
    smem_layout = cute.make_ordered_layout(tile_shape, order=(1, 0))
    sA = smem.allocate_tensor(gA.element_type, smem_layout, 128)
    sB = smem.allocate_tensor(gA.element_type, smem_layout, 128)

    # ---- Step 1: GMEM → SMEM (autovec_copy, 简单直接) ----
    # autovec_copy 是 CuTE 提供的自动向量化的 Load/Store 函数
    # 自动推断 tv_layout：根据 blockDim（当前 block 的线程数）和 tensor 的形状/对齐情况，自动决定每个线程搬多少元素、用多宽的向量指令
    # 自动向量化：如果对齐允许，会自动用 128-bit（8 个 FP16）的向量加载
    # 自动分区：自动把 tensor 按线程数切分，每个线程搬自己的那份
    # 执行 copy：等价于 CopyUniversalOp 的同步 Load/Store。隐式的使用了寄存器，做到了gm->smem。看起来像走的是copy.async这条路线，实际上是 CopyUniversalOp 的自动化版本
    cute.autovec_copy(blkA, sA)
    cute.autovec_copy(blkB, sB)
    cute.arch.sync_threads()


    # ══════════════════════════════════════════════════════════════════
    # make_tiled_mma 的三个参数 (和 make_tiled_copy 对比)
    # ══════════════════════════════════════════════════════════════════
    #
    #   make_tiled_copy(                    make_tiled_mma(
    #       copy_atom,       ← 最小操作        mma_op,         ← 最小操作
    #       tv_layout,       ← 线程怎么排      atom_layout_mnk,← Warp 怎么排
    #       block_shape,     ← 覆盖多大区域    permutation_mnk,← 覆盖多大区域
    #   )                                   )
    #
    # ══════════════════════════════════════════════════════════════════

    # 参数 1: mma_op — 选一条硬件指令
    # → mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    # 一个 Warp (32 线程) 协作完成: A(16×16,fp16) × B(16×8,fp16) + C(16×8,fp32)
    mma_op = cute.nvgpu.warp.MmaF16BF16Op(
        ab_dtype=Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16),
    )

    # 参数 2: atom_layout_mnk — Warp 怎么排 (类似 tiled_copy 的 tv_layout)
    # 就像 tv_layout 告诉 CuTE "256个线程排成 8×16 的网格" 一样,
    # atom_layout_mnk 告诉 CuTE "4 个 Warp 排成 M 方向 2 个、N 方向 2 个":
    #
    #            N 方向
    #        ┌────────┬────────┐
    #        │ Warp0  │ Warp1  │   每个 Warp 算 C 的 16×8
    # M 方向 │ 16×8   │ 16×8   │
    #        ├────────┼────────┤
    #        │ Warp2  │ Warp3  │
    #        │ 16×8   │ 16×8   │
    #        └────────┴────────┘
    #        4 个 Warp 直接覆盖: 32×16 的 C 子块
    atom_layout_mnk = (2, 2, 1)  # M 方向 2 个 Warp, N 方向 2 个, K 方向 1 个

    # 参数 3: permutation_mnk — 这些 Warp 合在一起要覆盖 C 的多大子块
    #
    # ══════ 思考方式: 先画覆盖图, 再反推参数 ══════
    #
    # 你先想好: "我希望这 4 个 Warp 一次覆盖 C 的多大区域?"
    # 然后 permutation_mnk 就是这个区域的 (M, N, K) 大小。
    #
    # ── 计算公式 ──
    #   permutation_M = atom_layout_M × mma_M × repeat_M   (repeat_M 通常 = 1)
    #   permutation_N = atom_layout_N × mma_N × repeat_N   (repeat_N 通常 = 2)
    #   permutation_K = atom_layout_K × mma_K              (K 方向不 repeat)
    #
    # ── 示例 A: 本代码的配置 ──
    #
    #   atom_layout = (2,2,1), MMA指令 = (16,8,16)
    #   最小覆盖: M=2×16=32, N=2×8=16 → 32×16 的 C
    #   但 32×16 太窄了 (N 只有 16 列), A 的数据复用率低。
    #   所以让每个 Warp 在 N 方向多算一倍 → permutation = (32, 32, 16):
    #
    #        N 方向 (32 列)
    #    ┌────────┬────────┬────────┬────────┐
    #    │ Warp0  │ Warp1  │ Warp0  │ Warp1  │  ← 同一个 Warp 在 N 方向
    #    │ 16×8   │ 16×8   │ 16×8   │ 16×8   │    重复算了 2 次
    #    ├────────┼────────┼────────┼────────┤
    #    │ Warp2  │ Warp3  │ Warp2  │ Warp3  │
    #    │ 16×8   │ 16×8   │ 16×8   │ 16×8   │
    #    └────────┴────────┴────────┴────────┘
    #    覆盖: 32×32 的 C, 每次消耗 K=16
    #
    # ── 示例 B: 不用 permutation (×1) ──
    #
    #   permutation = (32, 16, 16)
    #    ┌────────┬────────┐
    #    │ Warp0  │ Warp1  │   覆盖: 32×16 的 C
    #    │ 16×8   │ 16×8   │   更窄, A 复用率低
    #    ├────────┼────────┤
    #    │ Warp2  │ Warp3  │
    #    │ 16×8   │ 16×8   │
    #    └────────┴────────┘
    #
    # ── CTA tile 和 permutation 的关系 ──
    # permutation_mnk (32, 32, 16)
    # = 这些 Warp 执行一轮 MMA 指令组覆盖的区域
    # ≠ 这个 block 负责处理的区域
    # CTA tile (比如 128, 128, 64)
    #     = 这个 block 负责处理的区域

    #   permutation 是一次 MMA 指令组覆盖的区域 (32×32)
    #   CTA tile 是整个 block 要处理的区域 (比如 64×64 或 128×128)
    #   cute.gemm 内部自动循环: CTA_tile / permutation = 需要几轮
    #   例如 (64,64) / (32,32) = 2×2 = 4 轮 MMA 指令组

    permutation_mnk = (
        atom_layout_mnk[0] * 16,       # 2×16 = 32  (M: Warp数 × MMA的M)
        atom_layout_mnk[1] * 8 * 2,    # 2×8×2 = 32 (N: Warp数 × MMA的N × repeat=2)
        atom_layout_mnk[2] * 16,       # 1×16 = 16  (K: 每次 MMA 消耗的 K)
    )




    # 组装成 TiledMMA
    # tiled_mma 本身不是计算, 是计算的"配方" (编译期数据结构), 记录了:
    #   1. 用什么 MMA 指令
    #   2. 几个 Warp, 怎么排
    #   3. 这些warp一次执行能覆盖 C 的多大子块，比如上面的permutation_mnk (32, 32, 16)那么就是覆盖了C的 32×32的子块，最后的那个16是K维度的，控制的是内部循环的次数
    #   4. 每个线程持有 A/B/C 的哪些元素 (硬件定死的映射表)
    # 真正的计算在 cute.gemm(tiled_mma, D, A, B, C) 时才发生。
    # 类比: tiled_copy 是搬运配方, tiled_mma 是计算配方。
    tiled_mma = cute.make_tiled_mma(
        op_or_atom=mma_op,
        atom_layout_mnk=atom_layout_mnk,
        permutation_mnk=permutation_mnk,
    )

    # ══════════════════════════════════════════════════════════════════
    # 创建 ldmatrix 搬运计划
    # ══════════════════════════════════════════════════════════════════

    # ── 关于 LdMatrix8x8x16bOp ──
    #
    # 8x8x16b 是 PTX 指令 ldmatrix 的固定规格 (硬件只提供这一种):
    #   8×8 = 矩阵块大小, 16b = 每个元素 16 bit (FP16/BF16)
    #
    # 能改的是 num_matrices: 一次搬几个 8×8 块
    #   LdMatrix8x8x16bOp(False, 4) → ldmatrix.m8n8.x4 → 每线程拿 4 个寄存器 (8 个 FP16)
    #   LdMatrix8x8x16bOp(False, 2) → ldmatrix.m8n8.x2 → 每线程拿 2 个寄存器 (4 个 FP16)
    #   LdMatrix8x8x16bOp(False, 1) → ldmatrix.m8n8.x1 → 每线程拿 1 个寄存器 (2 个 FP16)
    #
    # 改 num_matrices 不影响线程排布 (谁负责哪些数据, 由 tiled_mma 决定),
    # 只影响每轮搬运量: x4 一轮搬满, x2 需要两轮, x1 需要四轮。
    # 实际上总是用 x4, 因为 mma.sync.m16n8k16 的 A 操作数刚好需要每线程 4 个寄存器。
    #
    # ldmatrix 是 warp 级指令: 32 个线程协作执行, 每个线程提供一个 SMEM 地址,
    # 硬件自动在 warp 内 shuffle, 把数据重排成 Tensor Core 要求的寄存器布局。
    ldmatrix_atom = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), Float16,
    )
    s2r_copy = cute.make_tiled_copy_A(ldmatrix_atom, tiled_mma)#根据每次的atom和tiled_mma这个warp需要负责的数据量，自动推导出tv_layout
    thr_copy = s2r_copy.get_slice(tidx)

    # ══════════════════════════════════════════════════════════════════
    # MMA 视角 vs Copy 视角: 同一块 sA, 两种不同的切分方式
    # ══════════════════════════════════════════════════════════════════
    #
    # 同一块 SMEM 数据 (sA), 不同"视角"看到的切分方式不同:
    #
    #   MMA 视角 (partition_A):
    #     "我做矩阵乘法时, 这个线程需要 A 的哪些元素?"
    #     → 由 MMA 指令的硬件映射决定, 有广播 (多线程可能读同一元素)
    #
    #   Copy 视角 (partition_S):
    #     "我用 ldmatrix 搬数据时, 这个线程从 SMEM 的哪里读?"
    #     → 由 ldmatrix 指令的硬件映射决定
    #
    # 两个 partition 切的是同一块 sA, 但切法不同!
    # 这就是为什么需要 retile: 在两种视角之间做翻译。
    #
    # ──────────────────────────────────────────────────────────────────

    #partition_A,partition_B,partition_C:专门用于mma相关切分;partition_S,partition_D:专门用于copy相关切分

    # MMA 视角的分区: 按 MMA 的线程-数据映射切分 sA/sB
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)    # MMA 视角: 这个线程做乘法时需要 sA 的哪些元素
    tCsB = thr_mma.partition_A(sB)

    # Copy 视角的分区: 按 ldmatrix 的线程-数据映射切分 sA/sB
    thr_copy_sA = thr_copy.partition_S(sA)  # Copy 视角: 这个线程用 ldmatrix 从 sA 的哪里读
    thr_copy_sB = thr_copy.partition_S(sB)

    # 分配寄存器 fragment (按 MMA 的 layout, 因为最终是 MMA 指令来消费这些数据)
    tCrA = thr_mma.make_fragment_A(tCsA)
    tCrB = thr_mma.make_fragment_A(tCsB)

    # ── retile: 同一块寄存器, 换一种索引方式去看它 ──
    #
    # 假设一个线程的寄存器里有 8 个 FP16: [r0, r1, r2, r3, r4, r5, r6, r7]
    #
    #   MMA 视角 (tCrA):               Copy 视角 (thr_copy.retile(tCrA)):
    #     shape = (4, 2)                 shape = (2, 4)
    #     ┌────┬────┐                    ┌────┬────┬────┬────┐
    #     │ r0 │ r1 │                    │ r0 │ r1 │ r2 │ r3 │
    #     │ r2 │ r3 │                    │ r4 │ r5 │ r6 │ r7 │
    #     │ r4 │ r5 │                    └────┴────┴────┴────┘
    #     │ r6 │ r7 │
    #     └────┴────┘
    #   物理上还是同一片 r0~r7, 只是 shape/stride 变了, 不搬数据。
    #
    # 为什么需要? 因为 cute.copy 要求 src 和 dst 的 shape 匹配:
    #   cute.copy(ldmatrix_atom, thr_copy_sA, thr_copy_rA)
    #                           ^^^^^^^^^^^  ^^^^^^^^^^^
    #                           SMEM 端       Register 端
    #                           Copy 视角     也必须是 Copy 视角!
    # 如果直接传 MMA 视角的 tCrA, shape 对不上会报错。
    #
    # 整个流程:
    #   1. tCrA = thr_mma.make_fragment_A(...)       ← MMA 视角分配寄存器
    #   2. thr_copy_rA = thr_copy.retile(tCrA)       ← 换成 Copy 视角 (不搬数据)
    #   3. cute.copy(..., thr_copy_sA, thr_copy_rA)   ← ldmatrix 按 Copy 视角写入寄存器
    #   4. cute.gemm(..., tCrA, ...)                   ← MMA 按 MMA 视角读同一块寄存器
    #   步骤 3 和 4 操作的是同一片物理寄存器, 只是通过不同的 shape/stride 去访问。
    thr_copy_rA = thr_copy.retile(tCrA)
    thr_copy_rB = thr_copy.retile(tCrB)

    # 执行 ldmatrix: SMEM → Register (LDSM 指令)
    cute.copy(ldmatrix_atom, thr_copy_sA, thr_copy_rA)
    cute.copy(ldmatrix_atom, thr_copy_sB, thr_copy_rB)

    # ---- Step 3: 在寄存器中计算 (用 MMA fragment) ----
    for i in range(cute.size(tCrA)):
        tCrA[i] = tCrA[i] + tCrB[i]

    # ---- Step 4: Register → SMEM(sC) → GMEM ----
    # MMA 的 partition_A 会让多个线程读同一个 SMEM 元素(广播)
    # 写回时不能直接用 tCsA (会有多线程写同一位置的冲突)
    # 解决: 分配独立 sC，用 MMA 分区视角写入 (只写属于自己的部分)
    # 不过其实，这段代码是一个 workaround，
    # 因为这个 kernel 本来就不是在做真正的 GEMM，只是借用 MMA 的 partition 来演示 ldmatrix。
    # 在真正的 GEMM kernel 里，结果是通过 partition_C（没有广播）写回的，不会有这个问题
    sC = smem.allocate_tensor(gA.element_type, smem_layout, 128)
    tCsC = thr_mma.partition_A(sC)
    for i in range(cute.size(tCrA)):
        tCsC[i] = tCrA[i]
    cute.arch.sync_threads()

    # SMEM → GMEM
    cute.autovec_copy(sC, blkC)


@cute.jit
def add_ldmatrix(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    M = a.shape[0]
    grid = (cute.ceil_div(M, M_TILE), 1, 1)
    add_ldmatrix_kernel(a, b, c).launch(
        grid=grid, block=(THREADS_LDMATRIX, 1, 1))


# =============================================================================
# 方法 4: TMA —— 独立 TMA 硬件单元 (CopyBulkTensorTileG2SOp / UTMALDG)
# =============================================================================
# 硬件: TMA (Tensor Memory Accelerator), 每个 SM 有 1 个, 独立于 LSU
# 等价原生 CUDA (只能内联 PTX):
#   // Host: cuTensorMapEncode(&desc, ...);
#   // Kernel (1 个线程):
#   asm("cp.async.bulk.tensor.1d.shared::cluster.global
#       .mbarrier::complete_tx::bytes [%0], [%1, {%2}], [%3];"
#       :: "r"(smem), "l"(&desc), "r"(coord), "r"(mbar));
# 数据路径: GMEM → SMEM (TMA 硬件, 不占 LSU, 不占寄存器)
# 只需 1 个线程发指令, 硬件自动: 多维地址计算 + swizzle + 搬整块 tile
#
# ═══ TMA vs 前三种方法的核心区别 ═══
#
#   方法 1-3: 每个线程各自发 load/store 请求, 走 LSU (32 个工人各自搬砖)
#     cute.copy(tiled_copy, src, dst)   ← 256 个线程各搬各的
#
#   TMA:      1 个线程发 1 条指令, TMA 硬件自己搬整块 tile (一台叉车)
#     cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar)  ← 1 条指令搬整块
#     + mbarrier 做异步同步
#
# ═══ TMA 的两步走 ═══
#
#   第 1 步 (Host 侧 / @cute.jit): 创建 TMA descriptor
#     TMA 硬件需要一个"描述符": tensor 的地址、形状、stride、数据类型等
#     这个描述符在 kernel launch 前就要准备好
#     → cpasync.make_tiled_tma_atom(op, tensor, smem_layout, tile_shape)
#
#   第 2 步 (Kernel 内): 发射 TMA + 用 mbarrier 等待完成
#     → cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar)
#     → cute.arch.mbarrier_wait(mbar, phase)
#
# ═══ mbarrier (memory barrier) ═══
#
#   TMA 是异步的 —— 你发完指令, TMA 在后台搬, 什么时候搬完你不知道。
#   mbarrier 就是同步机制:
#     1. mbarrier_init(ptr, cnt=1)            → 初始化, 期望 1 个 arrival
#     2. mbarrier_arrive_and_expect_tx(ptr, N) → 告诉它 "预计搬 N 字节"
#     3. cute.copy(..., tma_bar_ptr=ptr)       → TMA 搬完后自动通知 mbarrier
#     4. mbarrier_wait(ptr, phase)             → 阻塞直到所有数据搬完

@cute.kernel
def add_tma_kernel(
    tma_atom_a: cute.CopyAtom, mA_tma: cute.Tensor,
    tma_atom_b: cute.CopyAtom, mB_tma: cute.Tensor,
    gC: cute.Tensor,
    smem_layout: cute.Layout,
    shared_storage: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    dtype = gC.element_type

    # ---- 第 1 步: SMEM 分配 ----
    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(shared_storage)
    sA = storage.sA.get_tensor(smem_layout)
    sB = storage.sB.get_tensor(smem_layout)
    mbar_ptr = storage.mbar_ptr.data_ptr()

    # ---- 第 2 步: 初始化 mbarrier ----
    if tidx == 0:
        cute.arch.mbarrier_init(mbar_ptr, cnt=1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()

    # ---- 第 3 步: TMA 分区 ----
    blkA_tma = cute.local_tile(mA_tma, (ELEMS_PER_BLOCK_1,), (bidx,))
    blkB_tma = cute.local_tile(mB_tma, (ELEMS_PER_BLOCK_1,), (bidx,))

    # tma_partition: 告诉 TMA "从 GMEM 的哪块搬到 SMEM 的哪块"
    # 返回 (SMEM 端视图, GMEM 端视图), 之后 cute.copy 用
    #
    # 为什么不用 tiled_copy 的 partition_S/D?
    #   LSU: 256 个线程各搬各的 → 按线程切分, 需要 tidx
    #     thr_copy = tiled_copy.get_slice(tidx)   ← "我是第几号线程"
    #     src = thr_copy.partition_S(blkA)         ← 这个线程搬的那一小块
    #   TMA: 硬件搬整块 tile, 不按线程分 → 按整块切分, 不需要 tidx
    #     tAsA, tAgA = cpasync.tma_partition(...)  ← 整块 tile 的 SMEM/GMEM 视图
    #   LSU 是 "每个工人领自己的活", TMA 是 "叉车搬整箱"
    
    
    # 下面两个参数是关于 Cluster 的:
    #   Cluster (Hopper sm90 引入): 多个 CTA (thread block) 组成一组, 可以跨 block 访问对方的 SMEM
    #   TMA 需要知道 cluster 信息, 因为它可以把数据搬到 cluster 内其他 CTA 的 SMEM 里
    #   例: cta_layout=(4,) → 4 个 CTA 组成 cluster, cta_coord=2 → 我是第 3 个 CTA
    #   本例不用 cluster, 所以 cta_coord=0, cta_layout=(1,)
    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,                      # cta_coord: 当前 CTA 在 cluster 中的坐标 (不用 cluster, 填 0)
        cute.make_layout(1),    # cta_layout: cluster 布局 (不用 cluster, 就 1 个 CTA)
        smem_tensor=sA, gmem_tensor=blkA_tma,
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0, cute.make_layout(1),  # 同上
        smem_tensor=sB, gmem_tensor=blkB_tma,
    )

    # ---- 第 4 步: 发射 TMA (1 个线程) ----
    tma_bytes = cute.size_in_bytes(dtype, smem_layout) * 2
    if tidx == 0:
        cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_bytes)
        cute.copy(tma_atom_a, tAgA, tAsA, tma_bar_ptr=mbar_ptr)
        cute.copy(tma_atom_b, tBgB, tBsB, tma_bar_ptr=mbar_ptr)

    # ---- 第 5 步: 等 TMA 搬完 ----
    cute.arch.mbarrier_wait(mbar_ptr, 0)

    #====这下面就和tma没有关系了====

    # ---- 第 6 步: SMEM → Reg → 计算 → GMEM ----
    blkC = cute.local_tile(gC, (ELEMS_PER_BLOCK_1,), (bidx,))
    tv_layout = cute.make_layout((BLOCK_SIZE, VEC), stride=(VEC, 1))
    block_shape = (ELEMS_PER_BLOCK_1,)

    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype,
        num_bits_per_copy=dtype.width * VEC,
    )
    simt_tiled = cute.make_tiled_copy(simt_atom, tv_layout, block_shape)
    thr_simt = simt_tiled.get_slice(tidx)

    tsA = thr_simt.partition_S(sA)
    tsB = thr_simt.partition_S(sB)
    rA = cute.make_fragment_like(tsA)
    rB = cute.make_fragment_like(tsB)
    cute.copy(simt_tiled, tsA, rA)
    cute.copy(simt_tiled, tsB, rB)

    rC = cute.make_fragment_like(rA)
    for i in range(cute.size(rC)):
        rC[i] = rA[i] + rB[i]

    tC_dst = thr_simt.partition_D(blkC)
    cute.copy(simt_tiled, rC, tC_dst)


@cute.jit
def add_tma(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
    n = a.shape[0]
    a_dtype = a.element_type
    smem_layout = cute.make_layout(ELEMS_PER_BLOCK_1)

    # ---- Host 侧: 创建 TMA descriptor (必须在 kernel launch 前) ----
    # 返回两个东西:
    #   tma_atom   = 搬运指令, 传给 kernel 用于 cute.copy
    #   tma_tensor = 绑定了 descriptor 的 tensor, 传给 kernel 用于分区 (local_tile / tma_partition)
    #               和原始的 a 不一样! 原始 a 是普通 tensor, tma_tensor 带有 TMA descriptor,
    #               TMA 硬件需要通过 descriptor 来找到数据, 所以 kernel 里分区时必须用 tma_tensor
    tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),              # 怎么搬: GMEM → SMEM
        a,                   # 搬什么: tensor a (TMA 从这里读取地址、shape、stride、dtype)
        smem_layout,         # 搬到 SMEM 后怎么排
        (ELEMS_PER_BLOCK_1,) # 每次搬多大一块 tile (2048 个元素)
    )
    tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),             
        b,                  
        smem_layout,         
        (ELEMS_PER_BLOCK_1,) 
    )

    # ---- SharedStorage: 编译期定义 SMEM 内存布局 ----
    # 等价于 CUDA 里的:
    #   struct SharedStorage {
    #       int64_t mbar[2];                         // mbarrier, 16 bytes
    #       __align__(128) __half  sA[2048];         // 数据 buffer A
    #       __align__(128) __half  sB[2048];         // 数据 buffer B
    #   };
    #
    # 为什么不用 smem.allocate_tensor 分别分配 (像方法 2 那样)?
    #   因为 TMA 多了个 mbarrier — 它是 Int64 类型的硬件同步对象,
    #   allocate_tensor 没法分配它, 只能用 struct 把 mbarrier 和数据 buffer 打包一起分配
    @cute.struct
    class SharedStorage:
        # mbarrier 在 sm90+ 上是 128-bit 的硬件对象，因此我们使用2个Int64来表示
        mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
        # TMA 搬数据到 SMEM 时, 目标地址必须 128 字节对齐 (TMA 按 128B cacheline 粒度操作 SMEM)
        sA: cute.struct.Align[
            cute.struct.MemRange[a_dtype, cute.cosize(smem_layout)], 128,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[a_dtype, cute.cosize(smem_layout)], 128,
        ]

    grid = (n // ELEMS_PER_BLOCK_1, 1, 1)
    add_tma_kernel(
        tma_atom_a, tma_tensor_a,
        tma_atom_b, tma_tensor_b,
        c, smem_layout, SharedStorage,
    ).launch(grid=grid, block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 主函数
# =============================================================================
def main():
    print("=" * 70)
    print("GPU 数据搬运硬件路径测试 (RTX 5050 / SM120)")
    print("=" * 70)

    cap = torch.cuda.get_device_capability(0)
    sm = cap[0] * 10 + cap[1]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM:  sm_{sm}0")
    print(f"N:   {N} elements (FP16)")
    print()

    A = torch.randn(N, device="cuda", dtype=torch.float16)
    B = torch.randn(N, device="cuda", dtype=torch.float16)
    ref = A + B
    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)

    results = []

    # ===== 测试 1: Load/Store (LSU 同步) =====
    print("-" * 70)
    print("方法 1: Load/Store — CopyUniversalOp")
    print("  硬件: LSU 管道 (同步)")
    print("  SASS: LDG.E.128 / STG.E.128")
    print("  路径: GMEM → Reg → 计算 → Reg → GMEM")

    C1 = torch.empty(N, device="cuda", dtype=torch.float16)
    C1_ = from_dlpack(C1, assumed_align=16)
    compiled1 = cute.compile(add_load_store, A_, B_, C1_)
    compiled1(A_, B_, C1_)
    ok1 = torch.allclose(C1, ref, atol=1e-2, rtol=1e-2)
    t1 = benchmark(compiled1, kernel_arguments=JitArguments(A_, B_, C1_))
    bw1 = 3 * N * 2 / (t1 * 1e-6) / 1e9
    print(f"  正确性: {'PASS' if ok1 else 'FAIL'}")
    print(f"  耗时:   {t1:.2f} us | 带宽: {bw1:.1f} GB/s")
    results.append(("Load/Store (LDG/STG)", t1, bw1, ok1))
    print()

    # ===== 测试 2: cp.async (LSU 异步) =====
    print("-" * 70)
    print("方法 2: cp.async — CopyG2SOp")
    print("  硬件: LSU 管道 (异步队列)")
    print("  SASS: LDGSTS")
    print("  路径: GMEM →(绕过 Reg)→ SMEM → Reg → 计算 → GMEM")

    C2 = torch.empty(N, device="cuda", dtype=torch.float16)
    C2_ = from_dlpack(C2, assumed_align=16)
    compiled2 = cute.compile(add_cpasync, A_, B_, C2_)
    compiled2(A_, B_, C2_)
    ok2 = torch.allclose(C2, ref, atol=1e-2, rtol=1e-2)
    t2 = benchmark(compiled2, kernel_arguments=JitArguments(A_, B_, C2_))
    bw2 = 3 * N * 2 / (t2 * 1e-6) / 1e9
    print(f"  正确性: {'PASS' if ok2 else 'FAIL'}")
    print(f"  耗时:   {t2:.2f} us | 带宽: {bw2:.1f} GB/s")
    results.append(("cp.async (LDGSTS)", t2, bw2, ok2))
    print()

    # ===== 测试 3: ldmatrix (LSU 矩阵搬运) =====
    print("-" * 70)
    print("方法 3: ldmatrix — LdMatrix8x8x16bOp")
    print("  硬件: LSU 管道 (warp 协作)")
    print("  SASS: LDSM (ldmatrix.sync.aligned.m8n8.x4)")
    print("  路径: GMEM →(cp.async)→ SMEM →(ldmatrix)→ Reg → 计算 → GMEM")
    print(f"  tile: {M_TILE}x{K_TILE}, threads: {THREADS_LDMATRIX} (4 warps)")

    M_total = N // K_TILE
    A2d = A.view(M_total, K_TILE).contiguous()
    B2d = B.view(M_total, K_TILE).contiguous()
    ref2d = A2d + B2d
    C3 = torch.empty(M_total, K_TILE, device="cuda", dtype=torch.float16)

    A2d_ = from_dlpack(A2d, assumed_align=16)
    B2d_ = from_dlpack(B2d, assumed_align=16)
    C3_ = from_dlpack(C3, assumed_align=16)

    compiled3 = cute.compile(add_ldmatrix, A2d_, B2d_, C3_)
    compiled3(A2d_, B2d_, C3_)
    ok3 = torch.allclose(C3, ref2d, atol=1e-2, rtol=1e-2)
    t3 = benchmark(compiled3, kernel_arguments=JitArguments(A2d_, B2d_, C3_))
    bw3 = 3 * N * 2 / (t3 * 1e-6) / 1e9
    print(f"  正确性: {'PASS' if ok3 else 'FAIL'}")
    print(f"  耗时:   {t3:.2f} us | 带宽: {bw3:.1f} GB/s")
    results.append(("ldmatrix (LDSM)", t3, bw3, ok3))
    print()

    # ===== 测试 4: TMA (独立硬件) =====
    print("-" * 70)
    print("方法 4: TMA — CopyBulkTensorTileG2SOp")
    print("  硬件: TMA 独立硬件单元 (不占 LSU)")
    print("  SASS: UTMALDG")
    print("  路径: GMEM →(TMA 硬件)→ SMEM → Reg → 计算 → GMEM")

    C4 = torch.empty(N, device="cuda", dtype=torch.float16)
    C4_ = from_dlpack(C4, assumed_align=16)

    compiled4 = cute.compile(add_tma, A_, B_, C4_)
    compiled4(A_, B_, C4_)
    ok4 = torch.allclose(C4, ref, atol=1e-2, rtol=1e-2)
    t4 = benchmark(compiled4, kernel_arguments=JitArguments(A_, B_, C4_))
    bw4 = 3 * N * 2 / (t4 * 1e-6) / 1e9
    print(f"  正确性: {'PASS' if ok4 else 'FAIL'}")
    print(f"  耗时:   {t4:.2f} us | 带宽: {bw4:.1f} GB/s")
    results.append(("TMA (UTMALDG)", t4, bw4, ok4))
    print()

    # ===== 汇总 =====
    print("=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"  {'方法':<30} {'硬件':<12} {'耗时(us)':>10} {'带宽(GB/s)':>12} {'正确':>6}")
    print(f"  {'-'*30} {'-'*12} {'-'*10} {'-'*12} {'-'*6}")
    hw_names = ["LSU", "LSU", "LSU", "TMA"]
    for (name, t, bw, ok), hw in zip(results, hw_names):
        print(f"  {name:<30} {hw:<12} {t:>10.2f} {bw:>12.1f} {'PASS' if ok else 'FAIL':>6}")
    print("=" * 70)
    print()
    print("四种方式的硬件资源对比:")
    print("  ┌────────────────┬──────────┬───────────────────────────────────┐")
    print("  │ 方法           │ 硬件单元 │ 特点                              │")
    print("  ├────────────────┼──────────┼───────────────────────────────────┤")
    print("  │ Load/Store     │ LSU ×32  │ 每线程独立, 同步, 经过寄存器      │")
    print("  │ cp.async       │ LSU ×32  │ 每线程独立, 异步, 绕过寄存器      │")
    print("  │ ldmatrix       │ LSU ×32  │ Warp 协作, 自动重排 TC layout     │")
    print("  │ TMA            │ TMA ×1   │ 1 线程发射, 硬件自动搬整块 tile   │")
    print("  └────────────────┴──────────┴───────────────────────────────────┘")


if __name__ == "__main__":
    main()
