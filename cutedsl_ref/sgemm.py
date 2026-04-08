"""
=============================================================================
CuTeDSL 复现 origin_cuda_kernel/sgemm/sgemm.cu  + 高级优化版本
=============================================================================

原生 CUDA 有 4 个版本：
  1. sgemm_naive                        — 每个 thread 算 C 中 1 个元素
  2. sgemm_sliced_k_f32_kernel          — 分块 + shared memory，每 thread 算 TM×TN 个元素
  3. sgemm_sliced_k_f32x4_padding_kernel — 上面 + float4 加载 + padding
  4. sgemm_sliced_k_f32x4_padding_reg_kernel — 上面 + 寄存器缓存累加器

当前保留的 CuTeDSL kernel：
  - cuda_core_scalar_gemm: CuTe local_tile + MmaUniversalOp 标量 FMA
  - GemmWmmaVectorizedCopy: Tensor Core + LdMatrix + 向量化 GMEM→SMEM
  - GemmWmmaTmaSingleStage: SM90 TMA 硬件搬运 + mbarrier
  - GemmWmmaCpAsyncMultistage: 手写 cp.async multi-stage 软件流水线
  - GemmWmmaPipelineCpAsync: PipelineCpAsync 封装的同构 producer/consumer 流水线
  - GemmWmmaTmaWarpSpecializedPipeline: PipelineTmaAsync + TMA + warp specialization

语义：
  - cuda_core_scalar_gemm: C[M,N] = A[M,K] × B[N,K]^T，fp32 输出
  - 其他 WMMA kernel: C[M,N] = A[M,K] × B[N,K]^T，fp16 输入、fp32 累加、fp16 输出

性能演进路线:
  cuda_core_scalar_gemm
    → GemmWmmaVectorizedCopy: Tensor Core + 向量化 G2S
    → GemmWmmaTmaSingleStage: TMA 硬件搬运 + mbarrier
    → GemmWmmaCpAsyncMultistage: cp.async 手写多 stage 流水线
    → GemmWmmaPipelineCpAsync: CuTe pipeline API 封装 cp.async producer/consumer
    → GemmWmmaTmaWarpSpecializedPipeline: TMA 硬件 producer + MMA consumer warp specialization

LeetCUDA 优化技术全面映射:
  ┌─────────────────────────────────────────────────────────────────────┐
  │ LeetCUDA SGEMM 优化路线:                                             │
  │   naive → sliced_k → t8x8+f32x4 → bcf(bank conflict free)         │
  │   → dbuf(double buffer) → async(cp.async) → wmma_tf32(Tensor Core) │
  │                                                                     │
  │ LeetCUDA HGEMM 额外优化 (本文件部分流水线示例参考):                  │
  │   - SMEM Swizzle (XOR-based, bank conflicts: 24576→0, +67% perf)   │
  │     ✅ hgemm.py 的 swizzle 示例: make_swizzle + allocate_tensor    │
  │   - Warp Specialization (producer/consumer 分离式流水线)              │
  │     来自 kernels/ws-hgemm/naive_ws_hgemm_sm8x.cu                   │
  │   - Multi-Stage Pipeline (2/3/4/5 stage, cp.async + fence/wait)    │
  │   - Block Swizzle (blockIdx.z 重映射, 改善 L2 Cache 局部性)          │
  │     ✅ hgemm.py 的 block-swizzle 示例: bidz*gdimx+bidx 重映射      │
  │   - K 维内层展开 (WARP_TILE_K=2, 减少同步开销)                       │
  │   - 寄存器双缓冲 (RA[2][M][4], RB[2][N][2], ping-pong 切换)        │
  │   - Epilogue via SMEM (R2S→S2G, 复用 sA 空间存 sC)                 │
  │   - CuTe Swizzle<3,3,3> 自动消除 bank conflict                     │
  │   - 128x256 大 CTA tile + 128-bit 向量化加载                        │
  │                                                                     │
  │ 注: HGEMM/ffpa-attn 子模块已拉取，完整分析见 hgemm.py/flash_attn.py │
  └─────────────────────────────────────────────────────────────────────┘

GPU Kernel 分层任务划分 (适用于所有 kernel, 不只是 GEMM):

  设计 kernel 时从粗到细三层思考:

  1. Grid → Block: 每个 Block 负责什么？
     - GEMM 中: 每个 block 计算输出矩阵的一个 Tile (BM×BN)
     - 代码: local_tile(mA, tiler, (bidx, bidy, None))  用 block_idx 选 Tile

  2. Block → Warp: Block 内的 Warps 怎么分工？
     - 同构模式: 多数 warp 做一样的事, 通过 sync_threads 同步
     - 异构模式: 不同 warp 不同角色, 通过 pipeline mbarrier 协调
       ┌─────────────────────────────────────────────────┐
       │  Warp 0~3: MMA consumer (全职计算, 不搬数据)     │
       │  Warp 4:   TMA producer (全职搬运, 不参与计算)   │
       └─────────────────────────────────────────────────┘
     - 代码: if is_tma_warp / if is_mma_warp  按 warp_idx 分流

  3. Warp → Thread: Warp 内 32 个线程怎么协作？
     - tiled_mma.get_slice(tid):    partition 把 MMA 的子矩阵分给每个线程
     - thr_s2r.partition_S / retile: 每个线程负责搬 SMEM 的一小片到寄存器
     - 每个线程持有 C 的一小片累加器, 最终各自写回 GMEM

  整体演进本质上是这三层分工不断精细化:
    cuda_core_scalar_gemm: Block 级分块 + CUDA Core 标量 FMA
    GemmWmmaVectorizedCopy: Thread 级向量化 (tiled_copy_tv 精确控制每个线程搬几个元素)
    GemmWmmaTmaWarpSpecializedPipeline: Warp 级异构分工 (warp specialization, 不同角色并行执行)
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import PipelineCpAsync, PipelineTmaAsync, CooperativeGroup, Agent, make_pipeline_state, PipelineUserType
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
import torch


# =============================================================================
# cuda_core_scalar_gemm: CuTE 标量 GEMM — MmaUniversalOp + local_tile + cute.gemm
# =============================================================================
# 使用 CuTE 的分块代数 + 标量 MMA 原子操作
# 语义：C[M,N] = A[M,K] × B[N,K]^T  (B 转置存储)
#
# 关键点：
#   - 用 local_tile 自动切 tile（不用手算指针偏移）
#   - 用 make_tiled_mma 定义计算计划
#   - 用 cute.gemm 一行搞定矩阵乘加
#   - 用 partition_A/B/C 自动分配线程-数据映射
#

SCALAR_TILE_M, SCALAR_TILE_N, SCALAR_TILE_K = 16, 32, 16  # CTA tile 大小

@cute.kernel
def cuda_core_scalar_gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    bidx, bidy, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    tiler = (SCALAR_TILE_M, SCALAR_TILE_N, SCALAR_TILE_K)

    # 注意：
    # coord 中的 None = Python 的 ":"，保留该维度产生额外迭代维          
    # proj  中的 None = "跳过"，该维度与当前张量无关  

    coord = (bidx, bidy, None)

    # CTA 级分块：
    #   gA_tile: (BM, BK, K//BK)  — A 的 M 和 K 维度
    #   gB_tile: (BN, BK, K//BK)  — B 的 N 和 K 维度
    #   gC_tile: (BM, BN)         — C 的 M 和 N 维度
    gA_tile = cute.local_tile(gA, tiler, coord, proj=(1, None, 1))
    gB_tile = cute.local_tile(gB, tiler, coord, proj=(None, 1, 1))
    gC_tile = cute.local_tile(gC, tiler, coord, proj=(1, 1, None))

    # ┌─────────────────────────────────────────────────────────────────────────┐
    # │  CuTeDSL MMA Atom 类型对比                                              │
    # ├──────────────────────────┬───────────────┬──────────────┬───────────────┤
    # │ Atom                     │ 硬件          │ 每 atom 粒度 │ PTX 指令      │
    # ├──────────────────────────┼───────────────┼──────────────┼───────────────┤
    # │ MmaUniversalOp(Float32)  │ CUDA Core     │ 1×1 FMA      │ fma.rn.f32   │
    # │ MmaF16BF16Op(16,8,16)   │ Tensor Core   │ 16×8×16      │ mma.sync     │
    # │ warpgroup.MmaF16BF16Op  │ Tensor Core   │ 64×N×16      │ wgmma.async  │
    # │ tcgen05.MmaF16BF16Op    │ Tensor Core   │ 128×256×16   │ tcgen05.mma  │
    # └──────────────────────────┴───────────────┴──────────────┴───────────────┘
    #
    # atom_layout 的维度含义取决于 atom 类型：
    #   MmaUniversalOp:  1 个 atom = 1 个 thread → atom_layout 是 thread 级排列
    #                    atoms_layout=(16,16,1) → 256 个线程排成 16×16 网格
    #   MmaF16BF16Op:    1 个 atom = 1 个 warp (32 threads) → atom_layout 是 warp 级排列
    #                    atom_layout_mnk=(2,2,1) → 4 个 warp 排成 2×2 = 128 threads
    #   warpgroup.Mma:   1 个 atom = 1 个 warp group (128 threads) → atom_layout 是 warp group 级
    #                    atom_layout_mnk=(2,1,1) → 2 个 warp group = 256 threads
    #   tcgen05.Mma:     1 个 atom = 1 个 thread (单线程发射) → 不需要 atom_layout
    #                    直接 cute.make_tiled_mma(op)，硬件自动覆盖 128×256×16
    #
    # MmaUniversalOp: 标量 FMA，每线程做 1×1 的乘加（CUDA Core，非 Tensor Core）
    # atoms_layout = (16, 16, 1): 256 线程排成 16×16 网格（thread 维度！）
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)

    # 线程分区
    thr_mma = tiled_mma.get_slice(tidx)
    tCgC = thr_mma.partition_C(gC_tile)
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0)

    # K 方向循环
    K_tiles = gA_tile.shape[2]
    for k in range(K_tiles):
        gA_k = gA_tile[None, None, k]
        gB_k = gB_tile[None, None, k]

        tCgA = thr_mma.partition_A(gA_k)
        tCgB = thr_mma.partition_B(gB_k)

        tCrA = tiled_mma.make_fragment_A(tCgA)
        tCrB = tiled_mma.make_fragment_B(tCgB)

        # GMEM → Register: 直接 load/store，生成普通 LDG 指令
        # 当然也可以写成 cute.copy 形式，不过生成的 GPU 代码完全一样，也是普通的LDG：
        #   atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)
        #   cute.copy(atom, tCgA, tCrA)
        #   cute.copy(atom, tCgB, tCrB)
        tCrA.store(tCgA.load())
        tCrB.store(tCgB.load())

        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    tCgC.store(tCrC.load())


@cute.jit
def cuda_core_scalar_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    grid_m = (mA.shape[0] + SCALAR_TILE_M - 1) // SCALAR_TILE_M
    grid_n = (mB.shape[0] + SCALAR_TILE_N - 1) // SCALAR_TILE_N
    cuda_core_scalar_gemm_kernel(mA, mB, mC).launch(
        grid=(grid_m, grid_n, 1), block=(256, 1, 1))


# =============================================================================
# GemmWmmaVectorizedCopy: Tensor Core + LdMatrix + 向量化 GMEM→SMEM
# =============================================================================
# 使用 Ampere SM80+ 的 Tensor Core (WMMA mma.sync.aligned.m16n8k16)
# 语义：C[M,N] = A[M,K] × B[N,K]^T  (fp16 输入, fp32 累加, fp16 输出)
#
# 数据流：GMEM → (向量化 copy) → SMEM → (LdMatrix) → RMEM → (WMMA) → RMEM → GMEM
#
# 关键组件：
#   MmaF16BF16Op: 硬件 mma.sync.aligned.m16n8k16 指令
#   make_tiled_mma: 4 个 Warp (2×2) 组成 TiledMMA，覆盖 32×32 的 C 子块
#   LdMatrix8x8x16bOp: 从 SMEM 加载到寄存器，自动满足 Tensor Core 布局要求
#   make_tiled_copy_tv: 线程级向量化 G2S，每线程一次搬 4 个 fp16 = 64 bits
#   allocate_tensor(..., 16, None): SMEM 起始地址 16 字节对齐，不启用 swizzle
#
# 手动逐元素 GMEM→SMEM 的 WMMA 基线已删掉；学习和实战都直接从这个版本开始。

class GemmWmmaVectorizedCopy:
    def __init__(self, cta_tiler=(128, 128, 64)):
        self._bM, self._bN, self._bK = cta_tiler
        self._cta_tiler = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self._num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self._smem_padding = 8
        self._num_vectorized = 4

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        # MMA
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)

        # SMEM layout:
        # shape 仍是有效数据区 (BM, BK)，但 stride 把物理行跨度扩成 BK + padding。
        # 概念上等价于 CUDA 里的 __shared__ half sA[BM][BK + padding]。
        padding = self._smem_padding
        sA_layout = cute.make_layout((self._bM, self._bK), stride=(self._bK + padding, 1))
        sB_layout = cute.make_layout((self._bN, self._bK), stride=(self._bK + padding, 1))

        # 向量化拷贝配置
        num_vec = self._num_vectorized
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vec)
        major_mode_size = self._bK // num_vec
        tA = cute.make_layout(
            shape=(self._num_threads // major_mode_size, major_mode_size),
            stride=(major_mode_size, 1))
        vA = cute.make_layout(shape=(1, num_vec), stride=(0, 1))
        tiled_copy_A = cute.make_tiled_copy_tv(atom_copy, tA, vA)
        tiled_copy_B = cute.make_tiled_copy_tv(atom_copy, tA, vA)

        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        self.kernel(
            mA, mB, mC, sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B, tiled_mma
        ).launch(grid=grid_dim, block=(self._num_threads, 1, 1))

    @cute.kernel
    def kernel(self, mA, mB, mC, sA_layout, sB_layout,
               tiled_copy_A, tiled_copy_B, tiled_mma):
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        allocator = cutlass.utils.SmemAllocator()
        sA = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
        sB = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)

        gA = cute.local_tile(mA, self._cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self._cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self._cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

        # 向量化拷贝分区
        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)
        tAgA = thr_copyA.partition_S(gA)
        tAsA = thr_copyA.partition_D(sA)
        tBgB = thr_copyB.partition_S(gB)
        tBsB = thr_copyB.partition_D(sB)

        # MMA 分区
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        # LdMatrix
        atom_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mA.element_type)
        atom_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mB.element_type)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_s2r_B.get_slice(tid)
        tCsA_view = thr_s2r_A.partition_S(sA)
        tCrA_view = thr_s2r_A.retile(tCrA)
        tCsB_view = thr_s2r_B.partition_S(sB)
        tCrB_view = thr_s2r_B.retile(tCrB)

        # 主循环
        tCrC.fill(0.0)
        for kidx in range(mA.shape[1] // self._bK):
            # GMEM → SMEM (向量化 64-bit load)
            cute.copy(tiled_copy_A, tAgA[None, None, None, kidx], tAsA[None, None, None])
            cute.copy(tiled_copy_B, tBgB[None, None, None, kidx], tBsB[None, None, None])
            cute.arch.sync_threads()

            # SMEM → Register (LdMatrix)
            cute.copy(tiled_s2r_A, tCsA_view, tCrA_view)
            cute.copy(tiled_s2r_B, tCsB_view, tCrB_view)

            # WMMA
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

        # 写回
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for i in range(cute.size(tCrC_out)):
            tCrC_out[i] = cutlass.Float16(tCrC[i])
        cute.copy(atom_store, tCrC_out, tCgC)


# =============================================================================
# GemmWmmaTmaSingleStage: Tensor Core + TMA 硬件搬运 + mbarrier 同步
# =============================================================================
# TMA (Tensor Memory Accelerator) 是 Hopper 引入的硬件单元
# GMEM → SMEM 的搬运完全由硬件完成，不占用线程资源
# 用 mbarrier 替代 sync_threads 进行异步同步
#
# 数据流：GMEM →(TMA 硬件)→ SMEM →(LdMatrix)→ RMEM →(WMMA)→ RMEM → GMEM
#
# 对比 GemmWmmaVectorizedCopy：
#   向量化版本用线程做 G2S 拷贝（线程被占用）
#   TMA 版本用硬件搬运（线程空闲，可以做其他事）

class GemmWmmaTmaSingleStage:
    def __init__(self, cta_tiler=(128, 128, 64)):
        self.tile_shape_mnk = cta_tiler
        self._bM, self._bN, self._bK = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self.warp_size = cute.arch.WARP_SIZE
        self.threads_per_cta = self.warp_size * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self.num_stages = 1
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        # Swizzled SMEM layout (自动消除 bank conflict)
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype, num_stages=self.num_stages)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype, num_stages=self.num_stages)

        # TMA 描述符
        tma_atom_a, tma_tensor_a = self._make_tma(
            a, self.a_smem_layout_staged, (self._bM, self._bK))
        tma_atom_b, tma_tensor_b = self._make_tma(
            b, self.b_smem_layout_staged, (self._bN, self._bK))

        # WMMA TiledMMA
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)

        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage

        grid_dim = *cute.ceil_div(c.shape, (self._bM, self._bN)), 1
        self.kernel(
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tiled_mma, c,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
        ).launch(grid=grid_dim, block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(self, tma_atom_a, mA_mk, tma_atom_b, mB_nk,
               tiled_mma, mC, a_smem_layout_staged, b_smem_layout_staged):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        gA = cute.local_tile(mA_mk, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB_nk, self.tile_shape_mnk, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, 1, None))

        # TMA 分区
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

        # MMA 分区
        sA_mma = cute.slice_(sA, (None, None, 0))
        sB_mma = cute.slice_(sB, (None, None, 0))
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA_mma)
        tCsB = thr_mma.partition_B(sB_mma)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        # LdMatrix
        atom_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        atom_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_s2r_B.get_slice(tid)
        tCsA_copy_view = thr_s2r_A.partition_S(sA_mma)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCsB_copy_view = thr_s2r_B.partition_S(sB_mma)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

        # mbarrier 初始化
        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        mbar_ptr = storage.mbar_ptr.data_ptr()

        if warp_idx == 0 and tid == 0:
            cute.arch.mbarrier_init(mbar_ptr, cnt=1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # ====== 主循环 ======
        tCrC.fill(0.0)
        phase = 0
        for kidx in range(mA_mk.shape[1] // self._bK):
            # TMA: GMEM → SMEM (硬件搬运，不占线程)
            if warp_idx == 0:
                cute.copy(tma_atom_a, tAgA[None, kidx], tAsA[None, 0], tma_bar_ptr=mbar_ptr)
                cute.copy(tma_atom_b, tBgB[None, kidx], tBsB[None, 0], tma_bar_ptr=mbar_ptr)
                if tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_transaction_bytes)

            # 等待 TMA 完成
            cute.arch.mbarrier_wait(mbar_ptr, phase)
            phase ^= 1

            # SMEM → Register (LdMatrix) + WMMA
            cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
            cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

        # 写回
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for i in range(cute.size(tCrC_out)):
            tCrC_out[i] = cutlass.Float16(tCrC[i])
        cute.copy(atom_store, tCrC_out, tCgC)

    @staticmethod
    def _make_tma(tensor, smem_layout_staged, smem_tile):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile)



# =============================================================================
# GemmWmmaCpAsyncMultistage: 手写 cp.async multi-stage 软件流水线
# =============================================================================
# 这是我们之前讨论的经典软件流水线的完整实现：
#
# 核心要素：
#   1. cp.async (CopyG2SOp): GMEM→SMEM 异步拷贝，绕过寄存器
#   2. cp_async_commit_group: 给已提交的异步拷贝打一个"事务点"
#   3. cp_async_wait_group(N): 等待直到最多还剩 N 个未完成的事务组
#   4. num_stages 个 SMEM buffer 轮转使用
#
# 流水线结构 (Prologue-MainLoop-Epilogue 三段式):
#   Prologue: 发射 stage-1 个异步 G→S 拷贝，灌满管线
#   MainLoop: 每次迭代 "发射1个新G→S + wait + S→R + MMA"，三层重叠
#   Epilogue: 只做 S→R + MMA，排空管线
#
# 对比 GemmWmmaVectorizedCopy:
#   向量化版本: 同步拷贝，load 完再算
#   cp.async 版本: 异步拷贝 + N 个 buffer 轮转，隐藏 G2S 延迟
#
# 对应 LeetCUDA: hgemm_wmma_stage.cu / sgemm_wmma_tf32_stage_kernel

class GemmWmmaCpAsyncMultistage:
    """
    手动 cp.async multi-stage 流水线 GEMM

    流水线时间线 (num_stages=3):
      t0   t1   t2   t3   t4   t5   ...
      G→S0 G→S1      G→S2      G→S0 ...    ← 异步拷贝 (cp.async)
                wait  wait  wait             ← 同步点
                S→R   S→R   S→R   S→R  ...   ← SMEM→寄存器 (LdMatrix)
                MMA   MMA   MMA   MMA  ...   ← Tensor Core 计算
      |Prologue|--------MainLoop--------|
    """
    def __init__(self, cta_tiler=(128, 128, 32), num_stages=3):
        self._bM, self._bN, self._bK = cta_tiler
        self._cta_tiler = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self._num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self._smem_padding = 8
        self._num_vectorized = 4
        self._num_stages = num_stages

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)

        # staged SMEM layout: (num_stages, BM, BK) — 第一维是 stage 索引
        padding = self._smem_padding
        sA_layout = cute.make_layout(
            shape=(self._num_stages, self._bM, self._bK),
            stride=(self._bM * (self._bK + padding), self._bK + padding, 1))
        sB_layout = cute.make_layout(
            shape=(self._num_stages, self._bN, self._bK),
            stride=(self._bN * (self._bK + padding), self._bK + padding, 1))

        # cp.async 异步拷贝 atom (CopyG2SOp = LDGSTS 指令)
        num_vec = self._num_vectorized
        cpasync_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vec)
        major_mode_size = self._bK // num_vec
        tA = cute.make_layout(
            shape=(self._num_threads // major_mode_size, major_mode_size),
            stride=(major_mode_size, 1))
        vA = cute.make_layout(shape=(1, num_vec), stride=(0, 1))
        tiled_cpasync_A = cute.make_tiled_copy_tv(cpasync_atom, tA, vA)
        tiled_cpasync_B = cute.make_tiled_copy_tv(cpasync_atom, tA, vA)

        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        self.kernel(
            mA, mB, mC, sA_layout, sB_layout,
            tiled_cpasync_A, tiled_cpasync_B, tiled_mma
        ).launch(grid=grid_dim, block=(self._num_threads, 1, 1))

    @cute.kernel
    def kernel(self, mA, mB, mC, sA_layout, sB_layout,
               tiled_cpasync_A, tiled_cpasync_B, tiled_mma):
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        # ====== 分配 staged SMEM (num_stages, BM, BK) ======
        allocator = cutlass.utils.SmemAllocator()
        sA = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
        sB = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)

        # CTA 级分块
        gA = cute.local_tile(mA, self._cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self._cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self._cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

        # cp.async G→S 分区 (GMEM side 只需一份)
        thr_cpA = tiled_cpasync_A.get_slice(tid)
        thr_cpB = tiled_cpasync_B.get_slice(tid)
        tAgA = thr_cpA.partition_S(gA)
        tBgB = thr_cpB.partition_S(gB)

        # ====== MMA 相关准备 (循环外做一次, 用 stage 0 做模板) ======
        thr_mma = tiled_mma.get_slice(tid)
        tCgC = thr_mma.partition_C(gC)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        # 用 stage 0 做模板创建 fragment (所有 stage layout 相同, 只是 base ptr 不同)
        sA_0 = cute.slice_(sA, (0, None, None))
        sB_0 = cute.slice_(sB, (0, None, None))
        tCsA_0 = thr_mma.partition_A(sA_0)
        tCsB_0 = thr_mma.partition_B(sB_0)
        tCrA = tiled_mma.make_fragment_A(tCsA_0)
        tCrB = tiled_mma.make_fragment_B(tCsB_0)

        # LdMatrix S→R copy (循环外做一次)
        atom_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mA.element_type)
        atom_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mB.element_type)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_s2r_B.get_slice(tid)

        # retile 只做一次 (register side 的布局和 stage 无关)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

        tCrC.fill(0.0)
        num_stages = self._num_stages
        num_k_tiles = mA.shape[1] // self._bK

        # ====== Prologue: 发射 stage-1 个异步 G→S ======
        for s in range(self._num_stages - 1):
            sA_s = cute.slice_(sA, (s, None, None))
            sB_s = cute.slice_(sB, (s, None, None))
            tAsA_s = thr_cpA.partition_D(sA_s)
            tBsB_s = thr_cpB.partition_D(sB_s)
            cute.copy(tiled_cpasync_A, tAgA[None, None, None, s], tAsA_s[None, None, None])
            cute.copy(tiled_cpasync_B, tBgB[None, None, None, s], tBsB_s[None, None, None])
            cute.arch.cp_async_commit_group()

        # wait S[0]: 确保第一个 buffer 就绪
        cute.arch.cp_async_wait_group(self._num_stages - 2)
        cute.arch.sync_threads()

        # ====== MainLoop ======
        for kidx in range(num_k_tiles):
            stage = kidx % num_stages

            # ① 发射下一个 Tile 的异步拷贝 (和当前计算重叠)
            next_load = kidx + num_stages - 1
            if next_load < num_k_tiles:
                next_stage = next_load % num_stages
                sA_next = cute.slice_(sA, (next_stage, None, None))
                sB_next = cute.slice_(sB, (next_stage, None, None))
                tAsA_next = thr_cpA.partition_D(sA_next)
                tBsB_next = thr_cpB.partition_D(sB_next)
                cute.copy(tiled_cpasync_A, tAgA[None, None, None, next_load], tAsA_next[None, None, None])
                cute.copy(tiled_cpasync_B, tBgB[None, None, None, next_load], tBsB_next[None, None, None])
                cute.arch.cp_async_commit_group()

            # ② wait
            cute.arch.cp_async_wait_group(self._num_stages - 2)
            cute.arch.sync_threads()

            # ③ SMEM → Register (LdMatrix) + MMA
            # 循环内只做 cute.slice_ + partition_S (SMEM source 分区)
            # fragment 和 retile 复用循环外的结果
            sA_cur = cute.slice_(sA, (stage, None, None))
            sB_cur = cute.slice_(sB, (stage, None, None))

            tCsA_copy_view = thr_s2r_A.partition_S(sA_cur)
            tCsB_copy_view = thr_s2r_B.partition_S(sB_cur)

            cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
            cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

            cute.arch.sync_threads()

        # ====== 写回 GMEM ======
        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for i in range(cute.size(tCrC_out)):
            tCrC_out[i] = cutlass.Float16(tCrC[i])
        cute.copy(atom_store, tCrC_out, tCgC)


# =============================================================================
# GemmWmmaPipelineCpAsync: PipelineCpAsync 封装流水线 (同构 Producer-Consumer)
# =============================================================================
# 使用 CuTeDSL 封装的 PipelineCpAsync API，实现生产者-消费者流水线
#
# 对比 GemmWmmaCpAsyncMultistage:
#   手写 cp.async: 自己管理 commit/wait/buffer轮转，灵活但容易出错
#   PipelineCpAsync: 封装 mbarrier 同步逻辑，代码更干净
#        producer.acquire → 等 buffer 可写
#        handle.commit    → 通知 consumer 数据已就绪 (内部做 cp_async_mbarrier_arrive)
#        consumer.wait    → 等 producer 填好数据
#        handle.release   → 通知 producer buffer 可复用
#
# --- 同构流水线 vs 异构流水线 ---
#   本文把"同构/异构"按执行路径来区分, 是不是 producer 和 consumer 用同一套硬件路径:
#
#   ┌─────────────────────────────────────────────────────────────┐
#   │ 同构 = Producer 和 Consumer 都跑在 SM 线程上              │
#   │   - Producer (warp 0, 32 线程): SM 线程标量发射 cp.async    │
#   │   - Consumer (warp 1~4, 128 线程): SM 线程 LdMatrix + WMMA  │
#   │   - 虽然 warp 分角色, 但都是"线程发指令"这一条路径          │
#   │   - 同步信号: producer_group.size = 32                      │
#   │     full-mbarrier arrive_count = 32, 靠 cp_async_mbarrier   │
#   │     _arrive_noinc 全员 arrive                               │
#   │   - Producer 占用 SM 的 issue slot, 和 Consumer 抢指令发射  │
#   └─────────────────────────────────────────────────────────────┘
#   ┌─────────────────────────────────────────────────────────────┐
#   │ 异构 = Producer 和 Consumer 走完全不同的硬件路径           │
#   │   - Producer: TMA 硬件 DMA 引擎 (只要 1 线程"发令")          │
#   │   - Consumer: SM 线程 + Tensor Core (WMMA)                  │
#   │   - producer_group.size = 1, mbarrier 靠 TMA 的 tx_count    │
#   │     硬件机制自动 arrive (producer_commit 是 NOP)            │
#   │   - Producer 几乎不占 SM issue slot, 真正时空并行            │
#   └─────────────────────────────────────────────────────────────┘
#
#   注意: "同构" 不代表"所有线程干一样的事"。PipelineCpAsync 仍然做了 warp 分工
#         (producer warp vs consumer warps), 但这些 warp 都在 SM 线程路径上跑,
#         所以归为同构。异构的关键是"走不同硬件单元", 只有 TMA/WGMMA 才算.
#
# 注意: PipelineCpAsync 需要 SM80+ (cp.async), mbarrier 需要 SM80+, 本 demo 按 SM90+ 起跑

class GemmWmmaPipelineCpAsync:
    """
    CuTeDSL PipelineCpAsync 封装的流水线 GEMM (【同构】流水线 + cp.async)

    同构 = Producer 和 Consumer 都是 SM 线程路径 (都靠 SM issue slot 发指令):
      Warp 0   = Producer (32 线程): 标量 cp.async 加载 GMEM → SMEM
                 producer_group = CooperativeGroup(Agent.Thread, 32)
                 mbarrier arrive-count = 32 (所有 producer 线程到齐)
      Warp 1~4 = Consumer (128 线程): SMEM → Register (LdMatrix) + WMMA
                 consumer_group = CooperativeGroup(Agent.Thread, 128)

    对比 TMA warp-specialized pipeline: Producer 是 TMA 硬件引擎 (非线程),
    Consumer 是 SM 线程，两者走不同硬件路径，真正并行不抢 issue slot.
    """
    def __init__(self, cta_tiler=(128, 128, 32), num_stages=4):
        self._bM, self._bN, self._bK = cta_tiler
        self._cta_tiler = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self._mma_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]  # 128
        self._smem_padding = 8
        self._num_stages = num_stages
        self._num_producer_threads = cute.arch.WARP_SIZE  # 1 warp = producer
        self._num_consumer_threads = self._mma_threads    # 128 线程 = 完整 MMA
        self._num_threads = self._num_producer_threads + self._num_consumer_threads  # 160

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)

        # staged SMEM layout
        padding = self._smem_padding
        sA_layout = cute.make_layout(
            shape=(self._num_stages, self._bM, self._bK),
            stride=(self._bM * (self._bK + padding), self._bK + padding, 1))
        sB_layout = cute.make_layout(
            shape=(self._num_stages, self._bN, self._bK),
            stride=(self._bN * (self._bK + padding), self._bK + padding, 1))

        @cute.struct
        class SharedStorage:
            pipeline_mbarrier_ptr: cute.struct.MemRange[
                cutlass.Int64, self._num_stages * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 1024]
            sB: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 1024]
        self.shared_storage = SharedStorage

        # cp.async 向量化拷贝 (producer 32 线程, CopyG2SOp 异步)
        num_vec = 4
        copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(), mA.element_type,
            num_bits_per_copy=mA.element_type.width * num_vec)
        major_mode_size = self._bK // num_vec
        tA_prod = cute.make_layout(
            shape=(self._num_producer_threads // major_mode_size, major_mode_size),
            stride=(major_mode_size, 1))
        vA_prod = cute.make_layout(shape=(1, num_vec), stride=(0, 1))
        tiled_copy_A = cute.make_tiled_copy_tv(copy_atom, tA_prod, vA_prod)
        tiled_copy_B = cute.make_tiled_copy_tv(copy_atom, tA_prod, vA_prod)

        grid_dim = *cute.ceil_div(mC.shape, (self._bM, self._bN)), 1
        self.kernel(mA, mB, mC, sA_layout, sB_layout,
                    tiled_copy_A, tiled_copy_B, tiled_mma
        ).launch(grid=grid_dim, block=(self._num_threads, 1, 1))

    @cute.kernel
    def kernel(self, mA, mB, mC, sA_layout, sB_layout,
               tiled_copy_A, tiled_copy_B, tiled_mma):
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        BM, BN, BK = self._bM, self._bN, self._bK

        # SMEM
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(layout=sA_layout)
        sB = storage.sB.get_tensor(layout=sB_layout)

        # PipelineCpAsync: 专门为 cp.async 设计
        # commit 内部调用 cp_async_mbarrier_arrive_noinc，将 cp.async 完成信号绑定到 mbarrier
        #
        # 【同构流水线】Producer/Consumer 都是 SM 线程路径:
        #   - producer_group.size = 32: full-mbarrier arrive-count = 32,
        #     必须 32 个生产者线程都执行 commit 才算 stage 就绪
        #   - 生产者线程和消费者线程抢 SM issue slot (同构的代价)
        #   对比 TMA 异构流水线:
        #   - producer_group.size = 1, mbarrier 靠 TMA 硬件 tx_count 自动 arrive
        #   - TMA 走独立 DMA 引擎, 不占 SM issue slot
        mainloop_pipeline = PipelineCpAsync.create(
            barrier_storage=storage.pipeline_mbarrier_ptr.data_ptr(),
            num_stages=self._num_stages,
            producer_group=CooperativeGroup(Agent.Thread, self._num_producer_threads),  # 32 SM 线程发 cp.async (同构)
            consumer_group=CooperativeGroup(Agent.Thread, self._num_consumer_threads),  # 128 SM 线程做 MMA
        )
        producer, consumer = mainloop_pipeline.make_participants()

        # CTA 分块
        gA = cute.local_tile(mA, self._cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self._cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self._cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

        num_k_tiles = mA.shape[1] // BK

        # ====== Producer (Warp 0): 向量化同步加载 GMEM → SMEM ======
        if warp_idx == 0:
            thr_cpA = tiled_copy_A.get_slice(tid)
            thr_cpB = tiled_copy_B.get_slice(tid)
            tAgA = thr_cpA.partition_S(gA)
            tBgB = thr_cpB.partition_S(gB)

            for kidx in range(num_k_tiles):
                handle = producer.acquire_and_advance()

                sA_stage = cute.slice_(sA, (handle.index, None, None))
                sB_stage = cute.slice_(sB, (handle.index, None, None))
                tAsA = thr_cpA.partition_D(sA_stage)
                tBsB = thr_cpB.partition_D(sB_stage)
                cute.copy(tiled_copy_A, tAgA[None, None, None, kidx], tAsA[None, None, None])
                cute.copy(tiled_copy_B, tBgB[None, None, None, kidx], tBsB[None, None, None])
                cute.arch.cp_async_commit_group()

                handle.commit()

            producer.tail()

        # ====== Consumer (Warp 1~4, 128 线程): SMEM → R + WMMA ======
        if warp_idx != 0:
            consumer_tid = tid - self._num_producer_threads

            thr_mma = tiled_mma.get_slice(consumer_tid)
            tCgC = thr_mma.partition_C(gC)
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            # 用 stage 0 做模板, 循环外做一次 partition_A/make_fragment/retile
            sA_0 = cute.slice_(sA, (0, None, None))
            sB_0 = cute.slice_(sB, (0, None, None))
            tCsA_0 = thr_mma.partition_A(sA_0)
            tCsB_0 = thr_mma.partition_B(sB_0)
            tCrA = tiled_mma.make_fragment_A(tCsA_0)
            tCrB = tiled_mma.make_fragment_B(tCsB_0)

            atom_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mA.element_type)
            atom_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mB.element_type)
            tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
            thr_s2r_A = tiled_s2r_A.get_slice(consumer_tid)
            thr_s2r_B = tiled_s2r_B.get_slice(consumer_tid)

            # retile 只做一次
            tCrA_copy_view = thr_s2r_A.retile(tCrA)
            tCrB_copy_view = thr_s2r_B.retile(tCrB)

            for kidx in range(num_k_tiles):
                handle = consumer.wait_and_advance()

                # 循环内只做 cute.slice_ + partition_S
                sA_stage = cute.slice_(sA, (handle.index, None, None))
                sB_stage = cute.slice_(sB, (handle.index, None, None))

                tCsA_copy_view = thr_s2r_A.partition_S(sA_stage)
                tCsB_copy_view = thr_s2r_B.partition_S(sB_stage)

                cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
                cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
                cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

                handle.release()

            atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
            tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
            for i in range(cute.size(tCrC_out)):
                tCrC_out[i] = cutlass.Float16(tCrC[i])
            cute.copy(atom_store, tCrC_out, tCgC)


# =============================================================================
# GemmWmmaTmaWarpSpecializedPipeline: PipelineTmaAsync + TMA + Warp Specialization
#                                      (SM90+ 异构流水线)
# =============================================================================
#
# 核心思想: 【异构流水线】Producer 和 Consumer 走完全不同的硬件路径
#   ┌─────────────────────────────────────────────────────────────┐
#   │   Producer: TMA 硬件 DMA 引擎  (非 SM 线程, 只需 1 线程"发令") │
#   │   Consumer: SM 线程 + Tensor Core (LdMatrix + WMMA)           │
#   └─────────────────────────────────────────────────────────────┘
#   ==> 两条路径物理独立, Producer 不占 SM 的 issue slot, 真正时空并行.
#
#   CTA = 160 线程 = 5 warps
#   ┌──────────────────────────────────────────────────────┐
#   │  Warp 0~3 (128线程) │ MMA consumer: LdMatrix + WMMA │  ← 全职计算
#   │  Warp 4   (32 线程) │ TMA producer: 发射TMA拷贝指令 │  ← 全职搬运
#   └──────────────────────────────────────────────────────┘
#   (Warp 4 内部实际只有 1 个线程真正发 TMA; 整个搬运由硬件 DMA 引擎完成)
#
#   Warp 4 (producer)              Warp 0~3 (consumer)
#       │                               │
#       ├─ acquire(等buffer空)           │
#       ├─ TMA copy → S[stage]          │
#       │   tx_count→0 ───────────►     ├─ wait(数据就绪)
#       │                               ├─ LdMatrix + WMMA
#       │  ◄────────────────────────    ├─ release(buffer可复用)
#       ⋮                               ⋮
#
# --- 对比 PipelineCpAsync 同构流水线 ---
#   PipelineCpAsync: Producer/Consumer 都在 SM 线程路径上, 都靠 SM 发指令,
#            producer_group.size = 32, 抢占 SM issue slot.
#   TMA 异构: Producer 走 TMA 硬件引擎, Consumer 走 SM 线程路径,
#            producer_group.size = 1, mbarrier 靠 TMA 硬件 tx_count 自动 arrive,
#            producer_commit 是 NOP.
#
# 为什么比同步/手写 cp.async 路径更进一步:
#   普通 WMMA/cp.async: 线程先发加载，再做计算 → 仍消耗 SM issue slot
#   TMA warp-specialized: TMA 硬件做加载(不占线程) + MMA warps 全职计算
#
#   这是 CUTLASS 3.x 的标准架构: warp-specialized persistent kernel
#
# Pipeline 机制 (PipelineTmaAsync):
#   - producer commit 是 NOP: TMA 硬件自动通过 tx_count 信号完成
#   - consumer wait: mbarrier phase 等待, 零轮询开销
#   - 多 stage SMEM buffer 轮转: 当前 stage 在计算时, 下一个在传输

class GemmWmmaTmaWarpSpecializedPipeline:
    def __init__(self, cta_tiler=(128, 128, 64), num_stages=4):
        self.tile_shape_mnk = cta_tiler
        self._bM, self._bN, self._bK = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self.warp_size = cute.arch.WARP_SIZE

        # Warp specialization: 4 MMA warps + 1 TMA warp
        self.num_mma_warps = self.atom_layout_mnk[0] * self.atom_layout_mnk[1]  # 4
        self.mma_warp_ids = tuple(range(self.num_mma_warps))
        self.tma_warp_id = self.num_mma_warps  # warp 4
        self.threads_per_cta = self.warp_size * (self.num_mma_warps + 1)  # 160

        self.num_stages = num_stages
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        # Swizzled SMEM layouts with staging dimension
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype, num_stages=self.num_stages)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype, num_stages=self.num_stages)

        # TMA descriptors
        tma_atom_a, tma_tensor_a = self._make_tma(
            a, self.a_smem_layout_staged, (self._bM, self._bK))
        tma_atom_b, tma_tensor_b = self._make_tma(
            b, self.b_smem_layout_staged, (self._bN, self._bK))

        # WMMA TiledMMA (same as single-stage TMA path)
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tiled_mma = cute.make_tiled_mma(
            op_or_atom=mma_op,
            atom_layout_mnk=self.atom_layout_mnk,
            permutation_mnk=permutation_mnk)

        # SharedStorage: pipeline mbarriers + staged SMEM buffers
        @cute.struct
        class SharedStorage:
            pipeline_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)],
                self.buffer_align_bytes]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)],
                self.buffer_align_bytes]
        self.shared_storage = SharedStorage

        grid_dim = *cute.ceil_div(c.shape, (self._bM, self._bN)), 1
        self.kernel(
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tiled_mma, c,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
        ).launch(grid=grid_dim, block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(self, tma_atom_a, mA_mk, tma_atom_b, mB_nk,
               tiled_mma, mC, a_smem_layout_staged, b_smem_layout_staged):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        is_mma_warp = warp_idx <= self.mma_warp_ids[-1]
        is_tma_warp = warp_idx == self.tma_warp_id

        if is_tma_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ====== SMEM allocation ======
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)

        # CTA 分块
        gA = cute.local_tile(mA_mk, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB_nk, self.tile_shape_mnk, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, 1, None))

        # ====== TMA partition ======
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

        # ====== MMA partition (循环外, 用 stage 0 做模板) ======
        thr_mma = tiled_mma.get_slice(tid)
        tCgC = thr_mma.partition_C(gC)

        sA_0 = cute.slice_(sA, (None, None, 0))
        sB_0 = cute.slice_(sB, (None, None, 0))
        tCsA_0 = thr_mma.partition_A(sA_0)
        tCsB_0 = thr_mma.partition_B(sB_0)
        tCrA = tiled_mma.make_fragment_A(tCsA_0)
        tCrB = tiled_mma.make_fragment_B(tCsB_0)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        # LdMatrix (循环外)
        atom_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        atom_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_s2r_B.get_slice(tid)
        tCrA_copy_view = thr_s2r_A.retile(tCrA)
        tCrB_copy_view = thr_s2r_B.retile(tCrB)

        # ====== PipelineTmaAsync ======
        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        mbar_ptr = storage.pipeline_mbar_ptr.data_ptr()

        # PipelineTmaAsync: TMA producer + AsyncThread consumer
        #
        # 【异构流水线】Producer (TMA 硬件) 和 Consumer (SM 线程) 走不同硬件路径:
        #   - producer_group.size = 1: 只要 1 个线程"发令", TMA 硬件自己搬
        #     full-mbarrier 不是靠线程 arrive_count 计数, 而是靠 TMA 硬件 tx_count:
        #     TMA 每搬完 tx_count 字节, 硬件自动 arrive 对应 mbarrier.
        #     => producer_commit 是 NOP (TMA 指令本身完成 arrive), 和 cp.async pipeline 的 commit 语义完全不同.
        #   - producer 走 DMA 引擎, 不占 SM issue slot, 真正和 consumer 并行.
        #   对比 PipelineCpAsync (同构 + cp.async):
        #   - PipelineCpAsync producer_group.size = 32 个 SM 线程, 和 consumer 抢 issue slot.
        mainloop_pipeline = PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),                   # 1 线程发 TMA, 硬件异步搬运 (异构)
            consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_warps),  # 4 MMA warps
            barrier_storage=mbar_ptr,
            tx_count=tma_transaction_bytes,                                     # 硬件计数阈值 = 一个 stage 的字节数
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1))
        )

        producer_state = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        consumer_state = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)

        num_k_tiles = mA_mk.shape[1] // self._bK

        # ====== TMA warp (producer): 硬件搬运 GMEM → SMEM ======
        if is_tma_warp:
            # Prefetch: 灌满所有 pipeline stages
            for kidx in range(self.num_stages):
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(tma_atom_a, tAgA[None, producer_state.count],
                          tAsA[None, producer_state.index],
                          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state))
                cute.copy(tma_atom_b, tBgB[None, producer_state.count],
                          tBsB[None, producer_state.index],
                          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state))
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

            # Steady state: 继续为剩余 K tiles 生产
            for kidx in range(self.num_stages, num_k_tiles):
                mainloop_pipeline.producer_acquire(producer_state)
                cute.copy(tma_atom_a, tAgA[None, producer_state.count],
                          tAsA[None, producer_state.index],
                          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state))
                cute.copy(tma_atom_b, tBgB[None, producer_state.count],
                          tBsB[None, producer_state.index],
                          tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state))
                mainloop_pipeline.producer_commit(producer_state)
                producer_state.advance()

        # ====== MMA warps (consumer): LdMatrix + WMMA ======
        if is_mma_warp:
            tCrC.fill(0.0)

            for kidx in range(num_k_tiles):
                mainloop_pipeline.consumer_wait(consumer_state)

                # 循环内只做 cute.slice_ + partition_S (SMEM source)
                sA_stage = cute.slice_(sA, (None, None, consumer_state.index))
                sB_stage = cute.slice_(sB, (None, None, consumer_state.index))

                tCsA_copy_view = thr_s2r_A.partition_S(sA_stage)
                tCsB_copy_view = thr_s2r_B.partition_S(sB_stage)

                cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
                cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
                cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

                mainloop_pipeline.consumer_release(consumer_state)
                consumer_state.advance()

            # 写回 GMEM
            atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
            tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
            for i in range(cute.size(tCrC_out)):
                tCrC_out[i] = cutlass.Float16(tCrC[i])
            cute.copy(atom_store, tCrC_out, tCgC)

    @staticmethod
    def _make_tma(tensor, smem_layout_staged, smem_tile):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile)


# =============================================================================
# 测试
# =============================================================================
if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096

    print("=" * 70)
    print(f"CuTeDSL GEMM kernel 性能对比 (M={M}, N={N}, K={K})")
    print("=" * 70)

    results = []
    cc = torch.cuda.get_device_capability()

    A_f16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B_f16 = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref_f16 = torch.matmul(A_f16, B_f16.T)
    A_f16_ = from_dlpack(A_f16, assumed_align=16)
    B_f16_ = from_dlpack(B_f16, assumed_align=16)

    def record_result(name: str, elapsed_us: float):
        results.append((name, elapsed_us))
        print(f"✅ {name:<44} 耗时: {elapsed_us:.2f} µs")

    # CUDA Core 标量 FMA 路径：用 fp32 输入/输出验证 CuTe 分块与 MmaUniversalOp。
    C_scalar = torch.empty(M, N, device="cuda", dtype=torch.float32)
    C_scalar_ = from_dlpack(C_scalar, assumed_align=16)
    A_f32 = A_f16.float()
    B_f32 = B_f16.float()
    ref_f32 = torch.matmul(A_f32, B_f32.T)
    A_f32_ = from_dlpack(A_f32, assumed_align=16)
    B_f32_ = from_dlpack(B_f32, assumed_align=16)
    scalar_kernel = cute.compile(cuda_core_scalar_gemm, A_f32_, B_f32_, C_scalar_)
    scalar_kernel(A_f32_, B_f32_, C_scalar_)
    assert torch.allclose(C_scalar, ref_f32, atol=1e-1, rtol=1e-1), (
        f"cuda_core_scalar_gemm 失败 max_diff={(C_scalar - ref_f32).abs().max().item()}")
    scalar_time = benchmark(
        scalar_kernel, kernel_arguments=JitArguments(A_f32_, B_f32_, C_scalar_))
    record_result("cuda_core_scalar_gemm (fp32 scalar FMA)", scalar_time)

    # Tensor Core + 向量化 copy：普通 WMMA 学习和实战都从这里开始。
    C_vectorized = torch.empty(M, N, device="cuda", dtype=torch.float16)
    C_vectorized_ = from_dlpack(C_vectorized, assumed_align=16)
    vectorized_kernel_obj = GemmWmmaVectorizedCopy()
    vectorized_kernel = cute.compile(vectorized_kernel_obj, A_f16_, B_f16_, C_vectorized_)
    vectorized_kernel(A_f16_, B_f16_, C_vectorized_)
    assert torch.allclose(C_vectorized, ref_f16, atol=1e-1, rtol=1e-1), (
        f"GemmWmmaVectorizedCopy 失败 max_diff={(C_vectorized - ref_f16).abs().max().item()}")
    vectorized_time = benchmark(
        vectorized_kernel, kernel_arguments=JitArguments(A_f16_, B_f16_, C_vectorized_))
    record_result("GemmWmmaVectorizedCopy", vectorized_time)

    if cc >= (9, 0):
        C_tma = torch.empty(M, N, device="cuda", dtype=torch.float16)
        C_tma_ = from_dlpack(C_tma, assumed_align=16)
        tma_kernel_obj = GemmWmmaTmaSingleStage()
        tma_kernel = cute.compile(tma_kernel_obj, A_f16_, B_f16_, C_tma_)
        tma_kernel(A_f16_, B_f16_, C_tma_)
        assert torch.allclose(C_tma, ref_f16, atol=1e-1, rtol=1e-1), (
            f"GemmWmmaTmaSingleStage 失败 max_diff={(C_tma - ref_f16).abs().max().item()}")
        tma_time = benchmark(tma_kernel, kernel_arguments=JitArguments(A_f16_, B_f16_, C_tma_))
        record_result("GemmWmmaTmaSingleStage (SM90+)", tma_time)
    else:
        print(f"⏭️  GemmWmmaTmaSingleStage 跳过 (需要 SM90+, 当前 SM{cc[0]}{cc[1]})")

    print("-" * 70)
    print("软件流水线: 手写 cp.async 与 PipelineCpAsync")
    print("-" * 70)

    C_cpasync = torch.empty(M, N, device="cuda", dtype=torch.float16)
    C_cpasync_ = from_dlpack(C_cpasync, assumed_align=16)
    cpasync_kernel_obj = GemmWmmaCpAsyncMultistage(cta_tiler=(128, 128, 32), num_stages=3)
    cpasync_kernel = cute.compile(cpasync_kernel_obj, A_f16_, B_f16_, C_cpasync_)
    cpasync_kernel(A_f16_, B_f16_, C_cpasync_)
    cpasync_max_diff = (C_cpasync.float() - ref_f16.float()).abs().max().item()
    if cpasync_max_diff < 5.0:
        cpasync_time = benchmark(
            cpasync_kernel, kernel_arguments=JitArguments(A_f16_, B_f16_, C_cpasync_))
        record_result("GemmWmmaCpAsyncMultistage", cpasync_time)
    else:
        print(f"❌ GemmWmmaCpAsyncMultistage 精度不满足  max_diff={cpasync_max_diff:.4f}")

    if cc >= (9, 0):
        C_pipeline = torch.empty(M, N, device="cuda", dtype=torch.float16)
        C_pipeline_ = from_dlpack(C_pipeline, assumed_align=16)
        pipeline_kernel_obj = GemmWmmaPipelineCpAsync(cta_tiler=(128, 128, 32), num_stages=4)
        pipeline_kernel = cute.compile(pipeline_kernel_obj, A_f16_, B_f16_, C_pipeline_)
        pipeline_kernel(A_f16_, B_f16_, C_pipeline_)
        pipeline_max_diff = (C_pipeline.float() - ref_f16.float()).abs().max().item()
        if pipeline_max_diff < 5.0:
            pipeline_time = benchmark(
                pipeline_kernel, kernel_arguments=JitArguments(A_f16_, B_f16_, C_pipeline_))
            record_result("GemmWmmaPipelineCpAsync (SM90+)", pipeline_time)
        else:
            print(f"❌ GemmWmmaPipelineCpAsync 精度不满足  max_diff={pipeline_max_diff:.4f}")
    else:
        print(f"⏭️  GemmWmmaPipelineCpAsync 跳过 (需要 SM90+, 当前 SM{cc[0]}{cc[1]})")

    if cc >= (9, 0):
        C_tma_pipeline = torch.empty(M, N, device="cuda", dtype=torch.float16)
        C_tma_pipeline_ = from_dlpack(C_tma_pipeline, assumed_align=16)
        tma_pipeline_kernel_obj = GemmWmmaTmaWarpSpecializedPipeline(
            cta_tiler=(128, 128, 64), num_stages=2)
        tma_pipeline_kernel = cute.compile(
            tma_pipeline_kernel_obj, A_f16_, B_f16_, C_tma_pipeline_)
        tma_pipeline_kernel(A_f16_, B_f16_, C_tma_pipeline_)
        tma_pipeline_max_diff = (C_tma_pipeline.float() - ref_f16.float()).abs().max().item()
        if tma_pipeline_max_diff < 5.0:
            tma_pipeline_time = benchmark(
                tma_pipeline_kernel,
                kernel_arguments=JitArguments(A_f16_, B_f16_, C_tma_pipeline_))
            record_result("GemmWmmaTmaWarpSpecializedPipeline (SM90+)", tma_pipeline_time)
        else:
            print(
                "❌ GemmWmmaTmaWarpSpecializedPipeline 精度不满足  "
                f"max_diff={tma_pipeline_max_diff:.4f}")
    else:
        print(
            f"⏭️  GemmWmmaTmaWarpSpecializedPipeline 跳过 "
            f"(需要 SM90+, 当前 SM{cc[0]}{cc[1]})")

    # PyTorch 基准
    C_pt = torch.empty_like(ref_f16)
    for _ in range(10):
        torch.matmul(A_f16, B_f16.T, out=C_pt)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        torch.matmul(A_f16, B_f16.T, out=C_pt)
    end.record()
    torch.cuda.synchronize()
    pt_time = start.elapsed_time(end) * 1000 / 100
    print(f"📊 PyTorch torch.matmul              耗时: {pt_time:.2f} µs")
    results.append(("PyTorch torch.matmul (fp16)", pt_time))

    # 结果汇总
    flops = 2 * M * N * K
    print(f"\n{'='*70}")
    print(f"  {'Kernel':<52} {'耗时(µs)':<10} {'TFLOPS':<10}")
    print(f"  {'-'*70}")
    for name, elapsed_us in results:
        tflops = flops / (elapsed_us * 1e6)
        print(f"  {name:<52} {elapsed_us:<10.2f} {tflops:<10.4f}")
    print(f"{'='*70}")
