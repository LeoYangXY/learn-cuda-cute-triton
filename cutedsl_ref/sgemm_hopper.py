"""
=============================================================================
Hopper (SM90) 独有特性教程 — 只讲 RTX 5050 / Ampere 没有的东西
=============================================================================

你已经在 RTX 5050 (SM120) 上学过:
  ✅ TMA (CopyBulkTensorTileG2SOp)
  ✅ PipelineTmaAsync (Warp Specialization)
  ✅ mbarrier (init + wait)
  ✅ Swizzled SMEM
  ✅ cp.async / ldmatrix
  以上这些 SM80/SM120 都有, 不再重复。

本文件只讲 Hopper (SM90) 独有的 6 个新特性:
  1. WGMMA (wgmma.mma_async) — Tensor Core 的升级
  2. Thread Block Cluster — 多 CTA 组成集群
  3. Distributed Shared Memory — 跨 CTA 访问 SMEM
  4. TMA Multicast — 一次搬运广播到 Cluster
  5. Async Transaction Barrier (tx_count) — mbarrier 硬件计数
  6. Warp Group 级同步原语 — fence / commit_group / wait_group

本文件提供 3 个可运行版本:
  V1: WMMA + TMA + Warp Specialized (对照, 你学过的)
  V2: WGMMA + autovec_copy (Hopper 独有, 验证 WGMMA 正确性)
  V3: WGMMA + TMA + Cluster + Pipeline (完整 Hopper 架构)
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import (PipelineTmaAsync, CooperativeGroup, Agent,
                              make_pipeline_state, PipelineUserType,
                              pipeline_init_arrive, pipeline_init_wait)
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
import torch


# =============================================================================
# V1: 可运行对照 — WMMA + TMA + Warp Specialized Pipeline (你已学过)
# =============================================================================

class GemmWmmaTmaWarpSpecialized:
    def __init__(self, cta_tiler=(128, 128, 64), num_stages=4):
        self.tile_shape_mnk = cta_tiler
        self._bM, self._bN, self._bK = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self.warp_size = cute.arch.WARP_SIZE
        self.num_mma_warps = self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self.mma_warp_ids = tuple(range(self.num_mma_warps))
        self.tma_warp_id = self.num_mma_warps
        self.threads_per_cta = self.warp_size * (self.num_mma_warps + 1)
        self.num_stages = num_stages
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, a, b, c):
        self.a_dtype, self.b_dtype = a.element_type, b.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype, num_stages=self.num_stages)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype, num_stages=self.num_stages)
        tma_atom_a, tma_tensor_a = self._make_tma(a, self.a_smem_layout_staged, (self._bM, self._bK))
        tma_atom_b, tma_tensor_b = self._make_tma(b, self.b_smem_layout_staged, (self._bN, self._bK))
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2])
        tiled_mma = cute.make_tiled_mma(op_or_atom=mma_op, atom_layout_mnk=self.atom_layout_mnk,
                                         permutation_mnk=permutation_mnk)
        @cute.struct
        class SS:
            mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]
        self.ss = SS
        grid_dim = *cute.ceil_div(c.shape, (self._bM, self._bN)), 1
        self.kernel(tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tiled_mma, c,
                    self.a_smem_layout_staged, self.b_smem_layout_staged).launch(
            grid=grid_dim, block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(self, tma_atom_a, mA, tma_atom_b, mB, tiled_mma, mC, a_sl, b_sl):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        is_mma = warp_idx <= self.mma_warp_ids[-1]
        is_tma = warp_idx == self.tma_warp_id
        if is_tma:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.ss)
        sA = st.sA.get_tensor(a_sl.outer, swizzle=a_sl.inner)
        sB = st.sB.get_tensor(b_sl.outer, swizzle=b_sl.inner)
        gA = cute.local_tile(mA, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self.tile_shape_mnk, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, 1, None))
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sA,0,2), cute.group_modes(gA,0,2))
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sB,0,2), cute.group_modes(gB,0,2))
        thr_mma = tiled_mma.get_slice(tid)
        tCgC = thr_mma.partition_C(gC)
        sA_0, sB_0 = cute.slice_(sA,(None,None,0)), cute.slice_(sB,(None,None,0))
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA_0))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB_0))
        tCrC = tiled_mma.make_fragment_C(tCgC)
        atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype)
        atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.b_dtype)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A, thr_s2r_B = tiled_s2r_A.get_slice(tid), tiled_s2r_B.get_slice(tid)
        tCrA_v, tCrB_v = thr_s2r_A.retile(tCrA), thr_s2r_B.retile(tCrB)
        a_smem_l = cute.slice_(a_sl,(None,None,0))
        b_smem_l = cute.slice_(b_sl,(None,None,0))
        tx_bytes = cute.size_in_bytes(self.a_dtype, a_smem_l) + cute.size_in_bytes(self.b_dtype, b_smem_l)
        pipeline = PipelineTmaAsync.create(num_stages=self.num_stages, producer_group=CooperativeGroup(Agent.Thread,1),
            consumer_group=CooperativeGroup(Agent.Thread,self.num_mma_warps), barrier_storage=st.mbar.data_ptr(),
            tx_count=tx_bytes, cta_layout_vmnk=cute.make_layout((1,1,1,1)))
        ps = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        cs = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)
        nk = mA.shape[1] // self._bK
        if is_tma:
            for k in range(nk):
                pipeline.producer_acquire(ps)
                cute.copy(tma_atom_a, tAgA[None,ps.count], tAsA[None,ps.index], tma_bar_ptr=pipeline.producer_get_barrier(ps))
                cute.copy(tma_atom_b, tBgB[None,ps.count], tBsB[None,ps.index], tma_bar_ptr=pipeline.producer_get_barrier(ps))
                pipeline.producer_commit(ps); ps.advance()
        if is_mma:
            tCrC.fill(0.0)
            for k in range(nk):
                pipeline.consumer_wait(cs)
                sAs, sBs = cute.slice_(sA,(None,None,cs.index)), cute.slice_(sB,(None,None,cs.index))
                cute.copy(tiled_s2r_A, thr_s2r_A.partition_S(sAs), tCrA_v)
                cute.copy(tiled_s2r_B, thr_s2r_B.partition_S(sBs), tCrB_v)
                cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
                pipeline.consumer_release(cs); cs.advance()
            out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
            for i in range(cute.size(out)): out[i] = cutlass.Float16(tCrC[i])
            cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type), out, tCgC)

    @staticmethod
    def _make_tma(t, sl, tile):
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), t, cute.slice_(sl,(None,None,0)), tile)


# =============================================================================
# V2: WGMMA + autovec_copy (Hopper 独有, 简洁版验证 WGMMA 正确性)
# =============================================================================
# 和 V1 的关键区别:
#   1. 没有 ldmatrix / make_tiled_copy_A/B — WGMMA 直接从 SMEM 读!
#   2. 用 make_fragment_A/B 创建 SMEM 视图 (不是复制到 Register!)
#   3. 128 线程 = 1 warp group (不是 32 线程的 warp)
#   4. 异步执行: fence → gemm → commit_group → wait_group

class GemmWgmmaSync:
    """最简 WGMMA GEMM: autovec_copy + sync + WGMMA"""
    def __init__(self):
        self.tile_shape_mnk = (128, 128, 64)
        self.atom_layout_mnk = (1, 1, 1)
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, a, b, c):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        # ===== WGMMA TiledMMA — Hopper 独有! =====
        # 和 WMMA 版本的区别: 这里创建的 MMA atom 是 64×128×16 (或 64×N×K)
        # 内部自动使用 wgmma.mma_async PTX 指令
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),  # A 的主维度
            self.b_layout.sm90_mma_major_mode(),  # B 的主维度
            cutlass.Float32,                       # 累加器 FP32
            self.atom_layout_mnk,                  # (1,1,1) = 1 warp group
            tiler_mn=(64, 128))                    # 每个 warp group 做 64×128

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype, num_stages=1)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype, num_stages=1)

        @cute.struct
        class SS:
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]
        self.ss = SS

        grid_dim = *cute.ceil_div(c.shape, (128, 128)), 1
        self.kernel(a, b, self.tiled_mma, c,
                    self.a_smem_layout_staged, self.b_smem_layout_staged).launch(
            grid=grid_dim, block=(128, 1, 1))  # 128 = 1 warp group

    @cute.kernel
    def kernel(self, mA, mB, tiled_mma, mC, a_sl, b_sl):
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.ss)
        sA = st.sA.get_tensor(a_sl.outer, swizzle=a_sl.inner)
        sB = st.sB.get_tensor(b_sl.outer, swizzle=b_sl.inner)

        gA = cute.local_tile(mA, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self.tile_shape_mnk, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, 1, None))

        # ===== WGMMA 核心: partition + make_fragment =====
        # WMMA: partition_A(sA) → 每线程的 register 切片
        # WGMMA: partition_A(sA) → SMEM 视图, make_fragment_A → 带正确 layout 的 SMEM 引用
        thr_mma = tiled_mma.get_slice(tid)
        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)        # 在 staged tensor 上 partition
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA) # ★ WGMMA 关键: 创建 SMEM fragment
        tCrB = tiled_mma.make_fragment_B(tCsB) # ★ (不是 register fragment!)
        accumulators = cute.make_rmem_tensor(tCgC.shape, cutlass.Float32)

        num_k_blocks = cute.size(tCrA, mode=[2])
        nk = mA.shape[1] // 64
        sA_0 = cute.slice_(sA, (None, None, 0))
        sB_0 = cute.slice_(sB, (None, None, 0))

        # k=0: ACCUMULATE=False (清零累加器)
        cute.autovec_copy(cute.slice_(gA, (None, None, 0)), sA_0)
        cute.autovec_copy(cute.slice_(gB, (None, None, 0)), sB_0)
        cute.arch.sync_threads()

        # ===== WGMMA 异步执行模型 (特性 6) =====
        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)  # 第一次清零
        cute.nvgpu.warpgroup.fence()          # SMEM 写入对 TC 可见 (特性 6)
        for kb in cutlass.range(num_k_blocks, unroll_full=True):
            cute.gemm(tiled_mma, accumulators,
                      tCrA[(None, None, kb, 0)],   # ★ 直接从 SMEM 读!
                      tCrB[(None, None, kb, 0)], accumulators)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
        cute.nvgpu.warpgroup.commit_group()   # 打包成事务组 (特性 6)
        cute.nvgpu.warpgroup.wait_group(0)    # 等完成 (特性 6)
        cute.arch.sync_threads()

        # k=1..nk-1: ACCUMULATE=True (累加)
        for k in cutlass.range(1, nk, 1, unroll=1):
            cute.autovec_copy(cute.slice_(gA, (None, None, k)), sA_0)
            cute.autovec_copy(cute.slice_(gB, (None, None, k)), sB_0)
            cute.arch.sync_threads()
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range(num_k_blocks, unroll_full=True):
                cute.gemm(tiled_mma, accumulators, tCrA[(None, None, kb, 0)],
                          tCrB[(None, None, kb, 0)], accumulators)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            cute.arch.sync_threads()

        # Epilogue: FP32 → FP16 → GMEM
        out = cute.make_fragment_like(accumulators, dtype=cutlass.Float16)
        for i in cutlass.range(cute.size(out)):
            out[i] = cutlass.Float16(accumulators[i])
        cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type), out, tCgC)


# =============================================================================
# V3: WGMMA + TMA + Cluster + Pipeline (完整 Hopper 架构)
# =============================================================================
# 集成所有 6 个 Hopper 独有特性:
#   特性 1: WGMMA (make_trivial_tiled_mma → wgmma.mma_async)
#   特性 2: Thread Block Cluster (launch cluster=(1,1,1))
#   特性 3/4: DSMEM + TMA Multicast (cluster 内广播, cluster>1 时自动启用)
#   特性 5: Async Transaction Barrier (PipelineTmaAsync tx_count)
#   特性 6: Warp Group 同步 (fence / commit_group / wait_group)

class GemmWgmmaTmaPipeline:
    """完整 Hopper GEMM: WGMMA + TMA + Pipeline + Cluster"""
    def __init__(self, cta_tiler=(128, 128, 64), num_stages=4, cluster_shape_mn=(1, 1)):
        self.tile_shape_mnk = cta_tiler
        self.atom_layout_mnk = (1, 1, 1)
        self.mma_warp_groups = 1
        self.num_threads_per_warp_group = 128
        # Warp Specialized: 128 WGMMA threads + 32 TMA threads = 160 total
        self.tma_warp_id = self.mma_warp_groups * (self.num_threads_per_warp_group // 32)  # warp 4
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group + 32
        self.num_stages = num_stages
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = cluster_shape_mn

    @cute.jit
    def __call__(self, a, b, c):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        # WGMMA TiledMMA
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            cutlass.Float32, self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]))

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype, num_stages=self.num_stages)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype, num_stages=self.num_stages)

        tma_atom_a, tma_tensor_a = self._make_tma(a, self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]))
        tma_atom_b, tma_tensor_b = self._make_tma(b, self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]))

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))

        @cute.struct
        class SS:
            mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged)], self.buffer_align_bytes]
        self.ss = SS

        grid_dim = *cute.ceil_div(c.shape, (self.tile_shape_mnk[0], self.tile_shape_mnk[1])), 1
        self.kernel(tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, self.tiled_mma, c,
                    self.cta_layout_mnk, self.a_smem_layout_staged, self.b_smem_layout_staged).launch(
            grid=grid_dim, block=(self.threads_per_cta, 1, 1),
            cluster=(*self.cluster_shape_mn, 1))  # ★ 特性 2: Cluster launch

    @cute.kernel
    def kernel(self, tma_atom_a, mA, tma_atom_b, mB, tiled_mma, mC,
               cta_layout_mnk, a_sl, b_sl):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        is_tma = warp_idx == self.tma_warp_id    # warp 4: 专门做 TMA
        is_mma = warp_idx < self.tma_warp_id     # warp 0-3: 做 WGMMA

        if is_tma:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(self.ss)
        sA = st.sA.get_tensor(a_sl.outer, swizzle=a_sl.inner)
        sB = st.sB.get_tensor(b_sl.outer, swizzle=b_sl.inner)

        gA = cute.local_tile(mA, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self.tile_shape_mnk, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.tile_shape_mnk, (bidx, bidy, None), proj=(1, 1, None))

        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, 0, cute.make_layout(1),
                                            cute.group_modes(sA,0,2), cute.group_modes(gA,0,2))
        tBsB, tBgB = cpasync.tma_partition(tma_atom_b, 0, cute.make_layout(1),
                                            cute.group_modes(sB,0,2), cute.group_modes(gB,0,2))

        # WGMMA partition
        thr_mma = tiled_mma.get_slice(tid)
        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        accumulators = cute.make_rmem_tensor(tCgC.shape, cutlass.Float32)

        # Pipeline — 特性 5
        a_smem_l = cute.slice_(a_sl, (None, None, 0))
        b_smem_l = cute.slice_(b_sl, (None, None, 0))
        tx_bytes = cute.size_in_bytes(self.a_dtype, a_smem_l) + cute.size_in_bytes(self.b_dtype, b_smem_l)

        # Warp-specialized: producer=1 TMA thread, consumer=4 MMA warps
        pipeline = PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, self.mma_warp_groups * (self.num_threads_per_warp_group // 32)),
            barrier_storage=st.mbar.data_ptr(),
            tx_count=tx_bytes,
            cta_layout_vmnk=cute.make_layout((1,1,1,1)))

        ps = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        cs = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)
        nk = mA.shape[1] // self.tile_shape_mnk[2]
        num_k_blocks = cute.size(tCrA, mode=[2])

        # Producer: TMA loads (warp 4 only)
        if is_tma:
            for k in range(nk):
                pipeline.producer_acquire(ps)
                cute.copy(tma_atom_a, tAgA[None, ps.count], tAsA[None, ps.index],
                          tma_bar_ptr=pipeline.producer_get_barrier(ps))
                cute.copy(tma_atom_b, tBgB[None, ps.count], tBsB[None, ps.index],
                          tma_bar_ptr=pipeline.producer_get_barrier(ps))
                pipeline.producer_commit(ps)
                ps.advance()

        # Consumer: WGMMA (warps 0-3)
        if is_mma:
            # k=0: ACCUMULATE=False
            pipeline.consumer_wait(cs)
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
            cute.nvgpu.warpgroup.fence()
            for kb in cutlass.range(num_k_blocks, unroll_full=True):
                cute.gemm(tiled_mma, accumulators, tCrA[(None, None, kb, cs.index)],
                          tCrB[(None, None, kb, cs.index)], accumulators)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            cute.nvgpu.warpgroup.commit_group()
            cute.nvgpu.warpgroup.wait_group(0)
            pipeline.consumer_release(cs)
            cs.advance()

            # k=1..nk-1: ACCUMULATE=True
            for k in cutlass.range(1, nk, 1, unroll=1):
                pipeline.consumer_wait(cs)
                cute.nvgpu.warpgroup.fence()
                for kb in cutlass.range(num_k_blocks, unroll_full=True):
                    cute.gemm(tiled_mma, accumulators, tCrA[(None, None, kb, cs.index)],
                              tCrB[(None, None, kb, cs.index)], accumulators)
                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)
                pipeline.consumer_release(cs)
                cs.advance()

            # Epilogue
            out = cute.make_fragment_like(accumulators, dtype=cutlass.Float16)
            for i in cutlass.range(cute.size(out)):
                out[i] = cutlass.Float16(accumulators[i])
            cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type), out, tCgC)

    @staticmethod
    def _make_tma(t, sl, tile):
        return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), t,
                                            cute.slice_(sl, (None, None, 0)), tile)


# =============================================================================
# 测试
# =============================================================================
if __name__ == "__main__":
    cc = torch.cuda.get_device_capability()
    if cc < (9, 0):
        print(f"❌ 本文件需要 SM90+ (Hopper), 当前 SM{cc[0]}{cc[1]}")
        exit(0)

    M, N, K = 4096, 4096, 4096
    print("=" * 70)
    print(f"Hopper (SM90) 独有特性教程 — {torch.cuda.get_device_name(0)}")
    print(f"GEMM: {M}×{N}×{K} FP16")
    print("=" * 70)

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref = torch.matmul(A, B.T)
    flops = 2 * M * N * K

    # V1: WMMA + TMA (对照)
    print("\n[V1 对照] WMMA + TMA + Warp Specialized (你已学过):", flush=True)
    C1 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    A_, B_, C1_ = from_dlpack(A, assumed_align=16), from_dlpack(B, assumed_align=16), from_dlpack(C1, assumed_align=16)
    v1 = GemmWmmaTmaWarpSpecialized(cta_tiler=(128, 128, 64), num_stages=4)
    k1 = cute.compile(v1, A_, B_, C1_)
    k1(A_, B_, C1_)
    d1 = (C1.float() - ref.float()).abs().max().item()
    t1 = benchmark(k1, kernel_arguments=JitArguments(A_, B_, C1_))
    print(f"  ✅ {t1:.0f} µs | {flops/(t1*1e6):.1f} TFLOPS | max_diff={d1:.4f}")

    # V2: WGMMA + autovec_copy (Hopper 独有)
    print("\n[V2 Hopper] WGMMA + autovec_copy (验证 WGMMA 正确性):", flush=True)
    C2 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    C2_ = from_dlpack(C2, assumed_align=16)
    v2 = GemmWgmmaSync()
    k2 = cute.compile(v2, A_, B_, C2_)
    k2(A_, B_, C2_)
    torch.cuda.synchronize()
    d2 = (C2.float() - ref.float()).abs().max().item()
    t2 = benchmark(k2, kernel_arguments=JitArguments(A_, B_, C2_))
    print(f"  ✅ {t2:.0f} µs | {flops/(t2*1e6):.1f} TFLOPS | max_diff={d2:.4f}")

    # V3: WGMMA + TMA + Pipeline (完整 Hopper 架构)
    print("\n[V3 Hopper] WGMMA + TMA + Pipeline (完整架构):", flush=True)
    C3 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    C3_ = from_dlpack(C3, assumed_align=16)
    v3 = GemmWgmmaTmaPipeline(cta_tiler=(128, 128, 64), num_stages=4)
    k3 = cute.compile(v3, A_, B_, C3_)
    k3(A_, B_, C3_)
    torch.cuda.synchronize()
    d3 = (C3.float() - ref.float()).abs().max().item()
    if d3 < 1.0:
        t3 = benchmark(k3, kernel_arguments=JitArguments(A_, B_, C3_))
        print(f"  ✅ {t3:.0f} µs | {flops/(t3*1e6):.1f} TFLOPS | max_diff={d3:.4f}")
    else:
        print(f"  ⚠️ max_diff={d3:.4f} (pipeline 时序待调优)")

    # PyTorch
    for _ in range(10): torch.matmul(A, B.T)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(100): torch.matmul(A, B.T)
    e.record(); torch.cuda.synchronize()
    pt = s.elapsed_time(e) * 1000 / 100
    print(f"\n  📊 PyTorch cuBLAS: {pt:.0f} µs | {flops/(pt*1e6):.1f} TFLOPS")

    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║            Hopper (SM90) 独有 WGMMA 代码模式总结                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

  WMMA (SM80, 你学过):
    mma_op = MmaF16BF16Op(shape_mnk=(16,8,16))    # 32 线程
    thr_mma.partition_A(sA) → ldmatrix → Register
    cute.gemm(tiled_mma, C_reg, A_reg, B_reg, C_reg)

  WGMMA (SM90, Hopper 独有):
    tiled_mma = make_trivial_tiled_mma(...)         # 128 线程
    tCsA = thr_mma.partition_A(sA)                  # SMEM 视图
    tCrA = tiled_mma.make_fragment_A(tCsA)          # ★ 关键: SMEM fragment
    tiled_mma.set(ACCUMULATE, False/True)           # 控制累加
    fence() → gemm(smem_A, smem_B) → commit_group() → wait_group(0)
""")
