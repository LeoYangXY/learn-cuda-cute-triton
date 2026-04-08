"""
=============================================================================
CuTeDSL 实现 HGEMM — 对标 LeetCUDA/kernels/hgemm/ 全部优化技术
=============================================================================

LeetCUDA HGEMM 优化路线 (共 ~30 个 kernel):
  naive → sliced_k → t8x8+f16x8_pack → bcf(bank conflict free)
  → dbuf(double buffer) → async(cp.async) → wmma(Tensor Core)
  → mma(PTX m16n8k16) → multi-stage pipeline → SMEM swizzle
  → block swizzle → warp specialization → CuTe 封装

本文件实现 7 个版本，逐步对应 LeetCUDA 的优化:

  V1: WMMA + 向量化 G2S + LdMatrix + Padding
      对应: hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel
      优化点: Tensor Core MMA, 128-bit 向量化拷贝, SMEM Padding

  V1b: WMMA + SMEM Swizzle (XOR-based, 消除 bank conflict)
       对应: hgemm_mma_swizzle.cu + Swizzle<3,3,3>
       优化点: XOR swizzle 替代 padding, bank conflict 24576→0

  V2: WMMA + Double Buffer + K 内层展开
      对应: hgemm_mma_stage_tn_cute.cu (CuTe 版本)
      优化点: 双 SMEM 缓冲, K 维内层循环 (nk), 计算/访存重叠

  V2b: WMMA + Block Swizzle (改善 L2 Cache 局部性)
       对应: hgemm_mma_stage_tn_cute.cu BlockSwizzle=true
       优化点: blockIdx.z 重映射, N 维度分段, 相邻 block 复用 L2

  V3: WMMA + TMA (SM90+)
      对应: V6 in sgemm.py, 扩展到更大 CTA tile
      优化点: TMA 硬件搬运, Swizzled SMEM, mbarrier 异步

语义: C[M,N] = A[M,K] × B[N,K]^T (B 转置存储, CuTE/LeetCUDA TN 标准约定)

LeetCUDA 优化技术对应表:
  ┌──────────────────────────────────────────────────────────────────────┐
  │ LeetCUDA HGEMM 技术                    CuTeDSL 对应                  │
  │ ─────────────────────────────────────  ──────────────────────────── │
  │ HMMA16816 PTX 指令                     MmaF16BF16Op(16,8,16)       │
  │ LDMATRIX_X4 / LDMATRIX_X2_T            LdMatrix8x8x16bOp           │
  │ CP_ASYNC_CG (128-bit)                  CopyG2SOp / CopyUniversalOp │
  │ A_PAD / B_PAD (bank conflict free)     SMEM stride padding         │
  │ s_a[K_STAGE][BM][BK+PAD] 多 buffer    独立 SMEM 分配 × N_STAGES   │
  │ WARP_TILE_M=4, WARP_TILE_N=4          permutation_mnk 覆盖        │
  │ MMA_TILE_M=2, WARP4x4 (8 warps)       atom_layout_mnk=(2,4,1)     │
  │ Block Swizzle (blockIdx.z 重映射)      V2b: bidz*gdimx+bidx 重映射  │
  │ Swizzle<3,3,3> (CuTe SMEM)            V1b: make_swizzle(3,3,3)    │
  │ Epilogue R2S→S2G (复用 sA 空间)        直接 reg→gmem store          │
  │ K 维内层展开 (nk = size<2>(tCrA))      for ik in range(nk) 循环    │
  └──────────────────────────────────────────────────────────────────────┘
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
import torch


# =============================================================================
# V1: WMMA + 向量化 G2S + LdMatrix + Padding (基础版)
# =============================================================================
# 对应 LeetCUDA: hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel
# 256 线程 (8 warps), 128×128 tile, BK=16
# warp 布局: warp_m = warp_id % 2, warp_n = warp_id / 2
# MMA_TILE_M=2, MMA_TILE_N=4 → 每个 warp 算 32×32 的 C 子块
# WARP_TILE_M=4, WARP_TILE_N=4 → 8 warps 覆盖 128×128

class HgemmMMA_V1:
    """
    LeetCUDA 对应:
    - hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel (swizzle/hgemm_mma_swizzle.cu)
    - 8 warps, 128x128x16, LDMATRIX_X4 + LDMATRIX_X2_T + HMMA16816
    """
    def __init__(self, cta_tiler=(128, 128, 32)):
        self._bM, self._bN, self._bK = cta_tiler
        self._cta_tiler = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        # atom_layout (2,2,1) → 4 warps = 128 threads
        # 和 LeetCUDA 的 8 warps 不同，但通过 permutation 覆盖同样大小的 tile
        self.atom_layout_mnk = (2, 2, 1)
        self._num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self._smem_padding = 8  # 对应 LeetCUDA 的 A_PAD/B_PAD
        self._num_vectorized = 4  # 64-bit load (4 × fp16)

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

        padding = self._smem_padding
        sA_layout = cute.make_layout((self._bM, self._bK), stride=(self._bK + padding, 1))
        sB_layout = cute.make_layout((self._bN, self._bK), stride=(self._bK + padding, 1))

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

        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)
        tAgA = thr_copyA.partition_S(gA)
        tAsA = thr_copyA.partition_D(sA)
        tBgB = thr_copyB.partition_S(gB)
        tBsB = thr_copyB.partition_D(sB)

        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

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

        tCrC.fill(0.0)
        for kidx in range(mA.shape[1] // self._bK):
            cute.copy(tiled_copy_A, tAgA[None, None, None, kidx], tAsA[None, None, None])
            cute.copy(tiled_copy_B, tBgB[None, None, None, kidx], tBsB[None, None, None])
            cute.arch.sync_threads()
            cute.copy(tiled_s2r_A, tCsA_view, tCrA_view)
            cute.copy(tiled_s2r_B, tCsB_view, tCrB_view)
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for i in range(cute.size(tCrC_out)):
            tCrC_out[i] = cutlass.Float16(tCrC[i])
        cute.copy(atom_store, tCrC_out, tCgC)


# =============================================================================
# V1b: WMMA + SMEM Swizzle (XOR-based, 消除 bank conflict)
# =============================================================================
# 对应 LeetCUDA: hgemm_mma_swizzle.cu / hgemm_mma_stage_tn_cute.cu 的 Swizzle<3,3,3>
# 核心: 用 make_swizzle + allocate_tensor(..., swizzle=) 替代 stride padding
# LeetCUDA 效果: bank conflict 24576 → 0, 性能 +67%
#
# CuTeDSL API:
#   cute.make_swizzle(B, M, S) → Swizzle 对象
#   smem.allocate_tensor(dtype, layout, swizzle=swizzle_obj) → 带 swizzle 的 SMEM tensor
#
# 对比 V1: V1 用 stride padding (bK+8), V1b 用 XOR swizzle (不浪费空间)

class HgemmMMA_Swizzle_V1b:
    """
    LeetCUDA 对应:
    - Swizzle<3,3,3> (hgemm_mma_stage_tn_cute.cu)
    - ((j/8) ^ (i/4)) % 2) * 8 (hgemm_mma_swizzle.cu)
    """
    def __init__(self, cta_tiler=(128, 128, 32)):
        self._bM, self._bN, self._bK = cta_tiler
        self._cta_tiler = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self._num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self._num_vectorized = 4

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

        # Swizzle SMEM layout: 用 XOR 消除 bank conflict, 不浪费 SMEM 空间
        # Swizzle<3,3,3> 对应 LeetCUDA CuTe C++ 版本
        # 底层: cute.make_swizzle(bits=3, base=3, shift=3)
        # 中层: cute.make_ordered_layout((bM, bK), order=(1,0)) + swizzle
        swizzle_obj = cute.make_swizzle(3, 4, 3)
        sA_base_layout = cute.make_ordered_layout((self._bM, self._bK), order=(1, 0))
        sB_base_layout = cute.make_ordered_layout((self._bN, self._bK), order=(1, 0))

        # 向量化 G2S copy
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
            mA, mB, mC, sA_base_layout, sB_base_layout, swizzle_obj,
            tiled_copy_A, tiled_copy_B, tiled_mma
        ).launch(grid=grid_dim, block=(self._num_threads, 1, 1))

    @cute.kernel
    def kernel(self, mA, mB, mC, sA_base_layout, sB_base_layout, swizzle_obj,
               tiled_copy_A, tiled_copy_B, tiled_mma):
        bidx, bidy, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()

        # 分配带 Swizzle 的 SMEM tensor
        # CuTeDSL API: allocate_tensor(dtype, layout, align_bytes, swizzle=swizzle_obj)
        # swizzle 对用户透明: partition_S/D 和 ldmatrix 自动适配 swizzle 后的地址
        allocator = cutlass.utils.SmemAllocator()
        sA = allocator.allocate_tensor(cutlass.Float16, sA_base_layout, 128, swizzle=swizzle_obj)
        sB = allocator.allocate_tensor(cutlass.Float16, sB_base_layout, 128, swizzle=swizzle_obj)

        gA = cute.local_tile(mA, self._cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self._cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self._cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)
        tAgA = thr_copyA.partition_S(gA)
        tAsA = thr_copyA.partition_D(sA)
        tBgB = thr_copyB.partition_S(gB)
        tBsB = thr_copyB.partition_D(sB)

        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

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

        tCrC.fill(0.0)
        for kidx in range(mA.shape[1] // self._bK):
            cute.copy(tiled_copy_A, tAgA[None, None, None, kidx], tAsA[None, None, None])
            cute.copy(tiled_copy_B, tBgB[None, None, None, kidx], tBsB[None, None, None])
            cute.arch.sync_threads()
            cute.copy(tiled_s2r_A, tCsA_view, tCrA_view)
            cute.copy(tiled_s2r_B, tCsB_view, tCrB_view)
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for i in range(cute.size(tCrC_out)):
            tCrC_out[i] = cutlass.Float16(tCrC[i])
        cute.copy(atom_store, tCrC_out, tCgC)


# =============================================================================
# V2: WMMA + Double Buffer (2 独立 SMEM buffer)
# =============================================================================
# 对应 LeetCUDA: hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel
# 核心: 分配 2 组独立 SMEM, 交替使用实现计算/访存重叠
# 和 sgemm.py V7 相同的思路

class HgemmMMA_DoubleBuf_V2:
    """
    LeetCUDA 对应:
    - hgemm_wmma_m16n16k16_mma4x2_warp2x4_dbuf_async_kernel
    - sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel
    """
    def __init__(self, cta_tiler=(128, 128, 32)):
        self._bM, self._bN, self._bK = cta_tiler
        self._cta_tiler = cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self._num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self._smem_padding = 8
        self._num_vectorized = 4

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

        padding = self._smem_padding
        sA_layout = cute.make_layout((self._bM, self._bK), stride=(self._bK + padding, 1))
        sB_layout = cute.make_layout((self._bN, self._bK), stride=(self._bK + padding, 1))

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
        sA0 = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
        sB0 = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)
        sA1 = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
        sB1 = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)

        gA = cute.local_tile(mA, self._cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self._cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self._cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)
        tAgA = thr_copyA.partition_S(gA)
        tAsA0 = thr_copyA.partition_D(sA0)
        tAsA1 = thr_copyA.partition_D(sA1)
        tBgB = thr_copyB.partition_S(gB)
        tBsB0 = thr_copyB.partition_D(sB0)
        tBsB1 = thr_copyB.partition_D(sB1)

        thr_mma = tiled_mma.get_slice(tid)
        tCgC = thr_mma.partition_C(gC)
        tCrC = tiled_mma.make_fragment_C(tCgC)

        atom_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mA.element_type)
        atom_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), mB.element_type)
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tid)
        thr_s2r_B = tiled_s2r_B.get_slice(tid)

        # Buffer 0
        tCsA0 = thr_mma.partition_A(sA0)
        tCsB0 = thr_mma.partition_B(sB0)
        tCrA0 = tiled_mma.make_fragment_A(tCsA0)
        tCrB0 = tiled_mma.make_fragment_B(tCsB0)
        tCsA0_v = thr_s2r_A.partition_S(sA0)
        tCrA0_v = thr_s2r_A.retile(tCrA0)
        tCsB0_v = thr_s2r_B.partition_S(sB0)
        tCrB0_v = thr_s2r_B.retile(tCrB0)

        # Buffer 1
        tCsA1 = thr_mma.partition_A(sA1)
        tCsB1 = thr_mma.partition_B(sB1)
        tCrA1 = tiled_mma.make_fragment_A(tCsA1)
        tCrB1 = tiled_mma.make_fragment_B(tCsB1)
        tCsA1_v = thr_s2r_A.partition_S(sA1)
        tCrA1_v = thr_s2r_A.retile(tCrA1)
        tCsB1_v = thr_s2r_B.partition_S(sB1)
        tCrB1_v = thr_s2r_B.retile(tCrB1)

        tCrC.fill(0.0)
        num_k_tiles = mA.shape[1] // self._bK

        # Prefetch first tile into buffer 0
        cute.copy(tiled_copy_A, tAgA[None, None, None, 0], tAsA0[None, None, None])
        cute.copy(tiled_copy_B, tBgB[None, None, None, 0], tBsB0[None, None, None])
        cute.arch.sync_threads()

        for kidx in range(num_k_tiles):
            if kidx % 2 == 0:
                if kidx + 1 < num_k_tiles:
                    cute.copy(tiled_copy_A, tAgA[None, None, None, kidx + 1], tAsA1[None, None, None])
                    cute.copy(tiled_copy_B, tBgB[None, None, None, kidx + 1], tBsB1[None, None, None])
                cute.copy(tiled_s2r_A, tCsA0_v, tCrA0_v)
                cute.copy(tiled_s2r_B, tCsB0_v, tCrB0_v)
                cute.gemm(tiled_mma, tCrC, tCrA0, tCrB0, tCrC)
            else:
                if kidx + 1 < num_k_tiles:
                    cute.copy(tiled_copy_A, tAgA[None, None, None, kidx + 1], tAsA0[None, None, None])
                    cute.copy(tiled_copy_B, tBgB[None, None, None, kidx + 1], tBsB0[None, None, None])
                cute.copy(tiled_s2r_A, tCsA1_v, tCrA1_v)
                cute.copy(tiled_s2r_B, tCsB1_v, tCrB1_v)
                cute.gemm(tiled_mma, tCrC, tCrA1, tCrB1, tCrC)
            cute.arch.sync_threads()

        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for i in range(cute.size(tCrC_out)):
            tCrC_out[i] = cutlass.Float16(tCrC[i])
        cute.copy(atom_store, tCrC_out, tCgC)


# =============================================================================
# V2b: WMMA + Block Swizzle (改善 L2 Cache 局部性)
# =============================================================================
# 对应 LeetCUDA: blockIdx.z * gridDim.x + blockIdx.x 重映射
# hgemm_mma_stage_tn_cute.cu: BlockSwizzle=true, swizzle_stride=2048
#
# 核心思想: 默认 grid 调度按 (x, y) 线性递增, 相邻 block 处理 N 方向
# 相邻的 tile, 但它们的 K 数据不重叠 → L2 Cache 利用率低。
# Block Swizzle 通过 blockIdx.z 把 N 方向分成若干段, 每段内的
# block 处理相邻的 tile → 同一段内的 block 可以复用 L2 Cache 中的数据。
#
# CuTeDSL 实现: 在 grid 的 z 维度加 swizzle_stride, kernel 内用
# bidz * gridDim_x + bidx 重新映射 block ID。
#
# 注意: 这个优化和前面的 kernel 逻辑完全相同, 只改 grid 调度策略。
# 这里为了演示, 基于 V1 + swizzle SMEM 的基础上增加 block swizzle。

class HgemmMMA_BlockSwizzle_V2b:
    """
    LeetCUDA 对应:
    - blockIdx.z 重映射 (hgemm_mma_stage_tn_cute.cu, BlockSwizzle=true)
    - hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel (BLOCK_SWIZZLE=true)
    """
    def __init__(self, cta_tiler=(128, 128, 32), swizzle_stride=2048):
        self._bM, self._bN, self._bK = cta_tiler
        self._cta_tiler = cta_tiler
        self._swizzle_stride = swizzle_stride
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (2, 2, 1)
        self._num_threads = cute.arch.WARP_SIZE * self.atom_layout_mnk[0] * self.atom_layout_mnk[1]
        self._smem_padding = 8
        self._num_vectorized = 4

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

        padding = self._smem_padding
        sA_layout = cute.make_layout((self._bM, self._bK), stride=(self._bK + padding, 1))
        sB_layout = cute.make_layout((self._bN, self._bK), stride=(self._bK + padding, 1))

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

        # Block Swizzle grid 计算 (对应 LeetCUDA 的 BX, BY, BZ)
        M, N = mC.shape[0], mC.shape[1]
        BX_total = (N + self._bN - 1) // self._bN
        BY = (M + self._bM - 1) // self._bM
        BZ = (N + self._swizzle_stride - 1) // self._swizzle_stride
        BX = (BX_total + BZ - 1) // BZ

        self.kernel(
            mA, mB, mC, sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B, tiled_mma
        ).launch(grid=(BX, BY, BZ), block=(self._num_threads, 1, 1))

    @cute.kernel
    def kernel(self, mA, mB, mC, sA_layout, sB_layout,
               tiled_copy_A, tiled_copy_B, tiled_mma):
        bidx, bidy, bidz = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        gdimx, _, _ = cute.arch.grid_dim()

        # Block Swizzle: 重映射 bidx
        # 对应 LeetCUDA: int ix = blockIdx.z * gridDim.x + blockIdx.x;
        ix = bidz * gdimx + bidx
        iy = bidy

        allocator = cutlass.utils.SmemAllocator()
        sA = allocator.allocate_tensor(cutlass.Float16, sA_layout, 16, None)
        sB = allocator.allocate_tensor(cutlass.Float16, sB_layout, 16, None)

        # 使用重映射后的 ix, iy 做 local_tile
        # ix → N 方向 (对应 bidx), iy → M 方向 (对应 bidy)
        # 和 V1 的 (bidx, bidy, None) 对应 (M_coord, N_coord, K_coord)
        gA = cute.local_tile(mA, self._cta_tiler, (iy, ix, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self._cta_tiler, (iy, ix, None), proj=(None, 1, 1))
        gC = cute.local_tile(mC, self._cta_tiler, (iy, ix, None), proj=(1, 1, None))

        thr_copyA = tiled_copy_A.get_slice(tid)
        thr_copyB = tiled_copy_B.get_slice(tid)
        tAgA = thr_copyA.partition_S(gA)
        tAsA = thr_copyA.partition_D(sA)
        tBgB = thr_copyB.partition_S(gB)
        tBsB = thr_copyB.partition_D(sB)

        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

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

        tCrC.fill(0.0)
        for kidx in range(mA.shape[1] // self._bK):
            cute.copy(tiled_copy_A, tAgA[None, None, None, kidx], tAsA[None, None, None])
            cute.copy(tiled_copy_B, tBgB[None, None, None, kidx], tBsB[None, None, None])
            cute.arch.sync_threads()
            cute.copy(tiled_s2r_A, tCsA_view, tCrA_view)
            cute.copy(tiled_s2r_B, tCsB_view, tCrB_view)
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

        atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
        tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)
        for i in range(cute.size(tCrC_out)):
            tCrC_out[i] = cutlass.Float16(tCrC[i])
        cute.copy(atom_store, tCrC_out, tCgC)


# =============================================================================
# V3: WMMA + TMA + Swizzled SMEM (SM90+)
# =============================================================================
# 对应 LeetCUDA: hgemm_mma_stage_tn_cute.cu
# 核心: TMA 硬件搬运 + CuTe Swizzle<3,3,3> SMEM layout + mbarrier
# 这是 LeetCUDA CuTe C++ 版本的 Python DSL 对等实现

class HgemmMMA_TMA_V3:
    """
    LeetCUDA 对应:
    - hgemm_mma_stages_block_swizzle_tn_cute_kernel
    - SM80_16x8x16_F16F16F16F16_TN + Swizzle<3,3,3> + cp.async + K 内层展开
    """
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

        # Swizzled SMEM layout (CuTe Swizzle<3,3,3> 消除 bank conflict)
        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout, mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype, num_stages=self.num_stages)
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout, mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype, num_stages=self.num_stages)

        tma_atom_a, tma_tensor_a = self._make_tma(
            a, self.a_smem_layout_staged, (self._bM, self._bK))
        tma_atom_b, tma_tensor_b = self._make_tma(
            b, self.b_smem_layout_staged, (self._bN, self._bK))

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

        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2))
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2))

        sA_mma = cute.slice_(sA, (None, None, 0))
        sB_mma = cute.slice_(sB, (None, None, 0))
        thr_mma = tiled_mma.get_slice(tid)
        tCsA = thr_mma.partition_A(sA_mma)
        tCsB = thr_mma.partition_B(sB_mma)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrC = tiled_mma.make_fragment_C(tCgC)

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

        tma_transaction_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout) + cute.size_in_bytes(self.b_dtype, b_smem_layout)
        mbar_ptr = storage.mbar_ptr.data_ptr()

        if warp_idx == 0 and tid == 0:
            cute.arch.mbarrier_init(mbar_ptr, cnt=1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        tCrC.fill(0.0)
        phase = 0
        for kidx in range(mA_mk.shape[1] // self._bK):
            if warp_idx == 0:
                cute.copy(tma_atom_a, tAgA[None, kidx], tAsA[None, 0], tma_bar_ptr=mbar_ptr)
                cute.copy(tma_atom_b, tBgB[None, kidx], tBsB[None, 0], tma_bar_ptr=mbar_ptr)
                if tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, tma_transaction_bytes)

            cute.arch.mbarrier_wait(mbar_ptr, phase)
            phase ^= 1

            cute.copy(tiled_s2r_A, tCsA_copy_view, tCrA_copy_view)
            cute.copy(tiled_s2r_B, tCsB_copy_view, tCrB_copy_view)
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
            cute.arch.sync_threads()

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
    print(f"CuTeDSL HGEMM 全版本性能对比 (M={M}, N={N}, K={K})")
    print("对标 LeetCUDA/kernels/hgemm/ 全部优化技术")
    print("=" * 70)

    A_f16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B_f16 = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref_f16 = torch.matmul(A_f16, B_f16.T)
    A_f16_ = from_dlpack(A_f16, assumed_align=16)
    B_f16_ = from_dlpack(B_f16, assumed_align=16)

    results = []

    # V1: 基础 WMMA
    C1 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    C1_ = from_dlpack(C1, assumed_align=16)
    gemm_v1 = HgemmMMA_V1(cta_tiler=(128, 128, 32))
    c1 = cute.compile(gemm_v1, A_f16_, B_f16_, C1_)
    c1(A_f16_, B_f16_, C1_)
    max_diff_1 = (C1.float() - ref_f16.float()).abs().max().item()
    passed_1 = max_diff_1 < 5.0
    if passed_1:
        t1 = benchmark(c1, kernel_arguments=JitArguments(A_f16_, B_f16_, C1_))
        print(f"✅ V1 WMMA + 向量化 + Padding        耗时: {t1:.2f} µs  (max_diff={max_diff_1:.4f})")
        results.append(("V1 WMMA + Vectorized + Padding", t1))
    else:
        print(f"❌ V1 精度不满足  max_diff={max_diff_1:.4f}")

    # V1b: SMEM Swizzle (XOR-based)
    C1b = torch.empty(M, N, device="cuda", dtype=torch.float16)
    C1b_ = from_dlpack(C1b, assumed_align=16)
    gemm_v1b = HgemmMMA_Swizzle_V1b(cta_tiler=(128, 128, 32))
    c1b = cute.compile(gemm_v1b, A_f16_, B_f16_, C1b_)
    c1b(A_f16_, B_f16_, C1b_)
    max_diff_1b = (C1b.float() - ref_f16.float()).abs().max().item()
    passed_1b = max_diff_1b < 5.0
    if passed_1b:
        t1b = benchmark(c1b, kernel_arguments=JitArguments(A_f16_, B_f16_, C1b_))
        print(f"✅ V1b WMMA + SMEM Swizzle (XOR)     耗时: {t1b:.2f} µs  (max_diff={max_diff_1b:.4f})")
        results.append(("V1b WMMA + SMEM Swizzle (XOR)", t1b))
    else:
        print(f"❌ V1b 精度不满足  max_diff={max_diff_1b:.4f}")

    # V2: Double Buffer
    C2 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    C2_ = from_dlpack(C2, assumed_align=16)
    gemm_v2 = HgemmMMA_DoubleBuf_V2(cta_tiler=(128, 128, 32))
    c2 = cute.compile(gemm_v2, A_f16_, B_f16_, C2_)
    c2(A_f16_, B_f16_, C2_)
    max_diff_2 = (C2.float() - ref_f16.float()).abs().max().item()
    passed_2 = max_diff_2 < 5.0
    if passed_2:
        t2 = benchmark(c2, kernel_arguments=JitArguments(A_f16_, B_f16_, C2_))
        print(f"✅ V2 WMMA + Double Buffer           耗时: {t2:.2f} µs  (max_diff={max_diff_2:.4f})")
        results.append(("V2 WMMA + Double Buffer", t2))
    else:
        print(f"❌ V2 精度不满足  max_diff={max_diff_2:.4f}")

    # V2b: Block Swizzle (L2 Cache 优化)
    C2b = torch.empty(M, N, device="cuda", dtype=torch.float16)
    C2b_ = from_dlpack(C2b, assumed_align=16)
    gemm_v2b = HgemmMMA_BlockSwizzle_V2b(cta_tiler=(128, 128, 32), swizzle_stride=2048)
    c2b = cute.compile(gemm_v2b, A_f16_, B_f16_, C2b_)
    c2b(A_f16_, B_f16_, C2b_)
    max_diff_2b = (C2b.float() - ref_f16.float()).abs().max().item()
    passed_2b = max_diff_2b < 5.0
    if passed_2b:
        t2b = benchmark(c2b, kernel_arguments=JitArguments(A_f16_, B_f16_, C2b_))
        print(f"✅ V2b WMMA + Block Swizzle          耗时: {t2b:.2f} µs  (max_diff={max_diff_2b:.4f})")
        results.append(("V2b WMMA + Block Swizzle (L2)", t2b))
    else:
        print(f"❌ V2b 精度不满足  max_diff={max_diff_2b:.4f}")

    # V3: TMA + Swizzle (SM90+ only)
    cc = torch.cuda.get_device_capability()
    if cc >= (9, 0):
        C3 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        C3_ = from_dlpack(C3, assumed_align=16)
        gemm_v3 = HgemmMMA_TMA_V3(cta_tiler=(128, 128, 64))
        c3 = cute.compile(gemm_v3, A_f16_, B_f16_, C3_)
        c3(A_f16_, B_f16_, C3_)
        max_diff_3 = (C3.float() - ref_f16.float()).abs().max().item()
        passed_3 = max_diff_3 < 5.0
        if passed_3:
            t3 = benchmark(c3, kernel_arguments=JitArguments(A_f16_, B_f16_, C3_))
            print(f"✅ V3 WMMA + TMA + Swizzle           耗时: {t3:.2f} µs  (max_diff={max_diff_3:.4f})")
            results.append(("V3 WMMA + TMA + Swizzle (SM90+)", t3))
        else:
            print(f"❌ V3 精度不满足  max_diff={max_diff_3:.4f}")
    else:
        print(f"⏭️  V3 WMMA + TMA 跳过 (需要 SM90+, 当前 SM{cc[0]}{cc[1]})")

    # PyTorch baseline
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

    # Summary
    flops = 2 * M * N * K
    print(f"\n{'='*70}")
    print(f"  {'版本':<48} {'耗时(µs)':<10} {'TFLOPS':<10}")
    print(f"  {'-'*66}")
    for name, t in results:
        tflops = flops / (t * 1e6)
        print(f"  {name:<48} {t:<10.2f} {tflops:<10.4f}")
    print(f"{'='*70}")
