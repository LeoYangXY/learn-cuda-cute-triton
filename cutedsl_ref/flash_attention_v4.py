"""
=============================================================================
CuTeDSL Flash Attention V2 —— 对照 Dao-AILab FlashAttention-4 官方实现
=============================================================================

本文件是 FlashAttention-4 (Dao-AILab/flash-attention) 的 SM80/SM120 前向 kernel
的简化独立版本。

FA4 官方代码结构:
  flash_attn/cute/flash_fwd.py          → FlashAttentionForwardBase + FlashAttentionForwardSm80
  flash_attn/cute/ampere_helpers.py     → get_smem_layout_atom() + gemm() + gemm_rs()
  flash_attn/cute/softmax.py            → Softmax.online_softmax() + finalize() + rescale_O()
  flash_attn/cute/flash_fwd_sm120.py    → 直接复用 SM80 的实现（SM120 不支持 WGMMA）

核心技术栈（SM80/SM120 路径，全部在教程中讲过）:
  1. cp.async (CopyG2SOp)               → GMEM→SMEM 异步 (教程 10, data_movement 方法2)
  2. ldmatrix (LdMatrix8x8x16bOp)       → SMEM→Register (教程 05, data_movement 方法3)
  3. WMMA (MmaF16BF16Op)                → Tensor Core 矩阵乘法 (教程 05)
  4. Swizzle                            → 消除 SMEM bank conflict (教程 12)
  5. make_tiled_copy_A/B + retile       → 从 tiled_mma 自动推导 Copy (教程 05)
  6. Online Softmax                     → Flash Attention 核心算法 (教程 08)

与 CUTLASS 官方 example (ampere/flash_attention_v2.py) 对比：
  FA4 新增:
  - 更智能的 smem layout atom (根据 head_dim 自动选择 swizzle)
  - gemm 函数封装 (ldmatrix 预加载流水线)
  - gemm_rs (寄存器侧 A, S→P×V)
  - Softmax 用 reshape_acc_to_mn + load()/store() SSA 风格
  - 支持 causal mask、变长序列等

RTX 5050 (SM120) 说明：
  SM120 是消费级 Blackwell，不支持 WGMMA/tcgen05 UMMA，
  所以 FA4 在 SM120 上走的和 SM80 完全一样的代码路径：
  WMMA + cp.async + ldmatrix。
=============================================================================
"""

import math
from types import SimpleNamespace
from typing import Type, Callable, Optional

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import cutlass.pipeline as pipeline
import cutlass.utils as utils


# =============================================================================
# 辅助函数：对照 FA4 的 ampere_helpers.py
# =============================================================================

def get_smem_layout_atom(dtype: Type[cutlass.Numeric], k_dim: int):
    """
    对照 FA4: flash_attn/cute/ampere_helpers.py::get_smem_layout_atom()

    根据 head_dim 自动选择最优的 swizzle 配置。
    FA4 比 CUTLASS example 更智能:
      - CUTLASS example 固定用 smem_k_block_size=64 或 32
      - FA4 根据 k_dim 选择 128/64/32/16

    这个函数生成的 layout atom 用于 sQ/sK/sV/sO 的 SMEM 布局。
    """
    dtype_byte = cutlass.const_expr(dtype.width // 8)
    bytes_per_row = cutlass.const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        cutlass.const_expr(
            128 if bytes_per_row % 128 == 0
            else (64 if bytes_per_row % 64 == 0
            else (32 if bytes_per_row % 32 == 0 else 16))
        ) // dtype_byte
    )
    swizzle_bits = (
        4 if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64
        else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if cutlass.const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


# =============================================================================
# GEMM 辅助函数: 对照 FA4 的 ampere_helpers.py::gemm() / gemm_rs()
# =============================================================================

@cute.jit
def ampere_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
) -> None:
    """
    对照 FA4: ampere_helpers.py::gemm()

    封装 ldmatrix 预加载 + WMMA 计算的流水线。
    相比直接调用 cute.gemm，这里在 K 循环中:
      1. 先预加载下一个 k-block 到寄存器 (ldmatrix)
      2. 再执行当前 k-block 的 WMMA
    从而实现 copy-compute 重叠。
    """
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

    # 预加载第 0 个 k-block
    cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])

    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        # 预加载下一个 k-block (流水线)
        if k < cute.size(tCsA.shape[2]) - 1:
            cute.copy(smem_thr_copy_A, tCsA[None, None, k + 1], tCrA_copy_view[None, None, k + 1])
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])
        # 执行 WMMA
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def ampere_gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
) -> None:
    """
    对照 FA4: ampere_helpers.py::gemm_rs()

    A 已经在寄存器中 (rP = softmax 结果)，只需要从 SMEM 加载 B (V^T)。
    用于第二个 GEMM: O = P × V。
    """
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)

    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])

    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if cutlass.const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(smem_thr_copy_B, tCsB[None, None, k + 1], tCrB_copy_view[None, None, k + 1])
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


# =============================================================================
# Layout 辅助函数: 对照 FA4 的 layout_utils
# =============================================================================

def reshape_acc_to_mn(acc):
    """
    对照 FA4: quack/layout_utils.py::reshape_acc_to_mn()

    将 MMA 累加器的 (atom_vals, MMA_M, MMA_N) 布局
    重排为 (M, N) 布局，方便逐行处理 softmax。
    """
    acc_layout_col_major = cute.make_layout(acc.layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
            (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
        ),
        stride=(
            (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
            (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
        ),
    )
    acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
    return cute.make_tensor(acc.iterator, acc_layout_mn)


def reshape_acc_to_frgA(rP):
    """
    对照 FA4: quack/layout_utils.py::reshape_acc_to_frgA()

    将 softmax 的输出 rP (S 的 accumulator layout) 转换为
    第二个 GEMM (O = P × V) 的 A 操作数 layout。

    因为 MMA 指令形状是 16×8×16，需要从
    (4, MMA_M, MMA_N) 转换为 ((4, 2), MMA_M, MMA_N/2)
    """
    rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
    rP_mma_view = cute.make_layout(
        (
            (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
            rP_layout_divided.shape[1],
            rP_layout_divided.shape[2][1],
        ),
        stride=(
            (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
            rP_layout_divided.stride[1],
            rP_layout_divided.stride[2][1],
        ),
    )
    return cute.make_tensor(rP.iterator, rP_mma_view)


# =============================================================================
# 主类: 对照 FA4 的 FlashAttentionForwardSm80
# =============================================================================

class FlashAttentionFA4:
    """
    对照 FA4: flash_attn/cute/flash_fwd.py::FlashAttentionForwardSm80

    Flash Attention V2 前向传播，使用和 FA4 SM80/SM120 路径完全相同的技术栈。
    """

    def __init__(
        self,
        head_dim: int,
        tile_m: int = 128,
        tile_n: int = 128,
        num_threads: int = 128,
        is_causal: bool = False,
    ):
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.tile_n = tile_n
        # FA4: padding head_dim to a multiple of 16
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.num_threads = num_threads
        self.is_causal = is_causal

        self.cta_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: Float32,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type == mO.element_type)
        ):
            raise TypeError("All tensors must have the same data type")
        self.dtype: Type[cutlass.Numeric] = mQ.element_type

        # ============================================================
        # SMEM Layout: 对照 FA4 的 get_smem_layout_atom()
        # ============================================================
        sQ_layout_atom = get_smem_layout_atom(self.dtype, self.tile_hdim)
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = sQ_layout_atom

        sQ_layout = cute.tile_to_shape(sQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1))
        sK_layout = cute.tile_to_shape(sK_layout_atom, (self.tile_n, self.tile_hdim), (0, 1))
        sV_layout = cute.tile_to_shape(sV_layout_atom, (self.tile_n, self.tile_hdim), (0, 1))
        sO_layout = sQ_layout

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sK_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(sV_layout)], 1024]

        # ============================================================
        # GMEM Tiled Copy: cp.async 128-bit
        # ============================================================
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype, num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_ordered_layout(
            (self.num_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))

        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(atom_async_copy, tQKV_layout, vQKV_layout)
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tQKV_layout, vQKV_layout)

        # ============================================================
        # Tiled MMA: 对照 FA4 的 _get_tiled_mma()
        # ============================================================
        # FA4 为 QK 和 PV 分别创建 tiled_mma (虽然 SM80 上配置一样)
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, 16),
        )

        # grid: (m_blocks, batch, num_head)
        grid_dim = (
            cute.ceil_div(mQ.shape[1], self.tile_m),
            cute.size(mQ.shape[0]),
            cute.size(mQ.shape[2]),
        )

        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E

        self.kernel(
            mQ, mK, mV, mO, softmax_scale_log2,
            sQ_layout, sK_layout, sV_layout, sO_layout,
            gmem_tiled_copy_QKV, gmem_tiled_copy_O,
            tiled_mma_qk, tiled_mma_pv, SharedStorage,
        ).launch(grid=grid_dim, block=[self.num_threads, 1, 1])

    # ==================================================================
    # Kernel 主函数: 对照 FA4 FlashAttentionForwardSm80.kernel()
    # ==================================================================
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor,
        softmax_scale_log2: Float32,
        sQ_layout: cute.ComposedLayout, sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout, sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_QKV: cute.TiledCopy, gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma, tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_size, num_head = cute.arch.block_idx()

        n_block_max = cute.ceil_div(mK.shape[1], self.tile_n)
        if self.is_causal:
            n_block_max = min(
                cute.ceil_div((m_block + 1) * self.tile_m, self.tile_n),
                n_block_max,
            )
        n_block = cutlass.max(n_block_max - 1, 0)

        # CTA 级分块
        gQ = cute.local_tile(
            mQ[batch_size, None, num_head, None],
            (self.tile_m, self.tile_hdim), (m_block, 0),
        )
        gK = cute.local_tile(
            mK[batch_size, None, num_head, None],
            (self.tile_n, self.tile_hdim), (None, 0),
        )
        gV = cute.local_tile(
            mV[batch_size, None, num_head, None],
            (self.tile_n, self.tile_hdim), (None, 0),
        )

        # SMEM 分配
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)

        # V 转置视图 (对照 FA4 的 layout_utils.transpose_view)
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self.tile_hdim, self.tile_n), stride=(self.tile_n, 1),
            ),
        )

        # GMEM→SMEM copy 分区
        gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
        tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
        tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
        tKsK = gmem_thr_copy_QKV.partition_D(sK)
        tKgK = gmem_thr_copy_QKV.partition_S(gK)
        tVsV = gmem_thr_copy_QKV.partition_D(sV)
        tVgV = gmem_thr_copy_QKV.partition_S(gV)

        # MMA 分区 + Fragment (对照 FA4 kernel)
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt))

        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdim))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        # SMEM→Register copy (ldmatrix): 对照 FA4
        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self.dtype,
        )
        smem_thr_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk).get_slice(tidx)
        smem_thr_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma_pv).get_slice(tidx)

        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        # Predicates (head_dim 越界保护)
        cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        cKV = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
        t0QcQ = gmem_tiled_copy_QKV.get_slice(0).partition_S(cQ)
        tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
        t0KVcKV = gmem_tiled_copy_QKV.get_slice(0).partition_S(cKV)

        tQpQ = self._predicate_k(tQcQ, mQ.shape[3])
        tKVpKV = self._predicate_k(tKVcKV, mK.shape[3])

        # ============================================================
        # Prologue: 加载 Q + 第一个 K tile
        # ============================================================
        # 对照 FA4: self.load_Q(...)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if t0QcQ[0, m, 0][0] < mQ.shape[1] - m_block * self.tile_m - tQcQ[0][0]:
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tQgQ[None, m, None], tQsQ[None, m, None],
                    pred=tQpQ[None, m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                )
        cute.arch.cp_async_commit_group()

        # 对照 FA4: self.load_K(...)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if t0KVcKV[0, n, 0][0] < mK.shape[1] - n_block * self.tile_n - tKVcKV[0][0]:
                cute.copy(
                    gmem_tiled_copy_QKV,
                    tKgK[None, n, None, n_block], tKsK[None, n, None],
                    pred=tKVpKV[None, n, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                )
        cute.arch.cp_async_commit_group()

        # ============================================================
        # Softmax state: 对照 FA4 的 Softmax.create()
        # ============================================================
        num_rows = acc_O.shape[0][0] * acc_O.shape[1]
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        row_max.fill(-Float32.inf)
        row_sum.fill(0.0)

        # ============================================================
        # Mainloop: 对照 FA4 的三段循环
        # ============================================================
        # 对照 FA4: 先确定需要 causal mask 的 block 范围
        # n_block_min_causal: causal mask 以下的 block 不需要 mask
        if cutlass.const_expr(self.is_causal):
            # 对于 causal，需要 mask 的 block 范围是 [n_block_min_causal, n_block_max)
            # n_block_min_causal = m_block * tile_m / tile_n
            n_block_min_causal = (m_block * self.tile_m) // self.tile_n
        else:
            n_block_min_causal = 0

        # 段1: 第一个 block (带 seqlen + causal mask)
        self._compute_one_n_block(
            n_block, True, True,
            mQ, mK, tidx, m_block, batch_size, num_head,
            thr_mma_qk, thr_mma_pv, tiled_mma_qk, tiled_mma_pv,
            tSrQ, tSrK, tOrVt, acc_O,
            smem_thr_copy_Q, smem_thr_copy_K, smem_thr_copy_V,
            tSsQ, tSsK, tOsVt,
            gmem_tiled_copy_QKV, tKgK, tKsK, tVgV, tVsV,
            tKVcKV, t0KVcKV, tKVpKV,
            row_max, row_sum, softmax_scale_log2,
        )

        if cutlass.const_expr(self.is_causal):
            # 段2: causal mask 区域内的后续 block
            for n_tile in range(1, n_block_max - n_block_min_causal, 1):
                cur_n_block = n_block_max - n_tile - 1
                self._compute_one_n_block(
                    cur_n_block, False, True,
                    mQ, mK, tidx, m_block, batch_size, num_head,
                    thr_mma_qk, thr_mma_pv, tiled_mma_qk, tiled_mma_pv,
                    tSrQ, tSrK, tOrVt, acc_O,
                    smem_thr_copy_Q, smem_thr_copy_K, smem_thr_copy_V,
                    tSsQ, tSsK, tOsVt,
                    gmem_tiled_copy_QKV, tKgK, tKsK, tVgV, tVsV,
                    tKVcKV, t0KVcKV, tKVpKV,
                    row_max, row_sum, softmax_scale_log2,
                )

            # 段3: 无 mask 区域
            for n_tile in range(n_block_min_causal):
                cur_n_block = n_block_min_causal - n_tile - 1
                self._compute_one_n_block(
                    cur_n_block, False, False,
                    mQ, mK, tidx, m_block, batch_size, num_head,
                    thr_mma_qk, thr_mma_pv, tiled_mma_qk, tiled_mma_pv,
                    tSrQ, tSrK, tOrVt, acc_O,
                    smem_thr_copy_Q, smem_thr_copy_K, smem_thr_copy_V,
                    tSsQ, tSsK, tOsVt,
                    gmem_tiled_copy_QKV, tKgK, tKsK, tVgV, tVsV,
                    tKVcKV, t0KVcKV, tKVpKV,
                    row_max, row_sum, softmax_scale_log2,
                )
        else:
            # non-causal: 后续 block 都无 mask
            for n_tile in range(1, n_block_max, 1):
                cur_n_block = n_block_max - n_tile - 1
                self._compute_one_n_block(
                    cur_n_block, False, False,
                    mQ, mK, tidx, m_block, batch_size, num_head,
                    thr_mma_qk, thr_mma_pv, tiled_mma_qk, tiled_mma_pv,
                    tSrQ, tSrK, tOrVt, acc_O,
                    smem_thr_copy_Q, smem_thr_copy_K, smem_thr_copy_V,
                    tSsQ, tSsK, tOsVt,
                    gmem_tiled_copy_QKV, tKgK, tKsK, tVgV, tVsV,
                    tKVcKV, t0KVcKV, tKVpKV,
                    row_max, row_sum, softmax_scale_log2,
                )

        # ============================================================
        # Finalize: 对照 FA4 的 softmax.finalize() + rescale_O()
        # ============================================================
        # quad reduction for row_sum
        acc_O_mn = reshape_acc_to_mn(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            is_zero_or_nan = (row_sum[r] == 0.0 or row_sum[r] != row_sum[r])
            scale = 1.0 if is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

        # ============================================================
        # Epilogue: 对照 FA4 的 epilogue()
        # ============================================================
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))

        sO = cute.make_tensor(sQ.iterator, sO_layout)
        smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.dtype)
        smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv)
        smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        self.cta_barrier.arrive_and_wait()

        gO = cute.local_tile(
            mO[batch_size, None, num_head, None],
            (self.tile_m, self.tile_hdim), (m_block, 0),
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOgO = gmem_thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, self.dtype)

        cute.autovec_copy(tOsO, tOrO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tOcO = gmem_thr_copy_O.partition_D(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_D(cO)
        tOpO = self._predicate_k(gmem_thr_copy_O.partition_S(cO), mO.shape[3])

        for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            if t0OcO[0, rest_m, 0][0] < mO.shape[1] - m_block * self.tile_m - tOcO[0][0]:
                cute.copy(
                    gmem_tiled_copy_O,
                    tOrO[None, rest_m, None], tOgO[None, rest_m, None],
                    pred=tOpO[None, rest_m, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                )

    # ==================================================================
    # compute_one_n_block: 对照 FA4 的同名函数
    # ==================================================================
    @cute.jit
    def _compute_one_n_block(
        self,
        n_block: Int32,
        is_first_n_block: cutlass.Constexpr,
        need_seqlen_mask: cutlass.Constexpr,
        mQ, mK, tidx, m_block, batch_size, num_head,
        thr_mma_qk, thr_mma_pv, tiled_mma_qk, tiled_mma_pv,
        tSrQ, tSrK, tOrVt, acc_O,
        smem_thr_copy_Q, smem_thr_copy_K, smem_thr_copy_V,
        tSsQ, tSsK, tOsVt,
        gmem_tiled_copy_QKV, tKgK, tKsK, tVgV, tVsV,
        tKVcKV, t0KVcKV, tKVpKV,
        row_max, row_sum, softmax_scale_log2,
    ):
        # 分配 S 累加器
        acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)

        # 等待 Q + K 的 cp.async 完成
        cute.arch.cp_async_wait_group(0)
        self.cta_barrier.arrive_and_wait()

        # 预加载 V tile
        if is_first_n_block:
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                if t0KVcKV[0, n, 0][0] < mK.shape[1] - n_block * self.tile_n - tKVcKV[0][0]:
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tVgV[None, n, None, n_block], tVsV[None, n, None],
                        pred=tKVpKV[None, n, None] if cutlass.const_expr(self.check_hdim_oob) else None,
                    )
        else:
            cute.copy(
                gmem_tiled_copy_QKV,
                tVgV[None, None, None, n_block], tVsV,
                pred=tKVpKV if cutlass.const_expr(self.check_hdim_oob) else None,
            )
        cute.arch.cp_async_commit_group()

        # ---- 第一个 GEMM: S = Q × K^T ----
        # 对照 FA4: ampere_helpers.gemm(...)
        ampere_gemm(
            tiled_mma_qk, acc_S, tSrQ, tSrK,
            tSsQ, tSsK, smem_thr_copy_Q, smem_thr_copy_K,
        )

        # 等 V 加载完
        cute.arch.cp_async_wait_group(0)
        self.cta_barrier.arrive_and_wait()

        # 预加载下一个 K tile
        if n_block > 0:
            cute.copy(
                gmem_tiled_copy_QKV,
                tKgK[None, None, None, n_block - 1], tKsK,
                pred=tKVpKV if cutlass.const_expr(self.check_hdim_oob) else None,
            )
            cute.arch.cp_async_commit_group()

        # ---- Causal Mask ----
        if cutlass.const_expr(self.is_causal and need_seqlen_mask):
            mcS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), mcS)
            tScS = thr_mma_qk.partition_C(cS)
            tScS_mn = reshape_acc_to_mn(tScS)
            acc_S_mn_mask = reshape_acc_to_mn(acc_S)
            for r in cutlass.range_constexpr(cute.size(acc_S_mn_mask.shape[0])):
                col_idx_limit = cutlass.min(tScS_mn[r, 0][0] + 1, mK.shape[1])
                for c in cutlass.range_constexpr(cute.size(acc_S_mn_mask.shape[1])):
                    if cute.elem_less(col_idx_limit, tScS_mn[0, c][1] + 1):
                        acc_S_mn_mask[r, c] = -Float32.inf

        # ---- Padding Mask (seqlen 边界) ----
        if cutlass.const_expr(need_seqlen_mask and not self.is_causal):
            mcS = cute.make_identity_tensor((self.tile_m, self.tile_n))
            cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), mcS)
            tScS = thr_mma_qk.partition_C(cS)
            tScS_mn = reshape_acc_to_mn(tScS)
            acc_S_mn_mask = reshape_acc_to_mn(acc_S)
            for r in cutlass.range_constexpr(cute.size(acc_S_mn_mask.shape[0])):
                for c in cutlass.range_constexpr(cute.size(acc_S_mn_mask.shape[1])):
                    if cute.elem_less(mK.shape[1], tScS_mn[0, c][1] + 1):
                        acc_S_mn_mask[r, c] = -Float32.inf

        # ---- Online Softmax: 对照 FA4 的 softmax.online_softmax() ----
        acc_S_mn = reshape_acc_to_mn(acc_S)
        acc_O_mn = reshape_acc_to_mn(acc_O)
        row_scale = cute.make_fragment_like(row_max, Float32)

        for r in cutlass.range_constexpr(cute.size(row_max)):
            acc_S_row = acc_S_mn[r, None].load()

            # row max (对照 FA4: utils.fmax_reduce + warp_reduction_max)
            row_max_cur = acc_S_row.reduce(
                cute.ReductionOp.MAX, -Float32.inf, 0
            )
            row_max_cur = self._threadquad_reduce_max(row_max_cur)

            if cutlass.const_expr(not is_first_n_block):
                row_max_cur = cute.arch.fmax(row_max[r], row_max_cur)

            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if cutlass.const_expr(self.is_causal):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            # exp2 (对照 FA4: cute.math.exp2(..., fastmath=True))
            row_max_cur_scaled = row_max_cur * softmax_scale_log2
            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_scale_log2 - row_max_cur_scaled, fastmath=True,
            )

            if cutlass.const_expr(is_first_n_block):
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
                row_scale[r] = 1.0
            else:
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * softmax_scale_log2, fastmath=True,
                )
                acc_S_row_sum = acc_S_row_exp.reduce(
                    cute.ReductionOp.ADD, Float32.zero, 0
                )
                acc_S_row_sum = acc_S_row_sum + row_sum[r] * row_scale[r]

            row_sum[r] = acc_S_row_sum
            acc_S_mn[r, None] = acc_S_row_exp

        # rescale O (对照 FA4: softmax.rescale_O())
        for r in cutlass.range_constexpr(cute.size(row_scale)):
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * row_scale[r]

        # ---- 第二个 GEMM: O += P × V ----
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = reshape_acc_to_frgA(rP)

        # 对照 FA4: ampere_helpers.gemm_rs(...)
        ampere_gemm_rs(
            tiled_mma_pv, acc_O, tOrP, tOrVt,
            tOsVt, smem_thr_copy_V,
        )

    # ==================================================================
    # 辅助函数
    # ==================================================================
    @cute.jit
    def _predicate_k(self, tCcT, limit):
        """对照 FA4: utils.predicate_k() — 生成 head_dim 方向的 predicate"""
        tpT = cute.make_rmem_tensor(
            cute.make_layout(
                (tCcT.shape[0][1], cute.size(tCcT, mode=[1]), cute.size(tCcT, mode=[2])),
                stride=(cute.size(tCcT, mode=[2]), 0, 1),
            ),
            cutlass.Boolean,
        )
        for rest_v in cutlass.range_constexpr(tpT.shape[0]):
            for rest_k in cutlass.range_constexpr(tpT.shape[2]):
                tpT[rest_v, 0, rest_k] = cute.elem_less(
                    tCcT[(0, rest_v), 0, rest_k][1], limit
                )
        return tpT

    def _threadquad_reduce(self, val, op):
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)


# =============================================================================
# 测试 + 性能对比
# =============================================================================

def create_tensor(batch_size, seqlen, num_head, head_dim, dtype_cutlass):
    import cutlass.torch as cutlass_torch
    torch_dtype = cutlass_torch.dtype(dtype_cutlass)
    shape = (batch_size, seqlen, num_head, head_dim)
    t = torch.empty(*shape, dtype=torch.int32).random_(-2, 2).to(dtype=torch_dtype).cuda()
    ct = (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=3)
        .mark_compact_shape_dynamic(
            mode=3, stride_order=t.dim_order(), divisibility=(128 // dtype_cutlass.width),
        )
    )
    return ct, t


def main():
    print("=" * 70)
    print("Flash Attention V2 —— 对照 Dao-AILab FA4 官方 CuTeDSL 实现")
    print("SM80/SM120 路径: WMMA + cp.async + ldmatrix + Online Softmax")
    print("=" * 70)

    cap = torch.cuda.get_device_capability(0)
    sm = cap[0] * 10 + cap[1]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM:  sm_{sm}0")
    print()

    dtype = cutlass.Float16
    batch_size = 4
    seqlen_q = 1024
    seqlen_k = 1024
    num_head = 16
    head_dim = 128
    softmax_scale = 1.0 / math.sqrt(head_dim)
    tile_m = 128
    tile_n = 64
    num_threads = 128
    is_causal = False

    print(f"参数: B={batch_size}, Sq={seqlen_q}, Sk={seqlen_k}, H={num_head}, D={head_dim}")
    print(f"Tile: M={tile_m}, N={tile_n}, threads={num_threads}")
    print()

    q, q_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
    k, k_torch = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
    v, v_torch = create_tensor(batch_size, seqlen_k, num_head, head_dim, dtype)
    o, o_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)

    fa4 = FlashAttentionFA4(head_dim, tile_m, tile_n, num_threads, is_causal)

    print("编译 kernel...")
    compiled = cute.compile(fa4, q, k, v, o, softmax_scale)
    print("运行...")
    compiled(q, k, v, o, softmax_scale)
    torch.cuda.synchronize()

    # 正确性验证
    q_ref = q_torch.permute(0, 2, 1, 3).float()
    k_ref = k_torch.permute(0, 2, 1, 3).float()
    v_ref = v_torch.permute(0, 2, 1, 3).float()
    ref_o = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, scale=softmax_scale, is_causal=is_causal
    ).permute(0, 2, 1, 3).to(q_torch.dtype)

    max_diff = (o_torch - ref_o).abs().max().item()
    mean_diff = (o_torch - ref_o).abs().mean().item()
    print(f"\n正确性: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"  {'PASS' if max_diff < 0.02 else 'FAIL'}")

    # 性能
    print("\n" + "-" * 70)
    flops = 4 * batch_size * num_head * seqlen_q * seqlen_k * head_dim

    time_us = benchmark(compiled, kernel_arguments=JitArguments(q, k, v, o, softmax_scale))
    tflops = flops / (time_us * 1e6)
    print(f"  FA4-style CuTeDSL: {time_us:.2f} us | {tflops:.4f} TFLOPS")

    # PyTorch SDPA
    q_pt = q_torch.permute(0, 2, 1, 3)
    k_pt = k_torch.permute(0, 2, 1, 3)
    v_pt = v_torch.permute(0, 2, 1, 3)
    for _ in range(10):
        torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, scale=softmax_scale)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt, scale=softmax_scale)
    end.record()
    torch.cuda.synchronize()
    pt_time_us = start.elapsed_time(end) * 1000 / 100
    pt_tflops = flops / (pt_time_us * 1e6)
    print(f"  PyTorch SDPA:      {pt_time_us:.2f} us | {pt_tflops:.4f} TFLOPS")

    ratio = pt_time_us / time_us
    print(f"\n  速度比: {ratio:.2f}x {'(更快)' if ratio > 1 else '(更慢)'}")

    # Causal
    print("\n" + "=" * 70)
    print("Causal Attention")
    print("=" * 70)
    o_c, o_c_torch = create_tensor(batch_size, seqlen_q, num_head, head_dim, dtype)
    fa4_c = FlashAttentionFA4(head_dim, tile_m, tile_n, num_threads, is_causal=True)
    compiled_c = cute.compile(fa4_c, q, k, v, o_c, softmax_scale)
    compiled_c(q, k, v, o_c, softmax_scale)
    torch.cuda.synchronize()

    ref_c = torch.nn.functional.scaled_dot_product_attention(
        q_ref, k_ref, v_ref, scale=softmax_scale, is_causal=True
    ).permute(0, 2, 1, 3).to(q_torch.dtype)
    max_diff_c = (o_c_torch - ref_c).abs().max().item()
    print(f"  Causal max_diff: {max_diff_c:.6f} {'PASS' if max_diff_c < 0.02 else 'FAIL'}")

    t_c = benchmark(compiled_c, kernel_arguments=JitArguments(q, k, v, o_c, softmax_scale))
    print(f"  Causal: {t_c:.2f} us | {flops / (t_c * 1e6):.4f} TFLOPS")

    # FA4 对照表
    print("\n" + "=" * 70)
    print("FA4 官方代码对照")
    print("=" * 70)
    print("  本文件函数                    → FA4 官方文件")
    print("  " + "-" * 60)
    print("  get_smem_layout_atom()       → ampere_helpers.py::get_smem_layout_atom()")
    print("  ampere_gemm()                → ampere_helpers.py::gemm()")
    print("  ampere_gemm_rs()             → ampere_helpers.py::gemm_rs()")
    print("  reshape_acc_to_mn()          → quack/layout_utils.py::reshape_acc_to_mn()")
    print("  reshape_acc_to_frgA()        → quack/layout_utils.py::reshape_acc_to_frgA()")
    print("  FlashAttentionFA4.__call__() → flash_fwd.py::FlashAttentionForwardSm80.__call__()")
    print("  FlashAttentionFA4.kernel()   → flash_fwd.py::FlashAttentionForwardSm80.kernel()")
    print("  _compute_one_n_block()       → flash_fwd.py::compute_one_n_block()")
    print("  online_softmax 部分          → softmax.py::Softmax.online_softmax()")
    print("  finalize 部分                → softmax.py::Softmax.finalize()")
    print("  _predicate_k()               → flash_attn/cute/utils.py::predicate_k()")


if __name__ == "__main__":
    main()
