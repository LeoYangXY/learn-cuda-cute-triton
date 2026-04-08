"""
=============================================================================
教程 06: Hopper 架构 —— TMA + WGMMA + Warp Specialization（概念讲解）
=============================================================================

⚠️ 注意：本教程的代码需要 SM90 (Hopper) 硬件才能运行。
你的 RTX 5050 是 SM120 (Blackwell)，不支持 SM90 的 WGMMA 指令。
Blackwell 使用 tcgen05 UMMA 替代了 WGMMA，见教程 07。

本文件作为概念讲解，展示 Hopper 架构的关键特性。
代码可以阅读学习，但不能在 SM120 上执行。
=============================================================================

Hopper (SM90) 引入了两个革命性的硬件特性：

1. TMA（Tensor Memory Accelerator）
   - 专用硬件单元，异步搬运 GMEM ↔ SMEM
   - 不占用线程资源（只需 1 个线程发起，硬件自动完成）
   - 支持 Swizzle（自动消除 bank conflict）
   - 通过 mbarrier 同步完成状态
   - ✅ Blackwell 继续支持 TMA

2. WGMMA（Warp Group MMA）
   - 4 个 Warp（128 线程）协作完成大矩阵乘法
   - 直接从 SMEM 读取操作数（不需要先加载到寄存器）
   - 异步执行：发射后不阻塞，通过 commit/wait 同步
   - ❌ Blackwell 用 tcgen05 UMMA 替代

3. Warp Specialization（Warp 特化）
   - 不同的 Warp 执行不同的任务
   - TMA Warp：专门负责数据搬运
   - MMA Warp：专门负责计算
   - 搬运和计算可以完全重叠
   - ✅ Blackwell 继续使用 Warp Specialization

数据流对比：
  Ampere:  GMEM → (线程搬运) → SMEM → (LdMatrix) → RMEM → (WMMA)  → RMEM
  Hopper:  GMEM → (TMA硬件)  → SMEM → (WGMMA直读) →        RMEM
  Blackwell: GMEM → (TMA硬件) → SMEM → (UMMA)     →        TMEM → RMEM

Swizzle（交织）：
  - SMEM 有 32 个 bank，每个 bank 4 字节宽
  - 如果同一 warp 的线程访问同一 bank → bank conflict → 串行化
  - Swizzle 通过 XOR 操作重新映射地址，消除 conflict
  - CuTeDSL 中用 make_swizzle(B, M, S) 或 tile_to_shape 自动处理

mbarrier（Memory Barrier）工作原理：
  - mbarrier 是 Hopper 引入的硬件同步原语
  - 支持"到达计数"和"期望字节数"两种同步模式
  - TMA 完成后自动到达 mbarrier
  - 消费者通过 mbarrier_wait 等待数据就绪
  - 使用 phase bit 实现多 stage 轮转

Pipeline 流水线模型：
  ┌─────────────────────────────────────────────────────┐
  │  TMA Warp:  [Load S0] [Load S1] [Load S2] [Load S3]│
  │  MMA Warps:          [Comp S0] [Comp S1] [Comp S2] │
  │                                                      │
  │  S0~S3 是 4 个 SMEM stage，轮流使用                   │
  │  TMA 和 MMA 通过 mbarrier 同步，实现重叠执行          │
  └─────────────────────────────────────────────────────┘
=============================================================================
"""

import torch
from typing import Tuple
import math

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils as utils
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.pipeline import (
    PipelineTmaAsync, CooperativeGroup, Agent,
    make_pipeline_state, PipelineUserType
)


class GemmHopper:
    """
    Hopper WGMMA GEMM：TMA 加载 + WGMMA 计算 + Warp 特化流水线

    架构：
      Warp 0-7 (threads 0-255):  MMA 计算（2 个 Warp Group）
      Warp 8   (threads 256-287): TMA 数据搬运

    流水线：4 个 stage，TMA 和 WGMMA 重叠执行
    """

    def __init__(self, cta_tiler=(128, 256, 64)):
        self.BM, self.BN, self.BK = cta_tiler
        self.tile_shape_mnk = cta_tiler

        # atom_layout: 控制用几个 Warp Group 做 MMA
        # BM>64 且 BN>64 时用 2 个 Warp Group
        self.atom_layout_mnk = (2, 1, 1) if self.BM > 64 and self.BN > 64 else (1, 1, 1)

        self.threads_per_warp = 32
        self.threads_per_warp_group = 128  # 4 warps = 1 warp group

        # Warp 特化：MMA warps + 1 个 TMA warp
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_mma_warps = self.num_mma_warp_groups * 4
        self.tma_warp_id = self.num_mma_warps
        self.threads_per_cta = self.threads_per_warp * (self.num_mma_warps + 1)

        self.num_stages = 4  # 流水线深度
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        # ====== SMEM Layout（带 Swizzle）======
        # sm90_utils 自动生成带 swizzle 的 SMEM layout
        # 返回 ComposedLayout = Swizzle ∘ Layout
        self.a_smem_layout = sm90_utils.make_smem_layout_a(
            a_layout=self.a_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            a_dtype=self.a_dtype,
            num_stages=self.num_stages
        )
        self.b_smem_layout = sm90_utils.make_smem_layout_b(
            b_layout=self.b_layout,
            mma_tiler_mnk=self.tile_shape_mnk,
            b_dtype=self.b_dtype,
            num_stages=self.num_stages
        )

        # ====== TMA 描述符 ======
        # TMA 需要一个"描述符"来告诉硬件：
        #   - 源数据在 GMEM 的什么位置
        #   - 目标在 SMEM 的什么 layout
        #   - 每次搬运多大的 tile
        tma_atom_a, tma_tensor_a = self._make_tma(a, self.a_smem_layout, (self.BM, self.BK))
        tma_atom_b, tma_tensor_b = self._make_tma(b, self.b_smem_layout, (self.BN, self.BK))

        # ====== WGMMA 操作 ======
        # make_trivial_tiled_mma: 创建 Hopper WGMMA 操作
        # 关键参数：
        #   - a_major_mode/b_major_mode: 操作数的主序（K-major 或 MN-major）
        #   - atom_layout_mnk: 用几个 warp group
        #   - tiler_mn: 每个 warp group 处理的 MN 大小
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.BN),
        )

        # ====== Shared Storage ======
        @cute.struct
        class SharedStorage:
            pipeline_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout)],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout)],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage

        grid_dim = *cute.ceil_div(c.shape, (self.BM, self.BN)), 1
        self.kernel(
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            self.tiled_mma, c,
            self.a_smem_layout, self.b_smem_layout,
        ).launch(grid=grid_dim, block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(
        self,
        tma_atom_a, mA_tma, tma_atom_b, mB_tma,
        tiled_mma, mC,
        a_smem_layout, b_smem_layout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        is_mma_warp = warp_idx < self.num_mma_warps
        is_tma_warp = warp_idx == self.tma_warp_id

        # TMA 描述符预取（减少首次 TMA 延迟）
        if is_tma_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        # ---- SMEM 分配 ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)

        tile_coord = (bidx, bidy, None)

        # ---- CTA 级分块 ----
        gA = cute.local_tile(mA_tma, self.tile_shape_mnk, tile_coord, (1, None, 1))
        gB = cute.local_tile(mB_tma, self.tile_shape_mnk, tile_coord, (None, 1, 1))
        gC = cute.local_tile(mC, self.tile_shape_mnk, tile_coord, (1, 1, None))

        # ---- TMA 分区 ----
        # group_modes: 将空间维度合并，留出 stage 维度
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 2), cute.group_modes(gA, 0, 2),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 2), cute.group_modes(gB, 0, 2),
        )

        # ---- MMA 分区 ----
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.threads_per_warp_group)
        wg_layout = cute.make_layout(self.num_mma_warp_groups, stride=self.threads_per_warp_group)
        thr_mma = tiled_mma.get_slice(wg_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        accumulators = cute.make_rmem_tensor(tCgC.shape, self.acc_dtype)

        # ---- 流水线设置 ----
        a_smem_one = cute.slice_(a_smem_layout, (None, None, 0))
        b_smem_one = cute.slice_(b_smem_layout, (None, None, 0))
        tma_bytes = cute.size_in_bytes(self.a_dtype, a_smem_one) + \
                    cute.size_in_bytes(self.b_dtype, b_smem_one)

        mbar_ptr = storage.pipeline_mbar_ptr.data_ptr()
        pipeline = PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, self.num_mma_warps),
            barrier_storage=mbar_ptr,
            tx_count=tma_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1))
        )

        p_state = make_pipeline_state(PipelineUserType.Producer, self.num_stages)
        c_state = make_pipeline_state(PipelineUserType.Consumer, self.num_stages)

        num_k_tiles = mA_tma.shape[1] // self.BK

        # ====== TMA Warp：数据搬运 ======
        if is_tma_warp:
            for kidx in range(num_k_tiles):
                pipeline.producer_acquire(p_state)
                cute.copy(tma_atom_a, tAgA[None, p_state.count], tAsA[None, p_state.index],
                          tma_bar_ptr=pipeline.producer_get_barrier(p_state))
                cute.copy(tma_atom_b, tBgB[None, p_state.count], tBsB[None, p_state.index],
                          tma_bar_ptr=pipeline.producer_get_barrier(p_state))
                pipeline.producer_commit(p_state)
                p_state.advance()

        # ====== MMA Warps：计算 ======
        if is_mma_warp:
            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)

            for kidx in range(num_k_tiles):
                pipeline.consumer_wait(c_state)
                cute.nvgpu.warpgroup.fence()

                # 沿 K 维度的子块循环
                for k_blk in range(self.BK // self.tiled_mma.shape_mnk[2]):
                    cute.gemm(
                        tiled_mma, accumulators,
                        tCrA[None, None, k_blk, c_state.index],
                        tCrB[None, None, k_blk, c_state.index],
                        accumulators,
                    )
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

                cute.nvgpu.warpgroup.commit_group()
                cute.nvgpu.warpgroup.wait_group(0)

                pipeline.consumer_release(c_state)
                c_state.advance()

            # ---- 写回 GMEM ----
            thr_mma_store = tiled_mma.get_slice(tidx)
            tCgC_store = thr_mma_store.partition_C(gC)
            atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
            tCrC_out = cute.make_fragment_like(accumulators, dtype=cutlass.Float16)
            for i in range(cute.size(tCrC_out)):
                tCrC_out[i] = cutlass.Float16(accumulators[i])
            cute.copy(atom_store, tCrC_out, tCgC_store)

    @staticmethod
    def _make_tma(tensor, smem_layout, tile_shape):
        op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        smem_one = cute.slice_(smem_layout, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_one, tile_shape)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("教程 06: Hopper TMA + WGMMA（概念讲解）")
    print("=" * 60)
    print()
    print("⚠️  本教程需要 SM90 (Hopper) 硬件。")
    print("    你的 GPU 是 SM120 (Blackwell)，不支持 WGMMA 指令。")
    print("    请阅读本文件中的代码和注释来学习 Hopper 架构概念。")
    print("    Blackwell 的 tcgen05 UMMA 请看教程 07。")
    print()
    print("Hopper vs Blackwell 关键区别：")
    print("  ┌──────────────┬──────────────────┬──────────────────┐")
    print("  │   特性        │  Hopper (SM90)   │ Blackwell (SM100)│")
    print("  ├──────────────┼──────────────────┼──────────────────┤")
    print("  │ MMA 指令      │  WGMMA           │ tcgen05 UMMA     │")
    print("  │ 累加器位置    │  寄存器 (RMEM)    │ TMEM (张量内存)  │")
    print("  │ MMA 发射      │  Warp Group      │ 单线程           │")
    print("  │ TMA           │  ✅ 支持          │ ✅ 支持          │")
    print("  │ Warp 特化     │  ✅ 支持          │ ✅ 支持          │")
    print("  │ 2CTA 协作     │  ❌ 不支持        │ ✅ 支持          │")
    print("  │ TMEM          │  ❌ 不存在        │ ✅ 新增          │")
    print("  └──────────────┴──────────────────┴──────────────────┘")
    print()
    print("🎉 教程 06 阅读完成！请继续教程 07。")
