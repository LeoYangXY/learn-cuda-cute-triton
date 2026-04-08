"""
=============================================================================
教程 07: Blackwell 架构 —— tcgen05 UMMA + TMEM + TMA
=============================================================================

Blackwell 家族包含多个 SM 版本：
  - SM100/SM101/SM103 (B100/B200): 数据中心 GPU，支持 tcgen05 UMMA
  - SM110 (GB100): 数据中心 GPU，支持 tcgen05 UMMA
  - SM120 (RTX 5050/5060/5070/5080/5090): 消费级 GPU

⚠️ 重要：SM120（你的 RTX 5050）不支持 tcgen05 UMMA 和 WGMMA！
  - CuTeDSL 4.4.2 的 tcgen05 MMA 只支持 SM100/SM110 家族
  - SM120 支持的 MMA 指令：
    1. SM80 级别的 WMMA（MmaF16BF16Op）—— 教程 05 使用的
    2. SM120 专属的 Block-Scaled MMA（MmaMXF4Op）—— 仅支持 FP4 数据类型
  - SM120 支持 TMA（继承自 Hopper）

本文件作为概念讲解，展示 Blackwell 数据中心 GPU 的关键特性。
代码可以阅读学习，但不能在 SM120 上执行。

1. tcgen05 UMMA（Unified MMA）
   - 替代 Hopper 的 WGMMA
   - 由单个线程发射（不需要 Warp Group 协作）
   - 支持更大的指令形状：128×256×16
   - 支持 2CTA 协作模式：两个 CTA 共享计算

2. TMEM（Tensor Memory）
   - 全新的内存层次：介于 SMEM 和 RMEM 之间
   - MMA 的累加器存储在 TMEM 中（而非寄存器）
   - 需要显式分配和释放
   - Epilogue 需要 TMEM → RMEM → GMEM 的数据搬运

3. TMA（继承自 Hopper）
   - 异步 GMEM ↔ SMEM 搬运
   - 支持 Multicast（一次 TMA 写入多个 CTA 的 SMEM）

4. 2CTA 协作 MMA
   - 两个 CTA 共享一个大的 MMA 指令
   - 每个 CTA 持有 A 的一半，共享 B
   - 硬件自动协调两个 CTA 的 SMEM 和 TMEM

Blackwell 数据流：
  GMEM → (TMA) → SMEM → (tcgen05 UMMA) → TMEM → (TMEM Load) → RMEM → GMEM

Warp 特化（本教程）：
  Warp 5 (threads 160-191): TMA 数据搬运
  Warp 4 (threads 128-159): tcgen05 UMMA 计算
  Warp 0-3 (threads 0-127): Epilogue（TMEM→RMEM→GMEM）

本教程可在你的 RTX 5050 (SM120) 上运行！
=============================================================================
"""

import torch
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.testing import benchmark, JitArguments


class GemmBlackwell:
    """
    Blackwell tcgen05 GEMM：TMA + UMMA + Warp Specialization + Pipeline

    这是 Blackwell 架构上的高性能 GEMM 实现，使用了所有新特性。
    """

    def __init__(
        self,
        cta_tile_shape_mnk=(128, 256, 64),  # 每个 CTA 处理的 tile 大小
    ):
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.BM, self.BN, self.BK = cta_tile_shape_mnk
        # MMA 指令形状：M=BM, N=min(BN,256), K=16
        self.mma_inst_shape_mnk = (self.BM, min(self.BN, 256), 16)

        # Warp 特化配置
        self.epi_warp_ids = (0, 1, 2, 3)  # Epilogue warps
        self.mma_warp_id = 4               # MMA warp
        self.tma_warp_id = 5               # TMA warp
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * 6  # 6 warps = 192 threads

        self.num_stages = 4  # 流水线深度

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.acc_dtype = cutlass.Float32

        # ====== 创建 tcgen05 UMMA 操作 ======
        # MmaF16BF16Op: Blackwell 的统一 MMA 操作
        # CtaGroup.ONE: 单 CTA 模式（不使用 2CTA 协作）
        # OperandSource.SMEM: 操作数从 SMEM 读取
        # OperandMajorMode.K: A 和 B 都是 K-major 存储
        op = tcgen05.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            self.mma_inst_shape_mnk,
            tcgen05.CtaGroup.ONE,          # 单 CTA
            tcgen05.OperandSource.SMEM,    # 从 SMEM 读操作数
            tcgen05.OperandMajorMode.K,    # A 是 K-major
            tcgen05.OperandMajorMode.K,    # B 是 K-major
        )
        self.tiled_mma = cute.make_tiled_mma(op)

        # ====== SMEM Layout（自动 Swizzle）======
        # sm100_utils 根据 MMA 操作自动生成最优的 SMEM layout
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            self.tiled_mma, self.cta_tile_shape_mnk, a.element_type, self.num_stages,
        )
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            self.tiled_mma, self.cta_tile_shape_mnk, b.element_type, self.num_stages,
        )

        # ====== TMA 描述符 ======
        # 单 stage 的 SMEM layout（用于创建 TMA 描述符）
        a_smem_one = cute.select(self.a_smem_layout, mode=[0, 1, 2])
        b_smem_one = cute.select(self.b_smem_layout, mode=[0, 1, 2])

        op_g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            op_g2s, a, a_smem_one, self.cta_tile_shape_mnk, self.tiled_mma,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            op_g2s, b, b_smem_one, self.cta_tile_shape_mnk, self.tiled_mma,
        )

        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # ====== Shared Storage ======
        @cute.struct
        class SharedStorage:
            mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32
        self.shared_storage = SharedStorage

        # ====== 启动 Kernel ======
        grid_dim = *cute.ceil_div(c.shape, (self.BM, self.BN)), 1
        self.kernel(
            self.tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            c, self.a_smem_layout, self.b_smem_layout,
        ).launch(grid=grid_dim, block=(self.threads_per_cta, 1, 1))

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom, mA_tma: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_tma: cute.Tensor,
        mC: cute.Tensor,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        bidx, bidy, _ = cute.arch.block_idx()
        mma_coord = (bidx, bidy, None)

        # TMA 描述符预取
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        # ---- SMEM 分配 ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = smem.allocate_tensor(self.a_dtype, a_smem_layout.outer, 128, swizzle=a_smem_layout.inner)
        sB = smem.allocate_tensor(self.b_dtype, b_smem_layout.outer, 128, swizzle=b_smem_layout.inner)

        # ---- TMEM 分配 ----
        # TMEM 是 Blackwell 新增的内存层次，用于存储 MMA 累加器
        # 需要显式分配和释放
        tmem_barrier_id = 1
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=tmem_barrier_id, num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
        )

        # ---- CTA 级分块 ----
        gA = cute.local_tile(mA_tma, self.cta_tile_shape_mnk, mma_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB_tma, self.cta_tile_shape_mnk, mma_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.cta_tile_shape_mnk, mma_coord, proj=(1, 1, None))

        # ---- MMA 分区 ----
        # tcgen05 的关键区别：get_slice(thr_idx=0)
        # 因为 UMMA 由单线程发射，分区只需要一个"虚拟"线程视图
        thr_mma = tiled_mma.get_slice(thr_idx=0)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgC = thr_mma.partition_C(gC)

        # Fragment A/B 直接引用 SMEM（多 stage）
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        # 累加器形状（存储在 TMEM 中）
        acc_shape = tiled_mma.partition_shape_C(self.cta_tile_shape_mnk[:2])
        tCtAcc = tiled_mma.make_fragment_C(acc_shape)

        # 分配 TMEM
        num_tmem_cols = utils.get_num_tmem_alloc_cols(tCtAcc)
        tmem.allocate(num_tmem_cols)

        # ---- TMA 分区 ----
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
        )

        # ---- 获取 TMEM 指针 ----
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)

        # ---- Epilogue 设置：TMEM → RMEM → GMEM ----
        epi_tile = self.cta_tile_shape_mnk[:2]
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, self.c_layout,
            self.c_dtype, self.acc_dtype, epi_tile, use_2cta_instrs=False,
        )
        tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0)], epi_tile)
        tmem_tiled_copy = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc_epi)
        tCgC_epi = cute.flat_divide(tCgC[((None, None), 0, 0)], epi_tile)
        tTR_gC = tmem_thr_copy.partition_D(tCgC_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0)].shape, self.acc_dtype)
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)

        # ---- 流水线设置 ----
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        tma_bytes = cute.size_in_bytes(
            self.a_dtype, cute.select(a_smem_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(
            self.b_dtype, cute.select(b_smem_layout, mode=[0, 1, 2])
        )

        producer, consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=tma_bytes,
            barrier_storage=storage.mbar_ptr.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        # MMA 完成信号的 mbarrier
        mma_mbar = storage.mma_mbar_ptr.data_ptr()
        if warp_idx == 0 and tidx == 0:
            cute.arch.mbarrier_init(mma_mbar, cnt=1)
            cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        num_k_tiles = mA_tma.shape[1] // self.BK
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)  # 清零累加器

        # ====== TMA Warp：数据搬运 ======
        if warp_idx == self.tma_warp_id:
            for kidx in cutlass.range(num_k_tiles):
                ab_empty = producer.acquire_and_advance()
                cute.copy(tma_atom_a, tAgA[None, ab_empty.count], tAsA[None, ab_empty.index],
                          tma_bar_ptr=ab_empty.barrier)
                cute.copy(tma_atom_b, tBgB[None, ab_empty.count], tBsB[None, ab_empty.index],
                          tma_bar_ptr=ab_empty.barrier)

        # ====== MMA Warp：tcgen05 UMMA 计算 ======
        if warp_idx == self.mma_warp_id:
            for kidx in cutlass.range(num_k_tiles):
                ab_full = consumer.wait_and_advance()

                # 沿 K 维度的子块循环
                num_k_blocks = cute.size(tCrA, mode=[2])
                for k_blk in cutlass.range_constexpr(num_k_blocks):
                    k_coord = (None, None, k_blk, ab_full.index)
                    cute.gemm(tiled_mma, tCtAcc, tCrA[k_coord], tCrB[k_coord], tCtAcc)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                ab_full.release()

            # 通知 UMMA 计算完成
            if tidx == self.mma_warp_id * self.threads_per_warp:
                tcgen05.commit(mma_mbar)

        # ====== Epilogue：TMEM → RMEM → GMEM ======
        tmem.relinquish_alloc_permit()

        # 等待 UMMA 完成
        cute.arch.mbarrier_wait(mma_mbar, 0)

        # 只有 Epilogue warps 执行写回
        if warp_idx <= self.epi_warp_ids[-1]:
            subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
            for subtile_idx in cutlass.range(subtile_cnt):
                # TMEM → RMEM
                cute.copy(tmem_tiled_copy, tTR_tAcc[(None, None, None, subtile_idx)], tTR_rAcc)
                # Float32 → Float16 类型转换
                tTR_rC.store(tTR_rAcc.load().to(self.c_dtype))
                # RMEM → GMEM
                cute.copy(simt_atom, tTR_rC, tTR_gC[(None, None, None, subtile_idx)])

        # TMA producer 收尾
        if warp_idx == self.tma_warp_id:
            producer.tail()

        # 同步后释放 TMEM
        pipeline.sync(barrier_id=tmem_barrier_id)
        tmem.free(tmem_ptr)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    import torch
    cap = torch.cuda.get_device_capability(0)
    sm = cap[0] * 10 + cap[1]

    print("=" * 60)
    print("教程 07: Blackwell tcgen05 GEMM（概念讲解 + 代码阅读）")
    print("=" * 60)
    print()
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM 版本: sm_{cap[0]}{cap[1]}0")
    print()

    # SM100/SM110 可以运行，SM120 目前 CuTeDSL 4.4.2 尚未完全支持
    if sm >= 100 and sm < 120:
        print("✅ 你的 GPU 支持 tcgen05，正在运行...")
        M, N, K = 4096, 4096, 4096
        A = torch.randn((M, K), device="cuda", dtype=torch.float16)
        B = torch.randn((N, K), device="cuda", dtype=torch.float16)
        ref = torch.matmul(A, B.T)
        A_ = from_dlpack(A, assumed_align=16)
        B_ = from_dlpack(B, assumed_align=16)
        C = torch.empty((M, N), device="cuda", dtype=torch.float16)
        C_ = from_dlpack(C, assumed_align=16)
        gemm = GemmBlackwell()
        compiled = cute.compile(gemm, A_, B_, C_)
        compiled(A_, B_, C_)
        assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "验证失败！"
        print("✅ 正确性验证通过！")
        time_us = benchmark(compiled, kernel_arguments=JitArguments(A_, B_, C_))
        tflops = (2 * M * N * K) / (time_us * 1e6)
        print(f"⏱  耗时: {time_us:.2f} µs | TFLOPS: {tflops:.4f}")
    else:
        print("⚠️  你的 GPU (SM120) 目前 CuTeDSL 4.4.2 尚未完全支持 tcgen05。")
        print("   这是因为 SM120 是 Blackwell 移动版，NVVM 编译器需要更新。")
        print("   请阅读本文件中的代码和注释来学习 Blackwell 架构概念。")
        print()
        print("Blackwell tcgen05 UMMA 的核心特性：")
        print()
        print("1. TMEM（Tensor Memory）—— 全新的内存层次")
        print("   - MMA 累加器存储在 TMEM 中，而非寄存器")
        print("   - 需要显式 alloc_tmem / dealloc_tmem")
        print("   - Epilogue: TMEM → RMEM → GMEM（三步写回）")
        print()
        print("2. tcgen05 UMMA —— 统一 MMA 指令")
        print("   - 由单线程发射（不需要 Warp Group 协作）")
        print("   - 支持 128×256×16 的大指令形状")
        print("   - 通过 tcgen05.commit(mbarrier) 通知完成")
        print()
        print("3. 2CTA 协作 MMA")
        print("   - 两个 CTA 共享一个 256×256×16 的 MMA 指令")
        print("   - 每个 CTA 持有 A 的一半（128 行），共享 B")
        print("   - TMA Multicast 一次写入两个 CTA 的 SMEM")
        print()
        print("4. Warp 特化（与 Hopper 类似）")
        print("   - Warp 5: TMA 数据搬运")
        print("   - Warp 4: UMMA 计算")
        print("   - Warp 0-3: Epilogue 写回")

    print("\n🎉 教程 07 完成！")
