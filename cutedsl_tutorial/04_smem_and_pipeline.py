"""
=============================================================================
教程 04: Shared Memory 与异步流水线
=============================================================================

为什么需要 Shared Memory（SMEM）？
  - 全局内存（GMEM）延迟高（~400 cycles），带宽有限
  - Shared Memory 是片上 SRAM，延迟低（~20 cycles），带宽高
  - 策略：先把数据从 GMEM 搬到 SMEM，再从 SMEM 读取计算

GEMM 中的 SMEM 使用模式：
  ┌─────────────────────────────────────────────────────┐
  │  GMEM                                               │
  │  ┌─────┐  ┌─────┐                                   │
  │  │  A  │  │  B  │                                   │
  │  └──┬──┘  └──┬──┘                                   │
  │     │ copy   │ copy                                  │
  │     ▼        ▼                                       │
  │  ┌─────┐  ┌─────┐  ← SMEM（片上，低延迟）            │
  │  │ sA  │  │ sB  │                                   │
  │  └──┬──┘  └──┬──┘                                   │
  │     │ load   │ load                                  │
  │     ▼        ▼                                       │
  │  ┌─────────────┐   ← 寄存器（最快）                  │
  │  │  MMA 计算    │                                    │
  │  └─────────────┘                                    │
  └─────────────────────────────────────────────────────┘

异步流水线（Pipeline）：
  不等上一轮数据用完就开始加载下一轮，让加载和计算重叠
  ┌──────────────────────────────────────────────────────┐
  │  Stage 0: [加载 K=0] [计算 K=0] [加载 K=2]          │
  │  Stage 1: [加载 K=1] [计算 K=1] [加载 K=3]          │
  │  ...                                                 │
  └──────────────────────────────────────────────────────┘

关键 API：
  SmemAllocator()                    — SMEM 分配器
  allocate_tensor(dtype, layout)     — 在 SMEM 中分配张量
  cute.arch.sync_threads()           — Block 内线程同步（__syncthreads）
  PipelineAsync                      — 异步流水线（生产者-消费者模型）
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
from cutlass.pipeline import PipelineAsync, CooperativeGroup, Agent
import torch


# =============================================================================
# 第一部分：SMEM GEMM（手动搬运，sync_threads 同步）
# =============================================================================
# 最朴素的 SMEM 使用方式：
# 1. 所有线程协作把 A/B 的一个 tile 从 GMEM 搬到 SMEM
# 2. sync_threads() 确保搬运完成
# 3. 从 SMEM 读取数据做计算
# 4. sync_threads() 确保计算完成后再搬下一个 tile

BM, BN, BK = 16, 16, 16
PAD = 8  # Padding 避免 bank conflict

@cute.kernel
def smem_gemm_kernel(
    gA: cute.Tensor,    # A: [M, K]
    gB: cute.Tensor,    # B: [N, K]
    gC: cute.Tensor,    # C: [M, N]
    M: int, N: int, K: int,
):
    # ---- SMEM 分配 ----
    # SmemAllocator 是 CuTeDSL 的 SMEM 分配器
    allocator = cutlass.utils.SmemAllocator()

    # 创建 SMEM Layout，加 PAD 避免 bank conflict
    # Bank conflict：同一 warp 的线程访问同一 bank 会串行化
    # 加 padding 让相邻行错开 bank，消除 conflict
    layout_sA = cute.make_layout((BM, BK), stride=(BK + PAD, 1))
    layout_sB = cute.make_layout((BN, BK), stride=(BK + PAD, 1))

    # 在 SMEM 中分配张量
    sA = allocator.allocate_tensor(cutlass.Float32, layout_sA, 16, None)
    sB = allocator.allocate_tensor(cutlass.Float32, layout_sB, 16, None)

    # ---- 线程索引 ----
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, bdimy, _ = cute.arch.block_dim()
    tidx, tidy, _ = cute.arch.thread_idx()
    tid = tidy * bdimx + tidx       # 线性线程 ID
    num_threads = bdimx * bdimy      # Block 内总线程数

    acc = cutlass.Float32(0)

    # ---- 主循环：沿 K 维度迭代 ----
    for ctak in range(0, K, BK):
        # 协作加载 A tile: 所有线程分工搬运
        # 每个线程搬运 BM*BK/num_threads 个元素
        num_loads_A = BM * BK
        for i in range(tid, num_loads_A, num_threads):
            row = i // BK
            col = i % BK
            sA[row, col] = gA[bidy * BM + row, ctak + col]

        # 协作加载 B tile
        num_loads_B = BN * BK
        for i in range(tid, num_loads_B, num_threads):
            row = i // BK
            col = i % BK
            sB[row, col] = gB[bidx * BN + row, ctak + col]

        # 同步：确保所有线程都完成了 SMEM 写入
        cute.arch.sync_threads()

        # 从 SMEM 读取并计算
        for mmak in range(BK):
            acc += sA[tidx, mmak] * sB[tidy, mmak]

        # 同步：确保所有线程都读完了 SMEM，才能开始下一轮写入
        cute.arch.sync_threads()

    gC[bidy * BM + tidx, bidx * BN + tidy] = acc


@cute.jit
def smem_gemm_host(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    M = mA.shape[0]
    N = mB.shape[0]
    K = mA.shape[1]

    smem_gemm_kernel(mA, mB, mC, M, N, K).launch(
        grid=(N // BN, M // BM, 1),
        block=(BM, BN, 1)
    )


# =============================================================================
# 第二部分：异步流水线 GEMM（PipelineAsync）
# =============================================================================
# 上面的版本有个问题：加载和计算是串行的
# 异步流水线让加载和计算重叠，大幅提升性能
#
# 核心思想：
#   - 多个 SMEM buffer（stage），轮流使用
#   - Producer warp 负责加载数据到 SMEM
#   - Consumer warp 负责从 SMEM 读取并计算
#   - 通过 mbarrier 同步：producer 写完通知 consumer，consumer 用完通知 producer

class GemmPipeAsync:
    def __init__(self):
        self.BM, self.BN, self.BK = 8, 4, 16
        self.padding = 8
        self.num_stages = 4          # 4 个 SMEM buffer 轮流使用
        self.num_producer_threads = 32  # 1 个 warp 做加载
        self.num_consumer_threads = 32  # 1 个 warp 做计算

    @cute.jit
    def __call__(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        # 多 stage 的 SMEM Layout：shape = (num_stages, BM, BK)
        # 第一个维度是 stage 索引
        sA_layout = cute.make_layout(
            shape=(self.num_stages, self.BM, self.BK),
            stride=(self.BM * (self.BK + self.padding), self.BK + self.padding, 1)
        )
        sB_layout = cute.make_layout(
            shape=(self.num_stages, self.BN, self.BK),
            stride=(self.BN * (self.BK + self.padding), self.BK + self.padding, 1)
        )

        # SharedStorage 结构体：包含 pipeline 的 mbarrier 和 SMEM buffer
        @cute.struct
        class SharedStorage:
            pipeline_mbarrier_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_stages * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)],
                1024,
            ]

        self.shared_storage = SharedStorage

        M, N = mC.shape
        self.kernel(mA, mB, mC, sA_layout, sB_layout).launch(
            grid=(N // self.BN, M // self.BM, 1),
            block=(self.num_producer_threads + self.num_consumer_threads, 1, 1)
        )

    @cute.kernel
    def kernel(self, gA, gB, gC, sA_layout, sB_layout):
        BM, BN, BK = self.BM, self.BN, self.BK

        # 分配 SMEM
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(layout=sA_layout)
        sB = storage.sB.get_tensor(layout=sB_layout)

        bidx, bidy, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # ---- 创建异步流水线 ----
        # PipelineAsync 实现了生产者-消费者模型
        # producer_group: 加载数据的线程组
        # consumer_group: 计算的线程组
        mainloop_pipeline = PipelineAsync.create(
            num_stages=self.num_stages,
            producer_group=CooperativeGroup(Agent.Thread, self.num_producer_threads),
            consumer_group=CooperativeGroup(Agent.Thread, self.num_consumer_threads),
            barrier_storage=storage.pipeline_mbarrier_ptr.data_ptr()
        )

        producer, consumer = mainloop_pipeline.make_participants()

        # ---- Producer Warp（Warp 0）：加载数据 ----
        if warp_idx == 0:
            tid, _, _ = cute.arch.thread_idx()

            for ctak in range(0, gA.shape[1], BK):
                # acquire: 等待 consumer 释放这个 stage 的 buffer
                handle = producer.acquire_and_advance()

                # 协作加载 A tile 到 sA[stage_idx]
                num_loads_A = BM * BK
                for i in range(tid, num_loads_A, self.num_producer_threads):
                    row = i // BK
                    col = i % BK
                    sA[handle.index, row, col] = gA[bidy * BM + row, ctak + col]

                # 协作加载 B tile 到 sB[stage_idx]
                num_loads_B = BN * BK
                for i in range(tid, num_loads_B, self.num_producer_threads):
                    row = i // BK
                    col = i % BK
                    sB[handle.index, row, col] = gB[bidx * BN + row, ctak + col]

                # commit: 通知 consumer 这个 stage 的数据已就绪
                handle.commit()

            # tail: 通知 consumer 所有数据加载完毕
            producer.tail()

        # ---- Consumer Warp（Warp 1）：计算 ----
        if warp_idx == 1:
            tid, _, _ = cute.arch.thread_idx()
            tid = tid - self.num_producer_threads
            tidx = tid % BM
            tidy = tid // BM

            acc = cutlass.Float32(0)

            for ctak in range(0, gA.shape[1], BK):
                # wait: 等待 producer 填充好这个 stage 的数据
                handle = consumer.wait_and_advance()

                # 从 sA/sB[stage_idx] 读取并计算
                for mmak in range(BK):
                    acc += sA[handle.index, tidx, mmak] * sB[handle.index, tidy, mmak]

                # release: 通知 producer 这个 stage 的 buffer 可以复用了
                handle.release()

            gC[bidy * BM + tidx, bidx * BN + tidy] = acc


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    M_SIZE, N_SIZE, K_SIZE = 1024, 1024, 1024

    A = torch.randn((M_SIZE, K_SIZE), device="cuda", dtype=torch.float32)
    B = torch.randn((N_SIZE, K_SIZE), device="cuda", dtype=torch.float32)
    ref = torch.matmul(A, B.T)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)

    # ---- Part 1: SMEM GEMM ----
    print("=" * 60)
    print("第一部分：SMEM GEMM（手动搬运 + sync_threads）")
    print("=" * 60)

    C1 = torch.empty((M_SIZE, N_SIZE), device="cuda", dtype=torch.float32)
    C1_ = from_dlpack(C1, assumed_align=16)

    compiled_smem = cute.compile(smem_gemm_host, A_, B_, C1_)
    compiled_smem(A_, B_, C1_)

    assert torch.allclose(C1, ref, atol=1e-1, rtol=1e-1), "SMEM GEMM 验证失败！"
    print("✅ SMEM GEMM 正确性验证通过！")

    time_smem = benchmark(compiled_smem, kernel_arguments=JitArguments(A_, B_, C1_))
    tflops_smem = (2 * M_SIZE * N_SIZE * K_SIZE) / (time_smem * 1e6)
    print(f"⏱  耗时: {time_smem:.2f} µs | TFLOPS: {tflops_smem:.4f}")

    # ---- Part 2: Pipeline GEMM ----
    print("\n" + "=" * 60)
    print("第二部分：异步流水线 GEMM（PipelineAsync）")
    print("=" * 60)

    C2 = torch.empty((M_SIZE, N_SIZE), device="cuda", dtype=torch.float32)
    C2_ = from_dlpack(C2, assumed_align=16)

    gemm_pipe = GemmPipeAsync()
    compiled_pipe = cute.compile(gemm_pipe, A_, B_, C2_)
    compiled_pipe(A_, B_, C2_)

    assert torch.allclose(C2, ref, atol=1e-1, rtol=1e-1), "Pipeline GEMM 验证失败！"
    print("✅ Pipeline GEMM 正确性验证通过！")

    time_pipe = benchmark(compiled_pipe, kernel_arguments=JitArguments(A_, B_, C2_))
    tflops_pipe = (2 * M_SIZE * N_SIZE * K_SIZE) / (time_pipe * 1e6)
    print(f"⏱  耗时: {time_pipe:.2f} µs | TFLOPS: {tflops_pipe:.4f}")

    print(f"\n📊 Pipeline vs SMEM 加速比: {time_smem / time_pipe:.2f}x")

    # ---- PyTorch 性能对比 ----
    print("\n" + "=" * 60)
    print("📊 性能对比：CuTeDSL vs PyTorch")
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
    pt_tflops = (2 * M_SIZE * N_SIZE * K_SIZE) / (pt_time_us * 1e6)
    print(f"⏱  PyTorch       耗时: {pt_time_us:.2f} µs | TFLOPS: {pt_tflops:.4f}")
    print(f"⏱  SMEM GEMM     耗时: {time_smem:.2f} µs | TFLOPS: {tflops_smem:.4f}")
    print(f"⏱  Pipeline GEMM 耗时: {time_pipe:.2f} µs | TFLOPS: {tflops_pipe:.4f}")

    print("\n🎉 教程 04 全部完成！")
