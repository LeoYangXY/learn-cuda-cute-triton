"""
=============================================================================
教程 09: 进阶技巧 —— Persistent Kernel、Dynamic Shape、Cluster
=============================================================================

本教程补充几个重要的进阶知识点：

1. Persistent Kernel（持久化 Kernel）
   - 传统方式：每个 CTA 处理一个 tile，处理完就退出
   - Persistent：每个 CTA 循环处理多个 tile，直到所有 tile 完成
   - 优势：减少 kernel launch 开销，更好的 SM 利用率
   - 关键 API：PersistentTileSchedulerParams + StaticPersistentTileScheduler

2. Dynamic Shape（动态形状）
   - 编译一次，运行时可以改变某些维度的大小
   - 关键 API：mark_compact_shape_dynamic(mode=0)
   - 适用场景：batch size 变化、序列长度变化

3. zipped_divide（层次化分块）
   - 将张量按 tiler 大小切分为 ((tile_shape), (num_tiles)) 的层次结构
   - 可以用 [tile_coord, inner_coord] 来索引
   - 比 local_tile 更灵活，适合需要同时访问 tile 内外坐标的场景

4. Cluster（线程块集群）
   - Hopper+ 支持多个 CTA 组成 Cluster
   - Cluster 内的 CTA 可以通过 DSMEM 直接访问彼此的 SMEM
   - 用于 TMA Multicast 和跨 CTA 归约

5. elect_one（选举）
   - 在一个 Warp 中选出一个线程执行操作
   - 比 if tidx == 0 更高效（不会导致 warp divergence）

6. 其他实用 API
   - cute.math.exp / cute.math.log / cute.math.log2
   - cute.arch.shuffle_sync_bfly（butterfly shuffle）
   - cute.arch.warp_reduction（内置 warp 归约）
   - cute.arch.barrier（命名 barrier）
   - cutlass.range / cutlass.range_constexpr（循环展开控制）
=============================================================================
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments


# =============================================================================
# 第一部分：Persistent Kernel
# =============================================================================
# 传统 kernel：grid 中每个 CTA 处理一个 tile
# Persistent kernel：grid 只启动 SM 数量的 CTA，每个 CTA 循环处理多个 tile
#
# 优势：
#   1. 减少 kernel launch 和 CTA 调度开销
#   2. 更好的 L2 cache 局部性（相邻 tile 可能共享数据）
#   3. 是 Flash Attention v4 的关键技术之一

@cute.jit
def persistent_gemm_launcher(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
):
    BM, BN, BK = 16, 32, 16
    c_tile_shape = (BM, BN)

    # 计算总共有多少个输出 tile
    gC = cute.zipped_divide(mC, tiler=c_tile_shape)
    num_ctas_mnl = (*gC[(0, (None, None))].shape, 1)

    cluster_shape_mnl = (1, 1, 1)

    # PersistentTileSchedulerParams：告诉调度器有多少 tile 需要处理
    tile_sched_params = utils.PersistentTileSchedulerParams(
        num_ctas_mnl,
        cluster_shape_mnl,
        swizzle_size=1,          # Swizzle 大小（影响 tile 遍历顺序）
        raster_along_m=True,     # 沿 M 维度光栅化（影响 L2 cache 命中率）
    )

    # 计算 grid 大小：不是 tile 总数，而是 SM 数量
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    persistent_gemm_kernel(mA, mB, mC, tile_sched_params).launch(
        grid=grid,
        block=[256, 1, 1],
    )


@cute.kernel
def persistent_gemm_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    tile_sched_params: utils.PersistentTileSchedulerParams,
):
    BM, BN, BK = 16, 32, 16
    tidx, _, _ = cute.arch.thread_idx()

    tiler = (BM, BN, BK)
    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)
    thr_mma = tiled_mma.get_slice(tidx)

    # 创建 tile 调度器
    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()

    # ====== 持久化循环：每个 CTA 处理多个 tile ======
    while work_tile.is_valid_tile:
        tile_m = work_tile.tile_idx[0]
        tile_n = work_tile.tile_idx[1]
        coord = (tile_m, tile_n, None)

        # 标准的 CTA 级分块
        gA_tile = cute.local_tile(gA, tiler=tiler, coord=coord, proj=(1, None, 1))
        gB_tile = cute.local_tile(gB, tiler=tiler, coord=coord, proj=(None, 1, 1))
        gC_tile = cute.local_tile(gC, tiler=tiler, coord=coord, proj=(1, 1, None))

        tCgC = thr_mma.partition_C(gC_tile)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0)

        for k in range(gA_tile.shape[2]):
            gA_k = gA_tile[None, None, k]
            gB_k = gB_tile[None, None, k]
            tCgA = thr_mma.partition_A(gA_k)
            tCgB = thr_mma.partition_B(gB_k)
            tCrA = tiled_mma.make_fragment_A(tCgA)
            tCrB = tiled_mma.make_fragment_B(tCgB)
            tCrA.store(tCgA.load())
            tCrB.store(tCgB.load())
            cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

        tCgC.store(tCrC.load())

        # 获取下一个 tile（调度器自动分配）
        tile_sched.advance_to_next_work()
        work_tile = tile_sched.get_current_work()


# =============================================================================
# 第二部分：Dynamic Shape（动态形状）
# =============================================================================
# 编译一次 kernel，运行时可以改变 batch size
# 关键：mark_compact_shape_dynamic(mode=0) 标记哪个维度是动态的

@cute.jit
def batched_gemm(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    grid_m = (mA.shape[1] + BM - 1) // BM
    grid_n = (mB.shape[1] + BN - 1) // BN

    # grid 的 z 维度 = batch size（动态）
    batched_gemm_kernel(mA, mB, mC).launch(
        grid=[grid_m, grid_n, mA.shape[0]],
        block=[256, 1, 1]
    )


@cute.kernel
def batched_gemm_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    BM, BN, BK = 16, 32, 16
    bidx, bidy, bidz = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()

    # 先切 batch 维度，再做 2D tiling
    gA_batch = gA[bidz, None, None]  # (M, K)
    gB_batch = gB[bidz, None, None]  # (N, K)
    gC_batch = gC[bidz, None, None]  # (M, N)

    tiler = (BM, BN, BK)
    coord = (bidx, bidy, None)

    gA_tile = cute.local_tile(gA_batch, tiler=tiler, coord=coord, proj=(1, None, 1))
    gB_tile = cute.local_tile(gB_batch, tiler=tiler, coord=coord, proj=(None, 1, 1))
    gC_tile = cute.local_tile(gC_batch, tiler=tiler, coord=coord, proj=(1, 1, None))

    atoms_layout = cute.make_layout((16, 16, 1), stride=(16, 1, 0))
    mma_atom = cute.nvgpu.MmaUniversalOp(cutlass.Float32)
    tiled_mma = cute.make_tiled_mma(mma_atom, atoms_layout)
    thr_mma = tiled_mma.get_slice(tidx)

    tCgC = thr_mma.partition_C(gC_tile)
    tCrC = tiled_mma.make_fragment_C(tCgC)
    tCrC.fill(0)

    for k in range(gA_tile.shape[2]):
        gA_k = gA_tile[None, None, k]
        gB_k = gB_tile[None, None, k]
        tCgA = thr_mma.partition_A(gA_k)
        tCgB = thr_mma.partition_B(gB_k)
        tCrA = tiled_mma.make_fragment_A(tCgA)
        tCrB = tiled_mma.make_fragment_B(tCgB)
        tCrA.store(tCgA.load())
        tCrB.store(tCgB.load())
        cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)

    tCgC.store(tCrC.load())


# =============================================================================
# 第三部分：zipped_divide 详解
# =============================================================================

@cute.kernel
def zipped_divide_demo_kernel(gA: cute.Tensor):
    """演示 zipped_divide 的层次化索引"""
    tidx, tidy, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    # zipped_divide 将 (M, N) 切分为 ((TileM, TileN), (NumTilesM, NumTilesN))
    # 第一个元素是 tile 内部坐标，第二个是 tile 索引
    TILE_M, TILE_N = 4, 4
    gA_tiled = cute.zipped_divide(gA, (TILE_M, TILE_N))

    # 层次化坐标：((tile 内行, tile 内列), (tile 行索引, tile 列索引))
    coord = ((tidy, tidx), (bidy, bidx))

    # 直接用层次化坐标索引
    val = gA_tiled[coord]
    if bidx == 0 and bidy == 0:
        cute.printf("Thread (%d,%d): A[%d,%d] = %f\n", tidx, tidy, tidy, tidx, val)


@cute.jit
def zipped_divide_demo(mA: cute.Tensor):
    TILE_M, TILE_N = 4, 4
    M, N = mA.shape
    zipped_divide_demo_kernel(mA).launch(
        grid=(N // TILE_N, M // TILE_M, 1),
        block=(TILE_N, TILE_M, 1)
    )


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    # ---- Part 1: Persistent Kernel ----
    print("=" * 60)
    print("第一部分：Persistent Kernel GEMM")
    print("=" * 60)

    M, N, K = 1024, 1024, 1024
    A = torch.randn((M, K), device="cuda", dtype=torch.float32)
    B = torch.randn((N, K), device="cuda", dtype=torch.float32)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)

    A_ = from_dlpack(A, assumed_align=16)
    B_ = from_dlpack(B, assumed_align=16)
    C_ = from_dlpack(C, assumed_align=16)

    device = torch.cuda.current_device()
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    max_active_clusters = sm_count

    compiled_persistent = cute.compile(persistent_gemm_launcher, A_, B_, C_, max_active_clusters)
    compiled_persistent(A_, B_, C_)

    ref = torch.matmul(A, B.T)
    assert torch.allclose(C, ref, atol=1e-1, rtol=1e-1), "Persistent GEMM 验证失败！"
    print(f"✅ Persistent Kernel GEMM 正确性验证通过！（SM 数量: {sm_count}）")

    time_us = benchmark(compiled_persistent, kernel_arguments=JitArguments(A_, B_, C_))
    tflops = (2 * M * N * K) / (time_us * 1e6)
    print(f"⏱  Persistent GEMM 耗时: {time_us:.2f} µs | TFLOPS: {tflops:.4f}")

    # PyTorch 对比
    C_pt = torch.empty_like(ref)
    for _ in range(10):
        torch.matmul(A, B.T, out=C_pt)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        torch.matmul(A, B.T, out=C_pt)
    end.record()
    torch.cuda.synchronize()
    pt_time_us = start.elapsed_time(end) * 1000 / 100
    pt_tflops = (2 * M * N * K) / (pt_time_us * 1e6)
    print(f"⏱  PyTorch        耗时: {pt_time_us:.2f} µs | TFLOPS: {pt_tflops:.4f}")

    # ---- Part 2: Dynamic Shape Batched GEMM ----
    print("\n" + "=" * 60)
    print("第二部分：Dynamic Shape Batched GEMM")
    print("=" * 60)

    BS, M, N, K = 8, 512, 512, 512
    A2 = torch.randn((BS, M, K), device="cuda", dtype=torch.float32)
    B2 = torch.randn((BS, N, K), device="cuda", dtype=torch.float32)
    C2 = torch.empty((BS, M, N), device="cuda", dtype=torch.float32)

    # mark_compact_shape_dynamic: 标记 mode=0（batch 维度）为动态
    A2_ = from_dlpack(A2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=A2.dim_order())
    B2_ = from_dlpack(B2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=B2.dim_order())
    C2_ = from_dlpack(C2, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=C2.dim_order())

    compiled_batched = cute.compile(batched_gemm, A2_, B2_, C2_)
    compiled_batched(A2_, B2_, C2_)

    ref2 = torch.bmm(A2, B2.mT)
    assert torch.allclose(C2, ref2, atol=1e-1, rtol=1e-1), "Batched GEMM (BS=8) 验证失败！"
    print(f"✅ Batched GEMM (BS={BS}) 正确性验证通过！")

    # 复用编译好的 kernel，改变 batch size（无需重新编译！）
    BS2 = 4
    A3 = torch.randn((BS2, M, K), device="cuda", dtype=torch.float32)
    B3 = torch.randn((BS2, N, K), device="cuda", dtype=torch.float32)
    C3 = torch.empty((BS2, M, N), device="cuda", dtype=torch.float32)

    A3_ = from_dlpack(A3, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=A3.dim_order())
    B3_ = from_dlpack(B3, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=B3.dim_order())
    C3_ = from_dlpack(C3, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=C3.dim_order())

    compiled_batched(A3_, B3_, C3_)  # 复用！不重新编译！

    ref3 = torch.bmm(A3, B3.mT)
    assert torch.allclose(C3, ref3, atol=1e-1, rtol=1e-1), "Batched GEMM (BS=4) 验证失败！"
    print(f"✅ Batched GEMM (BS={BS2}) 复用编译 kernel 验证通过！（无需重新编译）")

    # ---- Part 3: zipped_divide ----
    print("\n" + "=" * 60)
    print("第三部分：zipped_divide 层次化分块")
    print("=" * 60)

    A4 = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    A4_ = from_dlpack(A4, assumed_align=16)
    zipped_divide_demo(A4_)
    torch.cuda.synchronize()
    print("✅ zipped_divide 演示完成！")

    print("\n🎉 教程 09 全部完成！")
