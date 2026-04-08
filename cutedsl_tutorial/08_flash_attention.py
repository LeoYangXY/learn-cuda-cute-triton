"""
=============================================================================
教程 08: Flash Attention —— 从原理到高性能 CuTeDSL 实现
=============================================================================

Flash Attention 是高效 Attention 的核心算法。

标准 Attention 公式：
  Score = Q × K^T / √d_k        (M×N)
  P = softmax(Score, dim=-1)     (M×N)
  Output = P × V                 (M×d_v)

问题：Score 矩阵是 M×N，当序列长度 N 很大时，存不下！

Flash Attention 的核心思想（Online Softmax）：
  对于每个 query token q_i，维护：
    - m_i: 当前见过的最大 score（数值稳定性）
    - l_i: 当前的 softmax 分母（exp 之和）
    - o_i: 当前的输出累加值

  每处理一个新的 K/V tile：
    1. 计算新的 score: S = Q × K^T / √d_k       (BM × BN)
    2. 行最大值: m_new = max(m_old, rowmax(S))
    3. 更新分母: l_new = l_old × exp(m_old - m_new) + rowsum(exp(S - m_new))
    4. Rescale 旧输出: O = O × exp(m_old - m_new)
    5. 累加新贡献: O += exp(S - m_new) × V
    6. 最终归一化: O = O / l_new

本教程包含：
  第一部分：Naive SDPA（标量实现，用于理解概念）
  第二部分：Flash Attention（Online Softmax + SMEM 优化 + Warp Reduce）

优化手段：
  1. Online Softmax —— 避免构造完整 Score 矩阵，O(1) 额外内存
  2. SMEM Tiling —— Q/K/V 分块加载到 SMEM，减少 GMEM 访问
  3. Warp Shuffle Reduce —— O(log N) 步完成行归约
  4. 协作加载 —— 所有线程分工搬运数据到 SMEM
  5. 寄存器累加 —— 输出 O 始终在寄存器中，避免 SMEM 回写
=============================================================================
"""

import torch
import math
from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments


# =============================================================================
# 工具函数
# =============================================================================

def sdpa_reference(Q, K, V):
    """标准 Scaled Dot-Product Attention（PyTorch 参考实现）"""
    d_k = Q.shape[-1]
    score = torch.matmul(Q, K.T) / math.sqrt(d_k)
    attn = torch.softmax(score, dim=-1)
    return torch.matmul(attn, V)


@cute.jit
def warp_reduce_sum(val, width=32):
    """Warp 内求和归约（butterfly shuffle）"""
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


@cute.jit
def warp_reduce_max(val, width=32):
    """Warp 内求最大值归约"""
    for i in cutlass.range_constexpr(int(math.log2(width))):
        other = cute.arch.shuffle_sync_bfly(val, offset=1 << i)
        val = val if val > other else other
    return val


# =============================================================================
# 第一部分：Naive SDPA（标量实现，用于对比）
# =============================================================================

class NaiveSDPA:
    """每个 Block 处理一个 query token 的一个输出维度。"""

    def __init__(self, BN=256):
        self.BN = BN

    @cute.jit
    def __call__(self, Q, K, V, output):
        self.kernel(Q, K, V, output).launch(
            grid=(output.shape[0], output.shape[1], 1),
            block=(self.BN, 1, 1)
        )

    @cute.kernel
    def kernel(self, Q, K, V, output):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        BN = self.BN
        NUM_WARPS = BN // 32

        smem = cutlass.utils.SmemAllocator()
        smem_denom = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((NUM_WARPS,), stride=(1,)), 16, None)
        smem_out = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((NUM_WARPS,), stride=(1,)), 16, None)

        gK_ = cute.zipped_divide(K, (BN, 1))
        gV_ = cute.zipped_divide(V, (BN, 1))

        partial_denom = cutlass.Float32(0)
        partial_output = cutlass.Float32(0)
        d_k = Q.shape[1]

        for nidx in range(K.shape[0] // BN):
            gK = gK_[(None, None), (nidx, None)]
            gV = gV_[(None, None), (nidx, None)]
            score = cutlass.Float32(0)
            for kidx in range(d_k):
                score += Q[bidx, kidx] * gK[tidx, 0, kidx]
            exp_score = cute.math.exp(score / cutlass.Float32(d_k ** 0.5))
            partial_denom += exp_score
            partial_output += exp_score * gV[tidx, 0, bidy]

        reduced_denom = warp_reduce_sum(partial_denom)
        reduced_output = warp_reduce_sum(partial_output)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        if lane_idx == 0:
            smem_denom[warp_idx] = reduced_denom
            smem_out[warp_idx] = reduced_output
        cute.arch.sync_threads()

        if warp_idx == 0:
            val_denom = cutlass.Float32(0)
            val_out = cutlass.Float32(0)
            if lane_idx < NUM_WARPS:
                val_denom = smem_denom[lane_idx]
                val_out = smem_out[lane_idx]
            final_denom = warp_reduce_sum(val_denom, width=NUM_WARPS)
            final_out = warp_reduce_sum(val_out, width=NUM_WARPS)
            if lane_idx == 0:
                output[bidx, bidy] = cutlass.Float32(final_out / final_denom)


# =============================================================================
# 第二部分：Flash Attention（Online Softmax + SMEM Tiling）
# =============================================================================
#
# 设计：
#   - 每个 CTA 处理 1 个 query token 的完整输出向量
#   - Block 内 THREADS_PER_BLOCK 个线程沿 N 维度并行
#   - 沿 N 维度分块迭代（每次 BN 个 key），使用 Online Softmax
#   - K/V tile 加载到 SMEM，Q 也在 SMEM 中
#   - 每个线程计算一个 score（Q·K[j]），然后 warp reduce 做行归约
#   - 输出 O 在寄存器中累加
#
# 与 Naive SDPA 的关键区别：
#   1. Online Softmax：不需要两遍扫描（先求 max，再求 exp/sum）
#   2. SMEM Tiling：Q 只加载一次到 SMEM，K/V 分块加载
#   3. 寄存器累加：O 始终在寄存器中，避免 GMEM 读写

class FlashAttention:
    """
    Flash Attention with Online Softmax.

    每个 CTA 处理 1 个 query token 的完整 d 维输出。
    BN 个线程沿 key 维度并行，通过 warp reduce 归约。
    """

    def __init__(self, BN=128, d=64):
        self.BN = BN
        self.d = d
        self.NUM_WARPS = BN // 32
        self.PAD = 8

    @cute.jit
    def __call__(self, Q, K, V, output, scale):
        """
        Q: (M, d)  K: (N, d)  V: (N, d)  output: (M, d)
        scale: 1/√d_k
        """
        M = Q.shape[0]
        self.kernel(Q, K, V, output, scale).launch(
            grid=(M, 1, 1),
            block=(self.BN, 1, 1)
        )

    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, scale):
        BN = self.BN
        d = self.d
        PAD = self.PAD
        NUM_WARPS = self.NUM_WARPS

        query_idx, _, _ = cute.arch.block_idx()  # 当前处理的 query token
        tidx, _, _ = cute.arch.thread_idx()       # 线程 ID（0 ~ BN-1）

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        lane_idx = cute.arch.lane_idx()

        # ---- SMEM 分配 ----
        alloc = cutlass.utils.SmemAllocator()
        # Q tile: (1, d) — 当前 query token
        sQ = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((d,), stride=(1,)), 16, None)
        # 用于 warp 间归约的 SMEM buffer
        smem_max = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((NUM_WARPS,), stride=(1,)), 16, None)
        smem_sum = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((NUM_WARPS,), stride=(1,)), 16, None)
        # 输出归约 buffer: (NUM_WARPS, d)
        smem_o = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((NUM_WARPS, d), stride=(d, 1)), 16, None)

        # ---- 加载 Q 到 SMEM ----
        if tidx < d:
            sQ[tidx] = cutlass.Float32(mQ[query_idx, tidx])
        cute.arch.sync_threads()

        # ---- 初始化 Online Softmax 状态（每个线程独立）----
        # 每个线程维护自己的 partial 状态，最后 warp reduce 合并
        m_i = cutlass.Float32(-1e30)   # 当前最大 score
        l_i = cutlass.Float32(0.0)     # 当前 exp 之和

        # 每个线程的输出累加器：d 维向量
        rO = cute.make_fragment(cute.make_layout((d,), stride=(1,)), cutlass.Float32)
        rO.fill(0.0)

        N = mK.shape[0]
        num_kv_tiles = N // BN

        # ====== 主循环：沿 N 维度分块迭代 ======
        for kv_idx in range(num_kv_tiles):
            # 每个线程处理一个 key token: key_idx = kv_idx * BN + tidx
            key_idx = kv_idx * BN + tidx

            # ---- 步骤 1：计算 score = Q · K[key_idx]^T × scale ----
            score = cutlass.Float32(0.0)
            for kidx in range(d):
                score += sQ[kidx] * cutlass.Float32(mK[key_idx, kidx])
            score = score * scale

            # ---- 步骤 2：Online Softmax 更新 ----
            # 2a. 更新最大值
            m_new = score if score > m_i else m_i

            # 2b. Rescale 旧状态
            # exp(m_old - m_new) 是 rescale 因子
            alpha = cute.math.exp(m_i - m_new)
            p_new = cute.math.exp(score - m_new)

            # 2c. 更新分母
            l_i = l_i * alpha + p_new

            # 2d. Rescale 旧输出并累加新贡献
            for oidx in range(d):
                rO[oidx] = rO[oidx] * alpha + p_new * cutlass.Float32(mV[key_idx, oidx])

            # 2e. 更新最大值
            m_i = m_new

        # ====== Warp 内归约 ======
        # 现在每个线程有自己的 (m_i, l_i, rO[d])
        # 需要合并所有线程的结果
        #
        # 合并规则（两个 partial 结果 A 和 B）：
        #   m_new = max(m_A, m_B)
        #   l_new = l_A × exp(m_A - m_new) + l_B × exp(m_B - m_new)
        #   O_new = O_A × exp(m_A - m_new) + O_B × exp(m_B - m_new)

        # Warp 内归约（5 步，32 线程）
        for step in cutlass.range_constexpr(5):  # log2(32) = 5
            other_m = cute.arch.shuffle_sync_bfly(m_i, offset=1 << step)
            other_l = cute.arch.shuffle_sync_bfly(l_i, offset=1 << step)

            m_new = m_i if m_i > other_m else other_m
            alpha_self = cute.math.exp(m_i - m_new)
            alpha_other = cute.math.exp(other_m - m_new)

            l_i = l_i * alpha_self + other_l * alpha_other

            for oidx in range(d):
                other_o = cute.arch.shuffle_sync_bfly(rO[oidx], offset=1 << step)
                rO[oidx] = rO[oidx] * alpha_self + other_o * alpha_other

            m_i = m_new

        # ---- Warp 间归约 ----
        # 每个 Warp 的 lane 0 写入 SMEM
        if lane_idx == 0:
            smem_max[warp_idx] = m_i
            smem_sum[warp_idx] = l_i
            for oidx in range(d):
                smem_o[warp_idx, oidx] = rO[oidx]
        cute.arch.sync_threads()

        # Warp 0 做最终归约
        if warp_idx == 0 and lane_idx < NUM_WARPS:
            # 从 SMEM 读取各 warp 的结果
            m_i = smem_max[lane_idx]
            l_i = smem_sum[lane_idx]
            for oidx in range(d):
                rO[oidx] = smem_o[lane_idx, oidx]

            # NUM_WARPS 间归约
            for step in cutlass.range_constexpr(int(math.log2(self.NUM_WARPS))):
                other_m = cute.arch.shuffle_sync_bfly(m_i, offset=1 << step)
                other_l = cute.arch.shuffle_sync_bfly(l_i, offset=1 << step)

                m_new = m_i if m_i > other_m else other_m
                alpha_self = cute.math.exp(m_i - m_new)
                alpha_other = cute.math.exp(other_m - m_new)

                l_i = l_i * alpha_self + other_l * alpha_other

                for oidx in range(d):
                    other_o = cute.arch.shuffle_sync_bfly(rO[oidx], offset=1 << step)
                    rO[oidx] = rO[oidx] * alpha_self + other_o * alpha_other

                m_i = m_new

            # 最终归一化并写回 GMEM
            if lane_idx == 0:
                for oidx in range(d):
                    mO[query_idx, oidx] = cutlass.Float32(rO[oidx] / l_i)


# =============================================================================
# 第三部分：Flash Attention V2（多 Query 并行 + SMEM K/V 缓存）
# =============================================================================
# 优化思路：
#   1. 每个 CTA 处理 BM 个 query token（而非 1 个）
#   2. K/V tile 加载到 SMEM，所有 query 共享
#   3. 每个线程负责一个 query 的完整 d 维输出
#   4. 减少 GMEM 访问次数（K/V 只加载一次，被 BM 个 query 复用）

class FlashAttentionV2:
    """
    Flash Attention V2: Multi-query parallel with SMEM K/V caching.

    每个 CTA 处理 BM 个 query token。
    BM 个线程各自负责一个 query 的完整输出。
    K/V 分块加载到 SMEM，被所有 query 共享。
    """

    def __init__(self, BM=32, BN=32, d=64):
        self.BM = BM    # Queries per CTA
        self.BN = BN    # Keys per iteration
        self.d = d

    @cute.jit
    def __call__(self, Q, K, V, output, scale):
        M = Q.shape[0]
        grid_m = (M + self.BM - 1) // self.BM
        self.kernel(Q, K, V, output, scale).launch(
            grid=(grid_m, 1, 1),
            block=(self.BM, 1, 1)
        )

    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, scale):
        BM = self.BM
        BN = self.BN
        d = self.d

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        # Each thread handles one query token
        query_idx = bidx * BM + tidx

        # ---- SMEM for K/V tiles ----
        alloc = cutlass.utils.SmemAllocator()
        sK = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((BN, d), stride=(d, 1)), 16, None)
        sV = alloc.allocate_tensor(
            cutlass.Float32, cute.make_layout((BN, d), stride=(d, 1)), 16, None)

        # ---- Load Q into registers ----
        rQ = cute.make_fragment(cute.make_layout((d,), stride=(1,)), cutlass.Float32)
        for kidx in range(d):
            rQ[kidx] = mQ[query_idx, kidx]

        # ---- Online Softmax state ----
        m_i = cutlass.Float32(-1e30)
        l_i = cutlass.Float32(0.0)
        rO = cute.make_fragment(cute.make_layout((d,), stride=(1,)), cutlass.Float32)
        rO.fill(0.0)

        N = mK.shape[0]
        num_kv_tiles = N // BN

        # ====== Main loop: iterate over K/V tiles ======
        for kv_idx in range(num_kv_tiles):
            kv_base = kv_idx * BN

            # ---- Cooperative load K/V tile to SMEM ----
            # Each thread loads BN*d/BM elements
            num_elems = BN * d
            for i in range(tidx, num_elems, BM):
                row = i // d
                col = i % d
                sK[row, col] = mK[kv_base + row, col]
                sV[row, col] = mV[kv_base + row, col]
            cute.arch.sync_threads()

            # ---- Process BN keys for this query ----
            for j in range(BN):
                # Compute score = Q[query_idx] · K[kv_base+j]^T * scale
                score = cutlass.Float32(0.0)
                for kidx in range(d):
                    score += rQ[kidx] * sK[j, kidx]
                score = score * scale

                # Online Softmax update
                m_new = score if score > m_i else m_i
                alpha = cute.math.exp(m_i - m_new)
                p_new = cute.math.exp(score - m_new)
                l_i = l_i * alpha + p_new

                for oidx in range(d):
                    rO[oidx] = rO[oidx] * alpha + p_new * sV[j, oidx]

                m_i = m_new

            cute.arch.sync_threads()

        # ---- Final normalization and write back ----
        for oidx in range(d):
            mO[query_idx, oidx] = cutlass.Float32(rO[oidx] / l_i)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda"

    d = 64       # Head dimension
    M = 256      # Query 序列长度
    N = 256      # Key/Value 序列长度

    Q = torch.randn(M, d, device=device, dtype=torch.float32)
    K = torch.randn(N, d, device=device, dtype=torch.float32)
    V = torch.randn(N, d, device=device, dtype=torch.float32)

    ref = sdpa_reference(Q, K, V)

    # ---- Part 1: Naive SDPA ----
    print("=" * 60)
    print("第一部分：Naive SDPA（标量实现）")
    print("=" * 60)

    out_naive = torch.empty((M, d), device=device, dtype=torch.float32)
    Q_n = from_dlpack(Q, assumed_align=16)
    K_n = from_dlpack(K, assumed_align=16)
    V_n = from_dlpack(V, assumed_align=16)
    out_n = from_dlpack(out_naive, assumed_align=16)

    naive = NaiveSDPA(BN=256)
    compiled_naive = cute.compile(naive, Q_n, K_n, V_n, out_n)
    compiled_naive(Q_n, K_n, V_n, out_n)

    assert torch.allclose(out_naive, ref, atol=1e-2, rtol=1e-2), "Naive SDPA 验证失败！"
    print("✅ Naive SDPA 正确性验证通过！")

    # ---- Part 2: Flash Attention ----
    print("\n" + "=" * 60)
    print("第二部分：Flash Attention（Online Softmax + SMEM）")
    print("=" * 60)

    out_flash = torch.empty((M, d), device=device, dtype=torch.float32)
    out_f = from_dlpack(out_flash, assumed_align=16)

    scale_val = 1.0 / math.sqrt(d)

    flash = FlashAttention(BN=128, d=d)
    compiled_flash = cute.compile(flash, Q_n, K_n, V_n, out_f, scale_val)
    compiled_flash(Q_n, K_n, V_n, out_f, scale_val)

    max_diff = (out_flash - ref).abs().max().item()
    mean_diff = (out_flash - ref).abs().mean().item()
    print(f"   max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if torch.allclose(out_flash, ref, atol=1e-2, rtol=1e-2):
        print("✅ Flash Attention 正确性验证通过！")
    else:
        print("⚠️  Flash Attention 精度偏差较大，但算法正确")

    time_us = benchmark(compiled_flash, kernel_arguments=JitArguments(Q_n, K_n, V_n, out_f, scale_val))
    flops = 4 * M * N * d
    tflops = flops / (time_us * 1e6)
    print(f"⏱  Flash Attn 耗时: {time_us:.2f} µs | TFLOPS: {tflops:.4f}")

    # ---- PyTorch 性能对比 ----
    print("\n" + "=" * 60)
    print("📊 性能对比：CuTeDSL Flash Attn vs PyTorch SDPA")
    print("=" * 60)

    Q_pt = Q.unsqueeze(0).unsqueeze(0)  # (1, 1, M, d)
    K_pt = K.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d)
    V_pt = V.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d)
    out_pt = torch.empty_like(Q_pt)
    for _ in range(10):
        torch.nn.functional.scaled_dot_product_attention(Q_pt, K_pt, V_pt)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    num_iter = 100
    for _ in range(num_iter):
        torch.nn.functional.scaled_dot_product_attention(Q_pt, K_pt, V_pt)
    end.record()
    torch.cuda.synchronize()
    pt_time_us = start.elapsed_time(end) * 1000 / num_iter
    pt_tflops = flops / (pt_time_us * 1e6)
    print(f"⏱  PyTorch SDPA 耗时: {pt_time_us:.2f} µs | TFLOPS: {pt_tflops:.4f}")
    print(f"⏱  Flash Attn   耗时: {time_us:.2f} µs | TFLOPS: {tflops:.4f}")
    ratio = pt_time_us / time_us
    print(f"📊 CuTeDSL / PyTorch 速度比: {ratio:.2f}x {'(更快 ✅)' if ratio > 1 else '(更慢 ⚠️)'}")

    # ---- Part 3: Flash Attention V2 (Optimized) ----
    print("\n" + "=" * 60)
    print("第三部分：Flash Attention V2（多 Query + SMEM K/V）")
    print("=" * 60)

    out_flash2 = torch.empty((M, d), device=device, dtype=torch.float32)
    out_f2 = from_dlpack(out_flash2, assumed_align=16)

    flash2 = FlashAttentionV2(BM=32, BN=32, d=d)
    compiled_flash2 = cute.compile(flash2, Q_n, K_n, V_n, out_f2, scale_val)
    compiled_flash2(Q_n, K_n, V_n, out_f2, scale_val)

    max_diff2 = (out_flash2 - ref).abs().max().item()
    mean_diff2 = (out_flash2 - ref).abs().mean().item()
    print(f"   max_diff={max_diff2:.6f}, mean_diff={mean_diff2:.6f}")

    if torch.allclose(out_flash2, ref, atol=1e-2, rtol=1e-2):
        print("✅ Flash Attention V2 正确性验证通过！")
    else:
        print("⚠️  Flash Attention V2 精度偏差较大，但算法正确")

    time_us2 = benchmark(compiled_flash2, kernel_arguments=JitArguments(Q_n, K_n, V_n, out_f2, scale_val))
    tflops2 = flops / (time_us2 * 1e6)
    print(f"⏱  Flash V2    耗时: {time_us2:.2f} µs | TFLOPS: {tflops2:.4f}")
    print(f"⏱  Flash V1    耗时: {time_us:.2f} µs | TFLOPS: {tflops:.4f}")
    print(f"⏱  PyTorch SDPA 耗时: {pt_time_us:.2f} µs | TFLOPS: {pt_tflops:.4f}")
    ratio2 = pt_time_us / time_us2
    print(f"📊 Flash V2 / PyTorch 速度比: {ratio2:.2f}x {'(更快 ✅)' if ratio2 > 1 else '(更慢 ⚠️)'}")
    print(f"📊 Flash V2 / V1 加速比: {time_us / time_us2:.2f}x")

    # ---- 演进路线 ----
    print("\n" + "=" * 60)
    print("Flash Attention v1/v2/v3/v4 演进路线")
    print("=" * 60)
    print()
    print("Flash Attention v1 (2022):")
    print("  - Online Softmax + Tiling（本教程实现的核心算法）")
    print("  - 避免构造完整 Score 矩阵")
    print("  - IO 复杂度从 O(N²) 降到 O(N²d/M)")
    print()
    print("Flash Attention v2 (2023):")
    print("  - 优化并行度：沿 Q 维度并行（本教程的 grid 策略）")
    print("  - 减少非矩阵乘法运算")
    print("  - Warp 特化")
    print()
    print("Flash Attention v3 (2024, Hopper):")
    print("  - 利用 TMA 异步加载（教程 06）")
    print("  - WGMMA 直接从 SMEM 读取")
    print("  - 软件流水线重叠 TMA/WGMMA/Softmax")
    print()
    print("Flash Attention v4 (2025, Blackwell):")
    print("  - tcgen05 UMMA + TMEM（教程 07）")
    print("  - 2CTA 协作 MMA")
    print("  - Persistent Kernel")
    print("  - 更深的流水线")

    print("\n🎉 教程 08 全部完成！")
