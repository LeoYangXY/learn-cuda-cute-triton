"""
=============================================================================
CuTeDSL 复现 origin_cuda_kernel/embedding/embedding.cu
=============================================================================

原生 CUDA 有 2 个版本：
  1. embedding_float4 — 每个 block 负责 output 的一行，每个 thread 用 float4 搬 4 个 float
  2. embedding_pack   — 和 float4 一样，只是用 LDST128BITS 宏

两者本质相同，CuTeDSL 中用一个 128-bit copy_atom 即可覆盖。

语义：
  输入：idx[n] (int32), weight[vocab_size, emb_size] (float32)
  输出：output[n, emb_size]，其中 output[i] = weight[idx[i]]
  每个 Block 负责 output 的一行（即一个 idx 对应的 embedding 向量）
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch

EMB_VEC = 4  # 每线程搬 4 个 float = 128 bits

# 语义：
# weight = [[0.1, 0.2, 0.3],   # 词 0
#            [0.4, 0.5, 0.6],   # 词 1
#            [0.7, 0.8, 0.9],   # 词 2
#            [1.0, 1.1, 1.2]]   # 词 3

# indices = [3, 0, 2]

# output = [[1.0, 1.1, 1.2],   # weight[3]
#            [0.1, 0.2, 0.3],   # weight[0]
#            [0.7, 0.8, 0.9]]   # weight[2]


# =============================================================================
# 版本 1: 向量化 embedding lookup（对应 embedding_float4 和 embedding_pack）
# =============================================================================
# 原生 CUDA：
#   __global__ void embedding_float4(const int *idx, float *weight, float *output, int n, int emb_size) {
#       int i = blockIdx.x;           // 第 i 个 token
#       int row = idx[i];             // 查表得到 weight 的行号
#       int col = threadIdx.x * 4;    // 每个 thread 负责 4 个连续 float
#       float4* src = reinterpret_cast<float4*>(weight + row * emb_size + col);
#       float4* dst = reinterpret_cast<float4*>(output + i * emb_size + col);
#       *dst = *src;
#   }
#
# CuTeDSL 版本：
#   - 每个 Block 对应一个 token（blockIdx.x = token index）
#   - 用 TiledCopy 做 128-bit 向量化搬运

@cute.kernel
def embedding_kernel(
    gWeight: cute.Tensor,    # weight[vocab_size, emb_size]
    gOutput: cute.Tensor,    # output[n, emb_size]
    gIdx: cute.Tensor,       # idx[n], int32
    tv_layout: cute.Layout,
    tiler: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()  # bidx = token index

    # 原生 CUDA 等价：int row = idx[bidx];
    row = gIdx[bidx]

    # ── Step 1: 切出一行 ──────────────────────────────────────────
    # local_tile(tensor, tile_shape, coord) = 按 tile_shape 切块，取第 coord 块
    #
    # gWeight shape = (vocab_size, emb_size)
    # tile_shape    = (1, emb_size)   ← 每块是一整行
    # coord         = (row, 0)        ← 行方向取第 row 块，列方向取第 0 块
    # 结果 src_row shape = (1, emb_size)
    #
    # 注意：tiler = (emb_size,) 是从 host 传入的 tuple，tiler[0] = emb_size
    src_row = cute.local_tile(gWeight, (1, tiler[0]), (row, 0))
    dst_row = cute.local_tile(gOutput, (1, tiler[0]), (bidx, 0))

    # ── Step 2: reshape (1, emb_size) → (emb_size,) ──────────────
    # 为什么要 reshape?
    #   local_tile 切出来的是 2D: (1, emb_size)
    #   但下面的 tiled_copy 的 tv_layout 是 1D 的: (threads_per_block, EMB_VEC)
    #   tiled_copy.partition_S/D 要求输入 tensor 的维度和 tv_layout 匹配
    #   所以必须把多余的那个维度 1 去掉，变成纯 1D 的 (emb_size,)
    #
    # 做法：取 src_row 的起始地址(.iterator)，套上新的 1D layout
    # 等价于 PyTorch 的 src_row.reshape(-1)
    src_1d = cute.make_tensor(src_row.iterator, cute.make_layout((tiler[0],)))
    dst_1d = cute.make_tensor(dst_row.iterator, cute.make_layout((tiler[0],)))

    # ── Step 3: 构建 tiled_copy ─────────────────────────────────
    # copy_atom: 一条指令搬 128 bits = 4 个 float
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gWeight.element_type,
        num_bits_per_copy=gWeight.element_type.width * EMB_VEC,  # 32 × 4 = 128 bits
    )

    # tv_layout 是从 host 传入的，shape = (256, 4), stride = (4, 1)
    # 和 add.py 的 FP32 向量化版完全一样的模式：
    #
    # tv_layout 映射表（以 emb_size=1024 为例）：
    #                        v_idx
    #               ╭  0  ──>  (  0,    1,    2,    3  )   ← thread0 搬 float[0:4]
    #               │  1  ──>  (  4,    5,    6,    7  )   ← thread1 搬 float[4:8]
    #               │  2  ──>  (  8,    9,   10,   11  )
    # t_idx        <   :          :     :     :     :
    #               │  :          :     :     :     :
    #               │254  ──>  (1016, 1017, 1018, 1019 )
    #               ╰255  ──>  (1020, 1021, 1022, 1023 )
    #
    # 每行 = 1 个 thread 负责 4 个连续 float → 1 条 LDG.128 指令
    # 256 个 thread × 4 = 1024 = emb_size，刚好覆盖一整行
    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
    thr_copy = tiled_copy.get_slice(tidx)

    tSrc = thr_copy.partition_S(src_1d)
    tDst = thr_copy.partition_D(dst_1d)

    # 128-bit copy：weight[row][col:col+4] → output[bidx][col:col+4]
    cute.copy(tiled_copy, tSrc, tDst)#本质上就是每个 thread 一条 LDG.128 直接从 global 搬到寄存器，再一条 STG.128 写出去


@cute.jit
def embedding_lookup(mIdx: cute.Tensor, mWeight: cute.Tensor, mOutput: cute.Tensor):
    n = mOutput.shape[0]
    emb_size = mOutput.shape[1]
    threads_per_block = emb_size // EMB_VEC  # emb_size / 4

    tv_layout = cute.make_layout(
        (threads_per_block, EMB_VEC),
        stride=(EMB_VEC, 1)
    )
    tiler = (emb_size,)

    embedding_kernel(mWeight, mOutput, mIdx, tv_layout, tiler).launch(
        grid=(n, 1, 1), block=(threads_per_block, 1, 1))


# =============================================================================
# 对比直接 load/store 与 cp.async 与 TMA
# =============================================================================
#
# ── 1. 为什么 cp.async (Global → Shared) 对 embedding 没有收益 ──────────
#
# Shared Memory 的存在意义是"数据复用"：多个线程要反复读同一份数据时，
# 先搬一份到 smem，大家共享读，避免重复访问 global memory。
#
#   典型的 GEMM（有复用）:
#     线程 A 需要 weight[0][0:4]
#     线程 B 也需要 weight[0][0:4]    ← 同一份数据被多个线程读
#     线程 C 也需要 weight[0][0:4]
#     → 搬一次到 smem，大家共享读，省了 N 次 global 访问 ✅
#
#   Embedding（无复用）:
#     thread 0 读 weight[row][0:4]   → 写到 output[bidx][0:4]   → 再也不碰了
#     thread 1 读 weight[row][4:8]   → 写到 output[bidx][4:8]   → 再也不碰了
#     ...
#     → 每个 float 只被 1 个线程读 1 次，写 1 次，没有任何复用 ❌
#
# 如果强行走 smem，反而多一跳：
#
#   直接 load/store（当前实现）:
#     Global ──LDG.128──> Register ──STG.128──> Global
#         1 次读              1 次写           = 2 次访存
#
#   经 smem（cp.async）:
#     Global ──cp.async──> Shared ──LDS.128──> Register ──STG.128──> Global
#         1 次读           1 次读     暂存       1 次写    = 3 次访存 + syncthreads 开销
#
# 多了一跳（smem → register），还多了 syncthreads() 的同步等待，纯粹浪费。
#
#
# ── 2. 为什么 TMA 不适合 embedding ──────────────────────────────────────
#
# TMA 的 tensor map 描述的是一个规整的、连续的张量区域。
#
#   TMA 擅长的场景（GEMM 的 A 矩阵）：
#     Block 0 读 A[0:128, 0:64]       ← 连续的矩形区域
#     Block 1 读 A[128:256, 0:64]     ← 紧接着的连续矩形区域
#     → 每个 block 的 tile 在内存中是连续的、可预测的
#     → TMA 硬件一条指令搬整个 tile ✅
#
#   Embedding（随机行访问）：
#     Block 0: idx[0] = 7291   → 读 weight[7291, :]
#     Block 1: idx[1] = 42     → 读 weight[42, :]
#     Block 2: idx[2] = 15603  → 读 weight[15603, :]
#     → 每个 block 读的行在内存中完全随机！❌
#
# TMA 的 tensor map 只能描述"从坐标 (x, y) 开始切一个 tile"，假设你是按顺序
# 遍历张量的某个区域。embedding 的行号由 idx[i] 决定，完全随机。
#
# 虽然理论上可以把 (row, 0) 作为 TMA 坐标，但：
#   - 每个 block 只读 1 行，TMA 启动开销（描述符解析 + mbarrier 同步）
#     比直接 LDG.128 还大
#   - TMA 的优势是搬大块连续数据（如 128×64 的 tile），搬 1 行是杀鸡用牛刀
#
#
# ── 3. 为什么当前实现已经是最优 ─────────────────────────────────────────
#
#   当前路径：每个 thread 做 1 条 LDG.128 + 1 条 STG.128
#
#     thread 0:   LDG.128 weight[row][0:4]        → 寄存器 → STG.128 output[bidx][0:4]
#     thread 1:   LDG.128 weight[row][4:8]        → 寄存器 → STG.128 output[bidx][4:8]
#     ...
#     thread 255: LDG.128 weight[row][1020:1024]  → 寄存器 → STG.128 output[bidx][1020:1024]
#
#   ┌──────────────┬──────────────────────────────────────────────┐
#   │ 指标         │ 分析                                         │
#   ├──────────────┼──────────────────────────────────────────────┤
#   │ 指令数       │ 每个 thread 只有 2 条访存指令，不能更少了      │
#   │ 访存宽度     │ 128-bit 已经是单条指令最大宽度                 │
#   │ 无同步       │ 线程之间完全独立，不需要 syncthreads           │
#   │ 无 smem      │ 不占 shared memory，launch occupancy 更高     │
#   │ 合并访存     │ 同一 warp 的 32 个线程读连续 512 bytes，完美   │
#   └──────────────┴──────────────────────────────────────────────┘
#
# 唯一的瓶颈是 global memory 带宽（embedding 就是纯搬数据），当前实现已经把
# 带宽打满了。任何加 smem 或 TMA 的中间层都只会增加延迟，不会提高带宽利用率。
#
# 结论：Shared Memory / cp.async / TMA 解决的是"数据复用"和"搬运与计算重叠"
# 的问题。Embedding 既没有复用，也没有计算可重叠，所以最短路径
# （LDG → 寄存器 → STG）就是最优解。
# =============================================================================


# =============================================================================
# 测试
# =============================================================================
if __name__ == "__main__":
    n = 1024
    vocab_size = 32768
    emb_size = 1024

    print("=" * 60)
    print(f"CuTeDSL Embedding Lookup (n={n}, vocab={vocab_size}, emb={emb_size})")
    print("=" * 60)

    weight = torch.randn(vocab_size, emb_size, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, vocab_size, (n,), device="cuda", dtype=torch.int32)
    output = torch.empty(n, emb_size, device="cuda", dtype=torch.float32)

    ref = weight[idx.long()]

    idx_ = from_dlpack(idx, assumed_align=16)
    weight_ = from_dlpack(weight, assumed_align=16)
    output_ = from_dlpack(output, assumed_align=16)

    compiled = cute.compile(embedding_lookup, idx_, weight_, output_)
    compiled(idx_, weight_, output_)

    assert torch.allclose(output, ref, atol=1e-5), f"验证失败！max_diff={( output - ref).abs().max().item()}"
    print("✅ Embedding Lookup 正确")

    t = benchmark(compiled, kernel_arguments=JitArguments(idx_, weight_, output_))
    print(f"   耗时: {t:.2f} µs")
