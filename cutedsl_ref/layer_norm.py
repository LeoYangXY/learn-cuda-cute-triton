"""
=============================================================================
CuTeDSL 复现 origin_cuda_kernel/layer_norm/layer_norm.cu
=============================================================================

原生 CUDA 有 2 个版本：
  1. layer_norm_f32_kernel   — 每个 thread 处理 1 个元素
  2. layer_norm_float4_kernel — 每个 thread 处理 4 个元素（float4）

两者都是：一个 block 负责一行（K 个元素），用 block_reduce_sum 求均值和方差。

CuTeDSL 实现：
  版本 1: 每 thread 1 个元素（对应 layer_norm_f32_kernel）
  版本 2: 每 thread 4 个元素（对应 layer_norm_float4_kernel）

语义：
  输入 x[N, K]，标量 gamma (g) 和 beta (b)
  输出 y[N, K]
  y[i][j] = g * (x[i][j] - mean_i) / sqrt(var_i + eps) + b
  其中 mean_i 和 var_i 是第 i 行的均值和方差
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch
import math

WARP_SIZE = 32

# =============================================================================
# 工具：warp reduce sum（用 butterfly shuffle）
# =============================================================================
# 原生 CUDA 等价：
#   for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
#       val += __shfl_xor_sync(0xffffffff, val, mask);
@cute.jit
def warp_reduce_sum(val, width=WARP_SIZE):
    for i in cutlass.range_constexpr(int(math.log2(width))):
        val = val + cute.arch.shuffle_sync_bfly(val, offset=1 << i)
    return val


# =============================================================================
# 工具：block reduce sum（warp shuffle + shared memory）
# =============================================================================
# 原生 CUDA 等价：
#   val = warp_reduce_sum(val);
#   if (lane == 0) shared[warp] = val;
#   __syncthreads();
#   val = (lane < NUM_WARPS) ? shared[lane] : 0;
#   val = warp_reduce_sum<NUM_WARPS>(val);
#   shared[0] = val; __syncthreads();
#   return shared[0];
@cute.jit
def block_reduce_sum(val, sdata: cute.Tensor, num_warps: int, tidx):
    warp_idx = tidx // WARP_SIZE
    lane_idx = tidx % WARP_SIZE

    # 第 1 层：warp 内 shuffle 规约
    val = warp_reduce_sum(val)

    # warp leader 写 shared memory
    if lane_idx == 0:
        sdata[warp_idx] = val

    cute.arch.sync_threads()

    # 第 2 层：跨 warp 规约（用第 0 个 warp）
    if warp_idx == 0:
        val2 = sdata[lane_idx] if lane_idx < num_warps else cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(int(math.log2(WARP_SIZE))):
            val2 = val2 + cute.arch.shuffle_sync_bfly(val2, offset=1 << i)
        if lane_idx == 0:
            sdata[0] = val2

    cute.arch.sync_threads()
    return sdata[0]


# =============================================================================
# 版本 1: 每 thread 1 个元素
# =============================================================================
# 原生 CUDA：
#   int row = blockIdx.x; int col = threadIdx.x;
#   float val = x[row * K + col];
#   float sum = block_reduce_sum(val);
#   float mean = sum / K;
#   float var_val = (val - mean) * (val - mean);
#   float var_sum = block_reduce_sum(var_val);
#   float inv_std = rsqrtf(var_sum / K + 1e-5f);
#   y[row * K + col] = (val - mean) * inv_std * g + b;

NUM_THREADS_V1 = 256

@cute.kernel
def layer_norm_f32_kernel(
    gX: cute.Tensor, gY: cute.Tensor,
    g: cutlass.Float32, b: cutlass.Float32,
    smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    K = gX.shape[1]

    # 原生 CUDA 等价：float val = x[row * K + col];
    val = gX[row, tidx]

    # 第 1 次 block reduce：求和 → 均值
    sum_val = block_reduce_sum(val, sdata, NUM_THREADS_V1 // WARP_SIZE, tidx)
    mean = sum_val / cutlass.Float32(K)

    # 第 2 次 block reduce：求方差
    var_val = (val - mean) * (val - mean)
    var_sum = block_reduce_sum(var_val, sdata, NUM_THREADS_V1 // WARP_SIZE, tidx)
    inv_std = cute.math.rsqrt(var_sum / cutlass.Float32(K) + cutlass.Float32(1e-5))

    # 原生 CUDA 等价：y[row * K + col] = (val - mean) * inv_std * g + b;
    gY[row, tidx] = (val - mean) * inv_std * g + b


@cute.jit
def layer_norm_f32(mX: cute.Tensor, mY: cute.Tensor, g: float, b: float):
    N = mX.shape[0]
    smem_layout = cute.make_layout((NUM_THREADS_V1 // WARP_SIZE,))
    layer_norm_f32_kernel(mX, mY, cutlass.Float32(g), cutlass.Float32(b), smem_layout).launch(
        grid=(N, 1, 1), block=(NUM_THREADS_V1, 1, 1))


# =============================================================================
# 版本 2: 每 thread 4 个元素（float4 风格）
# =============================================================================
# 原生 CUDA：
#   float4 v = FLOAT4(x[row * K + col_start]);
#   float sum_of_thread = v.x + v.y + v.z + v.w;
#   float sum_of_block = block_reduce_sum(sum_of_thread);
#   ...（每个分量各自归一化）
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │          手动展开 vs CuTe TiledCopy 风格 对比                           │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │  ■ 手动展开（本版本 V2 的写法，对应原生 CUDA float4）                    │
# │                                                                         │
# │    col_start = tidx * 4                                                 │
# │    v0 = gX[row, col_start]                                              │
# │    v1 = gX[row, col_start + 1]       # 4 次标量索引                     │
# │    v2 = gX[row, col_start + 2]       # 编译器可能合并为 LDG.128         │
# │    v3 = gX[row, col_start + 3]       # 但不保证！可能是 4 条 LDG.32     │
# │    sum_thread = v0 + v1 + v2 + v3                                       │
# │                                                                         │
# │    优点: 代码简洁，1:1 对应原生 CUDA                                     │
# │    缺点: 向量化不保证；K 变大需要手动加循环；线程映射硬编码 (tidx * 4)    │
# │                                                                         │
# │  ■ CuTe TiledCopy 风格（版本 V3 的写法）                                │
# │                                                                         │
# │    # 1. 定义 copy atom：显式指定 128-bit 向量化                          │
# │    copy_atom = cute.make_copy_atom(                                      │
# │        cute.nvgpu.CopyUniversalOp(), dtype,                              │
# │        num_bits_per_copy=dtype.width * 4,  # 128 bits → 保证 LDG.128    │
# │    )                                                                     │
# │                                                                         │
# │    # 2. 定义 tv_layout：线程→数据的映射关系                               │
# │    #    tv_layout = ((256, 4), (4, 1))                                   │
# │    #    含义: 256 个线程，每个线程负责 4 个连续元素                        │
# │    #    等价于手动写的 tidx * 4, 但由框架管理                             │
# │    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)        │
# │    thr_copy = tiled_copy.get_slice(tidx)                                 │
# │                                                                         │
# │    # 3. partition 自动按 tv_layout 分配每个线程的源/目标                   │
# │    tSrc = thr_copy.partition_S(blk)                                      │
# │    rData = cute.make_fragment_like(tSrc)  # 寄存器 fragment              │
# │                                                                         │
# │    # 4. 一条 cute.copy 保证生成 LDG.128                                  │
# │    cute.copy(tiled_copy, tSrc, rData)                                    │
# │                                                                         │
# │    # 5. 用 size(rData) 遍历，不需要知道具体有几个元素                     │
# │    for vi in range(cute.size(rData)):                                    │
# │        partial_sum += rData[vi]                                          │
# │                                                                         │
# │    优点: 保证向量化；tv_layout 可换配置；配合 local_tile 天然支持多 tile   │
# │    缺点: setup 代码多几行                                                 │
# │                                                                         │
# │  ■ 总结                                                                  │
# │    - K 小且固定 → 手动展开更直接（V2）                                    │
# │    - K 大/需通用 → TiledCopy 更好（V3），向量化有保证，多 tile 自然扩展    │
# │    - 本质区别: 手动写法把 tv_layout 的逻辑 inline 到了 kernel 里          │
# │      (tidx*4)，CuTe 风格把映射外化成数据结构，框架自动处理                │
# └─────────────────────────────────────────────────────────────────────────┘

NUM_THREADS_V2 = 64  # K/4 个线程（K=256 → 64 线程）

@cute.kernel
def layer_norm_float4_kernel(
    gX: cute.Tensor, gY: cute.Tensor,
    g: cutlass.Float32, b: cutlass.Float32,
    smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    K = gX.shape[1]
    col_start = tidx * 4

    # 原生 CUDA 等价：float4 v = FLOAT4(x[row * K + col_start]);
    v0 = gX[row, col_start]
    v1 = gX[row, col_start + 1]
    v2 = gX[row, col_start + 2]
    v3 = gX[row, col_start + 3]

    # 原生 CUDA 等价：float sum_of_thread = v.x + v.y + v.z + v.w;
    sum_thread = v0 + v1 + v2 + v3
    sum_block = block_reduce_sum(sum_thread, sdata, NUM_THREADS_V2 // WARP_SIZE, tidx)
    mean = sum_block / cutlass.Float32(K)

    # 方差
    var_thread = (v0 - mean) * (v0 - mean) + (v1 - mean) * (v1 - mean) + \
                 (v2 - mean) * (v2 - mean) + (v3 - mean) * (v3 - mean)
    var_sum = block_reduce_sum(var_thread, sdata, NUM_THREADS_V2 // WARP_SIZE, tidx)
    inv_std = cute.math.rsqrt(var_sum / cutlass.Float32(K) + cutlass.Float32(1e-5))

    # 归一化并写回
    gY[row, col_start]     = (v0 - mean) * inv_std * g + b
    gY[row, col_start + 1] = (v1 - mean) * inv_std * g + b
    gY[row, col_start + 2] = (v2 - mean) * inv_std * g + b
    gY[row, col_start + 3] = (v3 - mean) * inv_std * g + b


@cute.jit
def layer_norm_float4(mX: cute.Tensor, mY: cute.Tensor, g: float, b: float):
    N = mX.shape[0]
    num_warps = max(NUM_THREADS_V2 // WARP_SIZE, 1)
    smem_layout = cute.make_layout((num_warps,))
    layer_norm_float4_kernel(mX, mY, cutlass.Float32(g), cutlass.Float32(b), smem_layout).launch(
        grid=(N, 1, 1), block=(NUM_THREADS_V2, 1, 1))


# =============================================================================
# 版本 3: K 较大时用 TiledCopy + 多 tile 循环（CuTe 风格）
# =============================================================================
# 当 K 远大于线程数时（比如 K=4096, 线程数=256），每个线程需要处理多个 tile。
# 这时候就需要用 CuTe 的 tiled API 来管理搬运了。
#
# 思路：
#   1. 把一行 K 个元素按 TILE_SIZE = threads * VEC 切成 num_tiles 块
#   2. 第一遍循环：每个 tile 用 TiledCopy (LDG.128) 搬到寄存器，累加 partial_sum
#   3. block_reduce_sum 得到均值
#   4. 第二遍循环：重新读每个 tile，累加 partial_var
#   5. block_reduce_sum 得到方差
#   6. 第三遍循环：重新读每个 tile，归一化后用 TiledCopy (STG.128) 写回
#
# 注意：这里读了 3 遍 global memory。如果 K 不是特别大（能塞进寄存器），
# 可以第一遍就把数据缓存在寄存器里，后面复用。下面实现的是通用版本（3 遍读）。
#
# 原生 CUDA 等价伪代码：
#   // Pass 1: sum
#   float partial_sum = 0;
#   for (int tile = 0; tile < num_tiles; tile++) {
#       int col = tile * TILE_SIZE + threadIdx.x * 4;
#       float4 v = *(float4*)(x + row * K + col);
#       partial_sum += v.x + v.y + v.z + v.w;
#   }
#   float mean = block_reduce_sum(partial_sum) / K;
#
#   // Pass 2: var
#   float partial_var = 0;
#   for (int tile = 0; tile < num_tiles; tile++) { ... }
#   float inv_std = rsqrtf(block_reduce_sum(partial_var) / K + eps);
#
#   // Pass 3: normalize + write
#   for (int tile = 0; tile < num_tiles; tile++) {
#       float4 v = *(float4*)(x + row * K + col);
#       v.x = (v.x - mean) * inv_std * g + b; ...
#       *(float4*)(y + row * K + col) = v;
#   }

#先思考任务划分：一个block负责一行。然后再思考每次处理的tile：每次处理1024个元素的tile，然后再在一个tile中思考每个thread要处理多少元素


NUM_THREADS_V3 = 256
VEC_V3 = 4  # 每线程 4 个 float = 128 bits
TILE_SIZE_V3 = NUM_THREADS_V3 * VEC_V3  # 1024 elements per tile
K_V3 = 4096  # 编译时确定的 K
NUM_TILES_V3 = K_V3 // TILE_SIZE_V3  # 4096 / 1024 = 4

@cute.kernel
def layer_norm_tiled_kernel(
    gX: cute.Tensor, gY: cute.Tensor,
    g: cutlass.Float32, b: cutlass.Float32,
    smem_layout: cute.Layout,
    tv_layout: cute.Layout,
    tiler: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    K = gX.shape[1]

    # ── 构建 TiledCopy (LDG.128) ──
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gX.element_type,
        num_bits_per_copy=gX.element_type.width * VEC_V3,  # 128 bits
    )
    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
    thr_copy = tiled_copy.get_slice(tidx)

    # ── 取出当前行（reshape 成 1D）──
    # gX shape = (N, K)，取第 row 行变成 (K,)
    # 用 local_tile 切出 (1, K) 的子张量，再 reshape 成 (K,)
    src_row = cute.local_tile(gX, (1, K), (row, 0))
    dst_row = cute.local_tile(gY, (1, K), (row, 0))
    row_x = cute.make_tensor(src_row.iterator, cute.make_layout((K,)))
    row_y = cute.make_tensor(dst_row.iterator, cute.make_layout((K,)))

    # ======== Pass 1: 多 tile 循环求和 → 均值 ========
    # 每个线程累加自己负责的所有 tile 的 partial sum
    partial_sum = cutlass.Float32(0.0)
    for tile_idx in cutlass.range_constexpr(NUM_TILES_V3):
        # local_tile 切出第 tile_idx 个 tile，shape = (TILE_SIZE,)
        blk = cute.local_tile(row_x, tiler, (tile_idx,))
        tSrc = thr_copy.partition_S(blk)

        # LDG.128 → 寄存器
        rData = cute.make_fragment_like(tSrc)
        cute.copy(tiled_copy, tSrc, rData)

        # 累加：每个线程持有 VEC_V3 个值，逐个加
        for vi in range(cute.size(rData)):
            partial_sum = partial_sum + rData[vi]

    # block reduce → 全行求和 → 均值
    sum_val = block_reduce_sum(partial_sum, sdata, NUM_THREADS_V3 // WARP_SIZE, tidx)
    mean = sum_val / cutlass.Float32(K)

    # ======== Pass 2: 多 tile 循环求方差 ========
    partial_var = cutlass.Float32(0.0)
    for tile_idx in cutlass.range_constexpr(NUM_TILES_V3):
        blk = cute.local_tile(row_x, tiler, (tile_idx,))
        tSrc = thr_copy.partition_S(blk)

        rData = cute.make_fragment_like(tSrc)
        cute.copy(tiled_copy, tSrc, rData)

        for vi in range(cute.size(rData)):
            diff = rData[vi] - mean
            partial_var = partial_var + diff * diff

    var_sum = block_reduce_sum(partial_var, sdata, NUM_THREADS_V3 // WARP_SIZE, tidx)
    inv_std = cute.math.rsqrt(var_sum / cutlass.Float32(K) + cutlass.Float32(1e-5))

    # ======== Pass 3: 多 tile 循环归一化 + 写回 ========
    for tile_idx in cutlass.range_constexpr(NUM_TILES_V3):
        blk_x = cute.local_tile(row_x, tiler, (tile_idx,))
        blk_y = cute.local_tile(row_y, tiler, (tile_idx,))

        tSrc = thr_copy.partition_S(blk_x)
        tDst = thr_copy.partition_D(blk_y)

        # 读
        rData = cute.make_fragment_like(tSrc)
        cute.copy(tiled_copy, tSrc, rData)

        # 归一化
        rOut = cute.make_fragment_like(rData)
        for vi in range(cute.size(rData)):
            rOut[vi] = (rData[vi] - mean) * inv_std * g + b

        # 写回 (STG.128)
        cute.copy(tiled_copy, rOut, tDst)


@cute.jit
def layer_norm_tiled(mX: cute.Tensor, mY: cute.Tensor, g: float, b: float):
    N = mX.shape[0]

    tv_layout = cute.make_layout(
        (NUM_THREADS_V3, VEC_V3),   # (256, 4)
        stride=(VEC_V3, 1)          # (4, 1) — 每线程 4 个连续 float
    )
    tiler = (TILE_SIZE_V3,)  # (1024,)

    num_warps = NUM_THREADS_V3 // WARP_SIZE
    smem_layout = cute.make_layout((num_warps,))

    layer_norm_tiled_kernel(
        mX, mY, cutlass.Float32(g), cutlass.Float32(b),
        smem_layout, tv_layout, tiler
    ).launch(grid=(N, 1, 1), block=(NUM_THREADS_V3, 1, 1))


# =============================================================================
# 版本 4: 寄存器缓存 —— 只读 1 次 GMEM，省掉 2 次访存
# =============================================================================
# 版本 3 读了 3 遍 GMEM（Pass1 求和 + Pass2 方差 + Pass3 归一化）。
# 当每线程负责的数据量不大时（比如 4 tiles × 4 values = 16 个 float = 64 字节），
# 完全可以全部缓存在寄存器里，后续复用。
#
# 寄存器预算：
#   每线程 NUM_TILES × VEC 个 float = 4 × 4 = 16 个 float = 16 个寄存器
#   SM120 每线程最多 255 个寄存器，16 个完全没问题
#   即使 K=16384, NUM_TILES=16, 也只用 64 个寄存器，很安全
#
# 访存对比（理论）：
#   版本 3: 3 次 LDG + 1 次 STG = 4 次 GMEM 访问（每个元素）
#   版本 4: 1 次 LDG + 1 次 STG = 2 次 GMEM 访问（每个元素）→ 省 50% 带宽
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ⚠️ 为什么实测 V4 并没有比 V3 快？—— L1 Cache 的隐藏作用              │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │  RTX 5050 (SM120) 的缓存层次：                                          │
# │    寄存器:   ~4 cycles    每线程 255 个 (32-bit)                         │
# │    L1 cache: ~28 cycles   每 SM ~128 KB                                 │
# │    L2 cache: ~200 cycles  全局 32 MB                                    │
# │    DRAM:     ~400 cycles  显存                                          │
# │                                                                         │
# │  Layer Norm 每个 block 处理一行：                                        │
# │    K=4096 × 4B = 16 KB  <<<  L1 的 128 KB                              │
# │                                                                         │
# │  V3 的 3 遍读实际发生了什么：                                            │
# │    Pass 1: LDG.128 → L1 miss → 从 L2/DRAM 加载 → 16KB 进入 L1          │
# │    Pass 2: LDG.128 → L1 HIT！同一行数据还在 L1 里 → ~28 cycles          │
# │    Pass 3: LDG.128 → L1 HIT！→ ~28 cycles                              │
# │                                                                         │
# │  V4 的寄存器缓存：                                                      │
# │    Pass 1: LDG.128 → 数据存入寄存器                                     │
# │    Pass 2: 从寄存器读 → ~4 cycles                                       │
# │    Pass 3: 从寄存器读 → ~4 cycles                                       │
# │                                                                         │
# │  省掉的只是 L1 (~28 cycles) vs 寄存器 (~4 cycles) 的差距，              │
# │  而不是 DRAM (~400 cycles) vs 寄存器 (~4 cycles) 的差距！                │
# │  再加上 V4 额外的寄存器管理开销（更多 fragment 分配、tuple 索引），       │
# │  反而可能略慢。                                                          │
# │                                                                         │
# │  什么时候 V4 会真正快过 V3？                                             │
# │    当一行数据 > L1 cache 时（比如 K=65536, 一行=256KB > L1=128KB），     │
# │    V3 的 Pass 2/3 会 L1 miss → 回到 L2/DRAM 延迟，                      │
# │    这时寄存器缓存才能省掉真正昂贵的访存。                                │
# │                                                                         │
# │  Layer Norm 实际上是 L1-friendly 的：                                    │
# │    每 block 反复读同一行小数据 → L1 cache 自然帮你缓存了                 │
# │    这也是为什么 PyTorch/Triton 的 LayerNorm 也不做寄存器缓存            │
# └─────────────────────────────────────────────────────────────────────────┘
#
# 原生 CUDA 等价思路：
#   float reg_cache[NUM_TILES][4];  // 寄存器数组缓存所有数据
#   // Pass 1: 读 GMEM → reg_cache，同时累加 sum
#   for (int t = 0; t < NUM_TILES; t++) {
#       float4 v = *(float4*)(x + row * K + t * TILE_SIZE + tid * 4);
#       reg_cache[t][0] = v.x; reg_cache[t][1] = v.y; ...
#       sum += v.x + v.y + v.z + v.w;
#   }
#   mean = block_reduce_sum(sum) / K;
#   // Pass 2: 从 reg_cache 读（不访存！），累加 var
#   // Pass 3: 从 reg_cache 读（不访存！），归一化后写回 GMEM

NUM_THREADS_V4 = 256
VEC_V4 = 4
TILE_SIZE_V4 = NUM_THREADS_V4 * VEC_V4  # 1024
K_V4 = 4096
NUM_TILES_V4 = K_V4 // TILE_SIZE_V4  # 4

@cute.kernel
def layer_norm_regcache_kernel(
    gX: cute.Tensor, gY: cute.Tensor,
    g: cutlass.Float32, b: cutlass.Float32,
    smem_layout: cute.Layout,
    tv_layout: cute.Layout,
    tiler: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sdata = smem.allocate_tensor(cutlass.Float32, smem_layout)

    K = gX.shape[1]

    # ── TiledCopy (LDG.128) ──
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gX.element_type,
        num_bits_per_copy=gX.element_type.width * VEC_V4,
    )
    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
    thr_copy = tiled_copy.get_slice(tidx)

    src_row = cute.local_tile(gX, (1, K), (row, 0))
    dst_row = cute.local_tile(gY, (1, K), (row, 0))
    row_x = cute.make_tensor(src_row.iterator, cute.make_layout((K,)))
    row_y = cute.make_tensor(dst_row.iterator, cute.make_layout((K,)))

    # ======== 唯一一次 GMEM 读取：全部缓存到寄存器 ========
    # reg_cache[tile_idx] 是每个 tile 的寄存器 fragment
    # 每线程缓存 NUM_TILES_V4 × VEC_V4 = 4 × 4 = 16 个 float（16 个寄存器）
    partial_sum = cutlass.Float32(0.0)

    # 预分配寄存器：先用 tile 0 的 shape 做模板
    blk_template = cute.local_tile(row_x, tiler, (0,))
    tSrc_template = thr_copy.partition_S(blk_template)

    # 为每个 tile 分配独立的寄存器 fragment 并加载
    reg_cache_0 = cute.make_fragment_like(tSrc_template)
    reg_cache_1 = cute.make_fragment_like(tSrc_template)
    reg_cache_2 = cute.make_fragment_like(tSrc_template)
    reg_cache_3 = cute.make_fragment_like(tSrc_template)
    reg_caches = (reg_cache_0, reg_cache_1, reg_cache_2, reg_cache_3)

    # Pass 1: LDG.128 → 寄存器缓存，同时累加 partial_sum
    for tile_idx in cutlass.range_constexpr(NUM_TILES_V4):
        blk = cute.local_tile(row_x, tiler, (tile_idx,))
        tSrc = thr_copy.partition_S(blk)
        cute.copy(tiled_copy, tSrc, reg_caches[tile_idx])

        for vi in range(cute.size(reg_caches[tile_idx])):
            partial_sum = partial_sum + reg_caches[tile_idx][vi]

    sum_val = block_reduce_sum(partial_sum, sdata, NUM_THREADS_V4 // WARP_SIZE, tidx)
    mean = sum_val / cutlass.Float32(K)

    # Pass 2: 从寄存器读（零访存！），累加方差
    partial_var = cutlass.Float32(0.0)
    for tile_idx in cutlass.range_constexpr(NUM_TILES_V4):
        for vi in range(cute.size(reg_caches[tile_idx])):
            diff = reg_caches[tile_idx][vi] - mean
            partial_var = partial_var + diff * diff

    var_sum = block_reduce_sum(partial_var, sdata, NUM_THREADS_V4 // WARP_SIZE, tidx)
    inv_std = cute.math.rsqrt(var_sum / cutlass.Float32(K) + cutlass.Float32(1e-5))

    # Pass 3: 从寄存器读（零访存！），归一化后 STG.128 写回
    for tile_idx in cutlass.range_constexpr(NUM_TILES_V4):
        blk_y = cute.local_tile(row_y, tiler, (tile_idx,))
        tDst = thr_copy.partition_D(blk_y)

        rOut = cute.make_fragment_like(reg_caches[tile_idx])
        for vi in range(cute.size(reg_caches[tile_idx])):
            rOut[vi] = (reg_caches[tile_idx][vi] - mean) * inv_std * g + b

        cute.copy(tiled_copy, rOut, tDst)


@cute.jit
def layer_norm_regcache(mX: cute.Tensor, mY: cute.Tensor, g: float, b: float):
    N = mX.shape[0]
    tv_layout = cute.make_layout(
        (NUM_THREADS_V4, VEC_V4),
        stride=(VEC_V4, 1)
    )
    tiler = (TILE_SIZE_V4,)
    num_warps = NUM_THREADS_V4 // WARP_SIZE
    smem_layout = cute.make_layout((num_warps,))

    layer_norm_regcache_kernel(
        mX, mY, cutlass.Float32(g), cutlass.Float32(b),
        smem_layout, tv_layout, tiler
    ).launch(grid=(N, 1, 1), block=(NUM_THREADS_V4, 1, 1))


# =============================================================================
# 测试
# =============================================================================
if __name__ == "__main__":
    N, K = 128, 256
    g, b = 1.2, 0.5

    print("=" * 60)
    print(f"CuTeDSL Layer Norm (N={N}, K={K}, g={g}, b={b})")
    print("=" * 60)

    x = torch.randn(N, K, device="cuda", dtype=torch.float32)
    eps = 1e-5
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    ref = (x - mean) / torch.sqrt(var + eps) * g + b

    x_ = from_dlpack(x, assumed_align=16)

    # 版本 1: 每 thread 1 个元素
    y1 = torch.empty_like(x)
    y1_ = from_dlpack(y1, assumed_align=16)
    c1 = cute.compile(layer_norm_f32, x_, y1_, g, b)
    c1(x_, y1_, g, b)
    assert torch.allclose(y1, ref, atol=1e-4, rtol=1e-4), f"V1 验证失败！max_diff={(y1-ref).abs().max().item()}"
    print("✅ 版本 1 (每线程 1 元素) 正确")
    t1 = benchmark(c1, kernel_arguments=JitArguments(x_, y1_, g, b))
    print(f"   耗时: {t1:.2f} µs")

    # 版本 2: 每 thread 4 个元素
    y2 = torch.empty_like(x)
    y2_ = from_dlpack(y2, assumed_align=16)
    c2 = cute.compile(layer_norm_float4, x_, y2_, g, b)
    c2(x_, y2_, g, b)
    assert torch.allclose(y2, ref, atol=1e-4, rtol=1e-4), f"V2 验证失败！max_diff={(y2-ref).abs().max().item()}"
    print("✅ 版本 2 (每线程 4 元素, float4) 正确")
    t2 = benchmark(c2, kernel_arguments=JitArguments(x_, y2_, g, b))
    print(f"   耗时: {t2:.2f} µs")

    # 版本 3: K 较大，TiledCopy + 多 tile 循环
    K3 = 4096
    print(f"\n===== 版本 3: TiledCopy + 多 tile 循环 (N={N}, K={K3}) =====")
    x3 = torch.randn(N, K3, device="cuda", dtype=torch.float32)
    mean3 = x3.mean(dim=-1, keepdim=True)
    var3 = ((x3 - mean3) ** 2).mean(dim=-1, keepdim=True)
    ref3 = (x3 - mean3) / torch.sqrt(var3 + eps) * g + b

    x3_ = from_dlpack(x3, assumed_align=16)
    y3 = torch.empty_like(x3)
    y3_ = from_dlpack(y3, assumed_align=16)
    c3 = cute.compile(layer_norm_tiled, x3_, y3_, g, b)
    c3(x3_, y3_, g, b)
    assert torch.allclose(y3, ref3, atol=1e-4, rtol=1e-4), f"V3 验证失败！max_diff={(y3-ref3).abs().max().item()}"
    print("✅ 版本 3 (TiledCopy, K=4096) 正确")
    t3 = benchmark(c3, kernel_arguments=JitArguments(x3_, y3_, g, b))
    print(f"   耗时: {t3:.2f} µs")

    # 版本 4: 寄存器缓存，只读 1 次 GMEM
    print(f"\n===== 版本 4: 寄存器缓存 (N={N}, K={K3}) =====")
    y4 = torch.empty_like(x3)
    y4_ = from_dlpack(y4, assumed_align=16)
    c4 = cute.compile(layer_norm_regcache, x3_, y4_, g, b)
    c4(x3_, y4_, g, b)
    assert torch.allclose(y4, ref3, atol=1e-4, rtol=1e-4), f"V4 验证失败！max_diff={(y4-ref3).abs().max().item()}"
    print("✅ 版本 4 (寄存器缓存, K=4096) 正确")
    t4 = benchmark(c4, kernel_arguments=JitArguments(x3_, y4_, g, b))
    print(f"   耗时: {t4:.2f} µs")
    print(f"   V4 vs V3 加速比: {t3 / t4:.2f}x")
    print(f"   ℹ️  K=4096 → 一行 16KB << L1 128KB → V3 的 3 遍读全命中 L1，寄存器缓存无优势")

    # ── L1 Cache 影响验证 ──
    # K=4096:  一行 = 16KB  << L1 128KB → V3 的 Pass 2/3 命中 L1
    # K=65536: 一行 = 256KB >> L1 128KB → V3 的 Pass 2/3 会 L1 miss，回到 L2/DRAM
    # 如果 V4 的寄存器缓存有效，应该在大 K 时看到明显加速

    # 放大 N 测试（K 不变=4096，数据超出 L2 但每行仍在 L1 内）
    N_large = 32768
    print(f"\n===== 放大 N 对比 (N={N_large}, K={K3}) =====")
    print(f"   总数据 = {N_large * K3 * 4 / 1024 / 1024:.0f} MB >> L2 32MB，但每行 16KB << L1 128KB")
    x_large = torch.randn(N_large, K3, device="cuda", dtype=torch.float32)
    mean_l = x_large.mean(dim=-1, keepdim=True)
    var_l = ((x_large - mean_l) ** 2).mean(dim=-1, keepdim=True)
    ref_l = (x_large - mean_l) / torch.sqrt(var_l + eps) * g + b

    x_l_ = from_dlpack(x_large, assumed_align=16)

    y3_l = torch.empty_like(x_large)
    y3_l_ = from_dlpack(y3_l, assumed_align=16)
    c3_l = cute.compile(layer_norm_tiled, x_l_, y3_l_, g, b)
    c3_l(x_l_, y3_l_, g, b)
    assert torch.allclose(y3_l, ref_l, atol=1e-4, rtol=1e-4), f"V3 大规模验证失败！"
    t3_l = benchmark(c3_l, kernel_arguments=JitArguments(x_l_, y3_l_, g, b))
    print(f"   V3 (3 遍 GMEM): {t3_l:.2f} µs")

    y4_l = torch.empty_like(x_large)
    y4_l_ = from_dlpack(y4_l, assumed_align=16)
    c4_l = cute.compile(layer_norm_regcache, x_l_, y4_l_, g, b)
    c4_l(x_l_, y4_l_, g, b)
    assert torch.allclose(y4_l, ref_l, atol=1e-4, rtol=1e-4), f"V4 大规模验证失败！"
    t4_l = benchmark(c4_l, kernel_arguments=JitArguments(x_l_, y4_l_, g, b))
    print(f"   V4 (寄存器缓存): {t4_l:.2f} µs")
    print(f"   V4 vs V3: {t3_l / t4_l:.2f}x")
    print(f"   ℹ️  每行仍然只有 16KB → L1 命中 → V3 和 V4 差不多")

    # ── 大 K 测试：让一行数据冲掉 L1 Cache ──
    # K=65536 → 一行 = 65536 × 4B = 256 KB >> L1 128 KB
    # 这时 V3 的 Pass 2/3 读同一行会 L1 miss，被迫回 L2/DRAM
    # V4 的寄存器缓存应该能体现优势
    #
    # 注意：NUM_TILES 必须是编译时常量（range_constexpr），
    # 所以大 K 需要重新定义 kernel。
    # 每线程缓存 = 64 tiles × 4 vec = 256 个 float = 256 个寄存器 > 255 上限！
    # 所以用 VEC=4, THREADS=256 → TILE=1024, K=65536 → 64 tiles → 会 spill
    #
    # 折中：K=32768 → 一行 128KB ≈ L1 大小（边界情况）
    #        K=49152 → 一行 192KB > L1（确保 miss）
    #        NUM_TILES = 49152/1024 = 48, 每线程缓存 48×4=192 个 float < 255 ✓
    print(f"\n{'='*60}")
    print(f"验证 L1 Cache 影响：大 K 让一行数据超出 L1")
    print(f"{'='*60}")

    K_BIG = 49152  # 192 KB/行 > L1 128 KB
    NUM_TILES_BIG = K_BIG // TILE_SIZE_V3  # 48
    N_BIG = 256  # 足够多行

    print(f"K={K_BIG}, 一行={K_BIG*4//1024} KB vs L1 ~128 KB → L1 放不下一行！")
    print(f"每线程寄存器缓存: {NUM_TILES_BIG}×{VEC_V3} = {NUM_TILES_BIG*VEC_V3} 个 float (<255 ✓)")

    x_big = torch.randn(N_BIG, K_BIG, device="cuda", dtype=torch.float32)
    mean_b = x_big.mean(dim=-1, keepdim=True)
    var_b = ((x_big - mean_b) ** 2).mean(dim=-1, keepdim=True)
    ref_b = (x_big - mean_b) / torch.sqrt(var_b + eps) * g + b

    # ── V3b: 3 遍读 GMEM, 大 K ──
    _TILER_BIG = (TILE_SIZE_V3,)

    @cute.kernel
    def v3b_kernel(gX, gY, g_val, b_val):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sdata = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NUM_THREADS_V3 // WARP_SIZE,)))
        K_local = gX.shape[1]
        tv = cute.make_layout((NUM_THREADS_V3, VEC_V3), stride=(VEC_V3, 1))
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type,
                                         num_bits_per_copy=gX.element_type.width * VEC_V3)
        tiled_copy = cute.make_tiled_copy(copy_atom, tv, _TILER_BIG)
        thr_copy = tiled_copy.get_slice(tidx)
        src_row = cute.local_tile(gX, (1, K_local), (row, 0))
        dst_row = cute.local_tile(gY, (1, K_local), (row, 0))
        row_x = cute.make_tensor(src_row.iterator, cute.make_layout((K_local,)))
        row_y = cute.make_tensor(dst_row.iterator, cute.make_layout((K_local,)))
        # Pass 1: sum
        partial_sum = cutlass.Float32(0.0)
        for tile_idx in cutlass.range_constexpr(NUM_TILES_BIG):
            blk = cute.local_tile(row_x, _TILER_BIG, (tile_idx,))
            tSrc = thr_copy.partition_S(blk)
            rData = cute.make_fragment_like(tSrc)
            cute.copy(tiled_copy, tSrc, rData)
            for vi in range(cute.size(rData)):
                partial_sum = partial_sum + rData[vi]
        mean_val = block_reduce_sum(partial_sum, sdata, NUM_THREADS_V3 // WARP_SIZE, tidx) / cutlass.Float32(K_local)
        # Pass 2: var
        partial_var = cutlass.Float32(0.0)
        for tile_idx in cutlass.range_constexpr(NUM_TILES_BIG):
            blk = cute.local_tile(row_x, _TILER_BIG, (tile_idx,))
            tSrc = thr_copy.partition_S(blk)
            rData = cute.make_fragment_like(tSrc)
            cute.copy(tiled_copy, tSrc, rData)
            for vi in range(cute.size(rData)):
                diff = rData[vi] - mean_val
                partial_var = partial_var + diff * diff
        inv_std = cute.math.rsqrt(block_reduce_sum(partial_var, sdata, NUM_THREADS_V3 // WARP_SIZE, tidx) / cutlass.Float32(K_local) + cutlass.Float32(1e-5))
        # Pass 3: normalize + write
        for tile_idx in cutlass.range_constexpr(NUM_TILES_BIG):
            blk_x = cute.local_tile(row_x, _TILER_BIG, (tile_idx,))
            blk_y = cute.local_tile(row_y, _TILER_BIG, (tile_idx,))
            tSrc = thr_copy.partition_S(blk_x)
            tDst = thr_copy.partition_D(blk_y)
            rData = cute.make_fragment_like(tSrc)
            cute.copy(tiled_copy, tSrc, rData)
            rOut = cute.make_fragment_like(rData)
            for vi in range(cute.size(rData)):
                rOut[vi] = (rData[vi] - mean_val) * inv_std * g_val + b_val
            cute.copy(tiled_copy, rOut, tDst)

    @cute.jit
    def v3b_host(mX, mY, g_val, b_val):
        N_val = mX.shape[0]
        v3b_kernel(mX, mY, cutlass.Float32(g_val), cutlass.Float32(b_val)).launch(
            grid=(N_val, 1, 1), block=(NUM_THREADS_V3, 1, 1))

    # ── V4b: 寄存器缓存, 大 K ──
    # 每线程缓存 48 个 tile × 4 个 float = 192 个寄存器
    @cute.kernel
    def v4b_kernel(gX, gY, g_val, b_val):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sdata = smem.allocate_tensor(cutlass.Float32, cute.make_layout((NUM_THREADS_V4 // WARP_SIZE,)))
        K_local = gX.shape[1]
        tv = cute.make_layout((NUM_THREADS_V4, VEC_V4), stride=(VEC_V4, 1))
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gX.element_type,
                                         num_bits_per_copy=gX.element_type.width * VEC_V4)
        tiled_copy = cute.make_tiled_copy(copy_atom, tv, _TILER_BIG)
        thr_copy = tiled_copy.get_slice(tidx)
        src_row = cute.local_tile(gX, (1, K_local), (row, 0))
        dst_row = cute.local_tile(gY, (1, K_local), (row, 0))
        row_x = cute.make_tensor(src_row.iterator, cute.make_layout((K_local,)))
        row_y = cute.make_tensor(dst_row.iterator, cute.make_layout((K_local,)))

        blk_t = cute.local_tile(row_x, _TILER_BIG, (0,))
        tSrc_t = thr_copy.partition_S(blk_t)

        frag_size = cute.size(tSrc_t)  # 每 tile 4 个 float
        total_regs = NUM_TILES_BIG * frag_size  # 48 × 4 = 192
        reg_all = cute.make_rmem_tensor((total_regs,), cutlass.Float32)

        # Pass 1: LDG → reg_all, 同时累加 sum
        partial_sum = cutlass.Float32(0.0)
        for tile_idx in cutlass.range_constexpr(NUM_TILES_BIG):
            blk = cute.local_tile(row_x, _TILER_BIG, (tile_idx,))
            tSrc = thr_copy.partition_S(blk)
            rData = cute.make_fragment_like(tSrc)
            cute.copy(tiled_copy, tSrc, rData)
            for vi in range(frag_size):
                reg_all[tile_idx * frag_size + vi] = rData[vi]
                partial_sum = partial_sum + rData[vi]

        mean_val = block_reduce_sum(partial_sum, sdata, NUM_THREADS_V4 // WARP_SIZE, tidx) / cutlass.Float32(K_local)

        # Pass 2: 从 reg_all 读（零访存）
        partial_var = cutlass.Float32(0.0)
        for tile_idx in cutlass.range_constexpr(NUM_TILES_BIG):
            for vi in range(frag_size):
                diff = reg_all[tile_idx * frag_size + vi] - mean_val
                partial_var = partial_var + diff * diff

        inv_std = cute.math.rsqrt(block_reduce_sum(partial_var, sdata, NUM_THREADS_V4 // WARP_SIZE, tidx) / cutlass.Float32(K_local) + cutlass.Float32(1e-5))

        # Pass 3: 从 reg_all 读（零访存）+ STG
        for tile_idx in cutlass.range_constexpr(NUM_TILES_BIG):
            blk_y = cute.local_tile(row_y, _TILER_BIG, (tile_idx,))
            tDst = thr_copy.partition_D(blk_y)
            rOut = cute.make_fragment_like(tSrc_t)
            for vi in range(frag_size):
                rOut[vi] = (reg_all[tile_idx * frag_size + vi] - mean_val) * inv_std * g_val + b_val
            cute.copy(tiled_copy, rOut, tDst)

    @cute.jit
    def v4b_host(mX, mY, g_val, b_val):
        N_val = mX.shape[0]
        v4b_kernel(mX, mY, cutlass.Float32(g_val), cutlass.Float32(b_val)).launch(
            grid=(N_val, 1, 1), block=(NUM_THREADS_V4, 1, 1))

    x_b_ = from_dlpack(x_big, assumed_align=16)

    y3b = torch.empty_like(x_big)
    y3b_ = from_dlpack(y3b, assumed_align=16)
    c3b = cute.compile(v3b_host, x_b_, y3b_, g, b)
    c3b(x_b_, y3b_, g, b)
    assert torch.allclose(y3b, ref_b, atol=1e-3, rtol=1e-3), f"V3b 验证失败！max_diff={(y3b-ref_b).abs().max().item()}"
    print("✅ V3b (3 遍 GMEM, K=49152) 正确")
    t3b = benchmark(c3b, kernel_arguments=JitArguments(x_b_, y3b_, g, b))
    print(f"   V3b 耗时: {t3b:.2f} µs")

    y4b = torch.empty_like(x_big)
    y4b_ = from_dlpack(y4b, assumed_align=16)
    c4b = cute.compile(v4b_host, x_b_, y4b_, g, b)
    c4b(x_b_, y4b_, g, b)
    assert torch.allclose(y4b, ref_b, atol=1e-3, rtol=1e-3), f"V4b 验证失败！max_diff={(y4b-ref_b).abs().max().item()}"
    print("✅ V4b (寄存器缓存, K=49152) 正确")
    t4b = benchmark(c4b, kernel_arguments=JitArguments(x_b_, y4b_, g, b))
    print(f"   V4b 耗时: {t4b:.2f} µs")
    print(f"   大 K V4b vs V3b: {t3b / t4b:.2f}x")
    if t3b / t4b > 1.05:
        print(f"   🎉 L1 miss 场景下寄存器缓存有效！省掉 2 次 L2/DRAM 读取")
    else:
        print(f"   ℹ️  可能 192 个寄存器导致 occupancy 下降或 spill，抵消了收益")
