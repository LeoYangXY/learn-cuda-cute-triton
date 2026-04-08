"""
=============================================================================
CuTeDSL 复现 origin_cuda_kernel/add/add.cu 中的 elementwise add kernel
=============================================================================

原生 CUDA 中有 7 个版本：
  1. eltwise_add_scaler         — FP32 标量版：每个 thread 处理 1 个元素
  2. eltwise_add_vector         — FP32 向量化版：每个 thread 用 float4 处理 4 个元素
  3. elementwise_add_f16_kernel — FP16 标量版
  4. elementwise_add_f16_vec2   — FP16 half2 向量化版
  5. elementwise_add_f16_vec2_4 — FP16 每线程 8 个元素 + __hadd2
  6. elementwise_add_f16_vec2_4_unroll — 上面的 #pragma unroll 版
  7. elementwise_add_f16_pack   — FP16 128-bit pack 版

在 CuTeDSL 中，向量化是通过 copy_atom 的 num_bits_per_copy 来控制的：
  - 32 bits  → LDG.32  → 等价于原生 CUDA 的标量访问
  - 64 bits  → LDG.64  → 等价于 float2 / half4
  - 128 bits → LDG.128 → 等价于 float4 / 8个half（128-bit pack）

因此，原生 CUDA 的 7 个版本在 CuTeDSL 中可以精简为：
  版本 1: FP32 标量版（num_bits_per_copy=32, 即每线程搬 1 个 float）
  版本 2: FP32 向量化版（num_bits_per_copy=128, 即每线程搬 4 个 float = float4）
  版本 3: FP16 标量版（num_bits_per_copy=16, 即每线程搬 1 个 half）
  版本 4: FP16 128-bit 向量化版（num_bits_per_copy=128, 即每线程搬 8 个 half = 128 bits）

原生 CUDA 的中间版本（half2、half2×4、unroll）在 CuTeDSL 中不需要手动写，
因为 CuTE 的 copy_atom 直接控制向量化宽度，编译器自动处理展开。
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments
import torch


# =============================================================================
# 版本 1: FP32 标量版 — 每个 thread 处理 1 个元素
# =============================================================================
# 原生 CUDA 等价：
#   __global__ void eltwise_add_scaler(float* a, float* b, float* c, int N) {
#       int idx = blockDim.x * blockIdx.x + threadIdx.x;
#       if (idx <= N) c[idx] = a[idx] + b[idx];
#   }
#
# CuTeDSL 中不使用 TiledCopy，直接用 local_partition 做最朴素的标量访问。
# 这等价于原生 CUDA 中每个 thread 用下标直接访问 a[idx]、b[idx]。

BLOCK_SIZE = 256

@cute.kernel
def add_f32_scalar_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    # 关键点：在 Python 中，(BLOCK_SIZE) 只是加了括号的普通表达式，等价于 BLOCK_SIZE 本身。要创建单元素元组，必须加逗号：
    # a = (128)    # 这是 int，值为 128
    # b = (128,)   # 这是 tuple，值为 (128,)

    # 原生 CUDA 等价：int idx = blockDim.x * blockIdx.x + threadIdx.x;
    # local_tile 切出当前 Block 负责的那一段（BLOCK_SIZE 个元素）
    blkA = cute.local_tile(gA, (BLOCK_SIZE,), (bidx,))
    blkB = cute.local_tile(gB, (BLOCK_SIZE,), (bidx,))
    blkC = cute.local_tile(gC, (BLOCK_SIZE,), (bidx,))

    # 原生 CUDA 等价：c[idx] = a[idx] + b[idx];
    # local_partition 把 BLOCK_SIZE 个元素平均分给每个线程
    # thread_layout = (BLOCK_SIZE,)：256 个线程，每线程 1 个元素
    thread_layout = cute.make_layout(BLOCK_SIZE)
    tA = cute.local_partition(blkA, thread_layout, tidx)
    tB = cute.local_partition(blkB, thread_layout, tidx)
    tC = cute.local_partition(blkC, thread_layout, tidx)

    # 直接标量赋值 → 编译器生成 LDG.32 / STG.32
    tC[0] = tA[0] + tB[0]


@cute.jit
def add_f32_scalar(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    n = mA.shape[0]
    num_blocks = n // BLOCK_SIZE
    add_f32_scalar_kernel(mA, mB, mC).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 版本 2: FP32 向量化版 — 每个 thread 用 float4 处理 4 个元素
# =============================================================================
# 原生 CUDA 等价：
#   __global__ void eltwise_add_vector(float* a, float* b, float* c, int N) {
#       int start = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
#       float4 a_vec = *reinterpret_cast<float4*>(a + start);
#       float4 b_vec = *reinterpret_cast<float4*>(b + start);
#       float4 c_vec;
#       c_vec.x = a_vec.x + b_vec.x; ... (4 个分量)
#       *reinterpret_cast<float4*>(c + start) = c_vec;
#   }
#
# CuTeDSL 中：
#   num_bits_per_copy = 32 * 4 = 128 → 编译器生成 LDG.128
#   等价于原生 CUDA 的 float4 向量化

VEC_F32 = 4  # 每线程 4 个 float = 128 bits

@cute.kernel
def add_f32_vec4_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    tv_layout: cute.Layout, tiler: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blkA = cute.local_tile(gA, tiler, (bidx,))
    blkB = cute.local_tile(gB, tiler, (bidx,))
    blkC = cute.local_tile(gC, tiler, (bidx,))

    # 原生 CUDA 等价：*(float4*)(&a[start])
    # copy_atom 的 num_bits_per_copy=128 → 一条 LDG.128 指令搬 4 个 float
    # 定义"一条指令搬多少数据"
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gA.element_type,
        num_bits_per_copy=gA.element_type.width * VEC_F32,  # 32 × 4 = 128 bits
    )


    # tv_layout其实全称可以理解为t_idx,v_idx的对应表
    # 比如我们现在是1D排布的256个元素，总共是1D排布的64个thread：
    # 原生cuda中做任务划分其实第一步也是要先直观的去想（thread0去处理元素0，1，2，3），然后再用数学公式表示出来
    # cute中我们也是直观的先想（thread0去处理元素0，1，2，3），然后画出这个t_idx与v_idx对应的表，然后再用tv_layout这个cute代数表示出来：
    
    # 比如对于thread0处理元素0，1，2，3  thread1去处理元素4，5，6，7 这样的，我们的tv_layout表可以画为：
    #                          v_idx
    #               ╭  0  ──>  (  0,    1,    2,    3  )
    #               │  1  ──>  (  4,    5,    6,    7  )
    #               │  2  ──>  (  8,    9,   10,   11  )
    # t_idx        <   :          :     :     :     :
    #               │  :          :     :     :     :
    #               │ 62  ──>  (248,  249,  250,  251  )
    #               ╰ 63  ──>  (252,  253,  254,  255  )
    # 用cute代数表示出来就是：tv_layout = cute.make_layout((64,4),(4,1))

    # 对于thread0处理元素0，64，128，192  thread1去处理元素1，65，129，193 这样的，我们的tv_layout表可以画为：
    #                          v_idx
    #               ╭  0  ──>  (  0,   64,  128,  192  )
    #               │  1  ──>  (  1,   65,  129,  193  )
    #               │  2  ──>  (  2,   66,  130,  194  )
    # t_idx        <   :          :     :     :     :
    #               │  :          :     :     :     :
    #               │ 62  ──>  ( 62,  126,  190,  254  )
    #               ╰ 63  ──>  ( 63,  127,  191,  255  )
    # 用cute代数表示出来就是：tv_layout = cute.make_layout((64,4),(1,64))

    # 换一个例子：对于2D排布的16×16 = 256 个元素，有1D排布的8 个线程：
    # 很直观的想就是thread0负责第 0、1 行，thread1负责第 2、3 行
    # 画出映射表就是：
    #                                            v_idx
    #               ╭         ╭ (0, 0), (0, 1), (0, 2)  ...  (0,15) ╮
    #               │  0  ──> ╰ (1, 0), (1, 1), (1, 2)  ...  (1,15) ╯
    #               │
    #               │         ╭ (2, 0), (2, 1), (2, 2)  ...  (2,15) ╮
    #               │  1  ──> ╰ (3, 0), (3, 1), (3, 2)  ...  (3,15) ╯
    #               │
    #               │         ╭ (4, 0), (4, 1), (4, 2)  ...  (4,15) ╮
    #               │  2  ──> ╰ (5, 0), (5, 1), (5, 2)  ...  (5,15) ╯
    # t_idx        <
    #               │            :        :        :      :     :
    #               │            :        :        :      :     :
    #               │
    #               │         ╭ (12,0), (12,1), (12,2)  ... (12,15) ╮
    #               │  6  ──> ╰ (13,0), (13,1), (13,2)  ... (13,15) ╯
    #               │
    #               │         ╭ (14,0), (14,1), (14,2)  ... (14,15) ╮
    #               ╰  7  ──> ╰ (15,0), (15,1), (15,2)  ... (15,15) ╯

    # 用shape=( (内部行数，外部行数…)(内部列数，外部列数…) ) stride：划线去看即可这样的方式去看出layout，写出cute代数：
    # shape  = ((2, 8), (16, 1))
    # stride = ((16, 32), (1, 1))    ← 划线算: 内行stride=16, 外行stride=2*16=32，内列stride=1,  外列不存在
    # 不过要注意这个数据的表示方式：
    # shape  = ((2, 8), (16, 1))       ← ((V_m, T_m), (V_n, T_n))
    # stride = ((16, 32), (1, 1))      ← ((s_Vm, s_Tm), (s_Vn, s_Tn))
    
    # CuTE 的 TV layout 惯例是写成 (T_shape, V_shape) : (T_stride, V_stride)，即把所有 T 的部分放一起，所有 V 的部分放一起：
    # T_shape  = (T_m, T_n)  = (8, 1)  → 简化掉 1 → 8
    # T_stride = (s_Tm, s_Tn) = (32, 1) → 简化掉 1   → 32
    # V_shape  = (V_m, V_n)   = (2, 16)
    # V_stride = (s_Vm, s_Vn) = (16, 1)

    # 所以，tv_layout = (8, (2, 16)) : (32, (16, 1))

    # 不过其实我们不用这么复杂的去看，很直观的就能发现T的shape是8，V的shape是(2,16)，T的stride是32，V的stride是(16,1)
    

    #描述"整个 thread block 如何协作搬运一个 tile"
    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
    
    # 从 tiled_copy 中取出当前线程（tidx）对应的那一份
    # 原生 CUDA 等价：int start = (blockDim.x * blockIdx.x + threadIdx.x) * 4; 中计算 start 的过程——确定"我这个线程从哪里开始"
    thr_copy = tiled_copy.get_slice(tidx)

    tAsA = thr_copy.partition_S(blkA)# Source 分区：当前线程从 blkA 中要读哪些元素
    tBsB = thr_copy.partition_S(blkB)# Source 分区
    tCsC = thr_copy.partition_D(blkC)# Destination 分区：当前线程往 blkC 写哪些元素

    # GMEM → 寄存器（LDG.128）
    rA = cute.make_fragment_like(tAsA)
    rB = cute.make_fragment_like(tBsB)
    cute.copy(tiled_copy, tAsA, rA)# 这里才真正发 LDG.128 指令
    cute.copy(tiled_copy, tBsB, rB)

    # 寄存器中逐元素加法
    # 原生 CUDA 等价：c_vec.x = a_vec.x + b_vec.x; ...
    rC = cute.make_fragment_like(rA)
    for i in range(cute.size(rC)):
        rC[i] = rA[i] + rB[i]

    # 寄存器 → GMEM（STG.128）
    cute.copy(tiled_copy, rC, tCsC)


@cute.jit
def add_f32_vec4(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    n = mA.shape[0]
    elems_per_block = BLOCK_SIZE * VEC_F32  # 256 × 4 = 1024
    num_blocks = n // elems_per_block

    tv_layout = cute.make_layout(
        (BLOCK_SIZE, VEC_F32),       # (256, 4)
        stride=(VEC_F32, 1)          # 每线程内 4 个值连续
    )
    tiler = (elems_per_block,)  # (1024,)

    add_f32_vec4_kernel(mA, mB, mC, tv_layout, tiler).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 版本 3: FP16 标量版 — 每个 thread 处理 1 个 half
# =============================================================================
# 原生 CUDA 等价：
#   __global__ void elementwise_add_f16_kernel(half* a, half* b, half* c, int N) {
#       int idx = blockIdx.x * blockDim.x + threadIdx.x;
#       if (idx < N) c[idx] = __hadd(a[idx], b[idx]);
#   }

@cute.kernel
def add_f16_scalar_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blkA = cute.local_tile(gA, (BLOCK_SIZE,), (bidx,))
    blkB = cute.local_tile(gB, (BLOCK_SIZE,), (bidx,))
    blkC = cute.local_tile(gC, (BLOCK_SIZE,), (bidx,))

    thread_layout = cute.make_layout(BLOCK_SIZE)
    tA = cute.local_partition(blkA, thread_layout, tidx)
    tB = cute.local_partition(blkB, thread_layout, tidx)
    tC = cute.local_partition(blkC, thread_layout, tidx)

    # 原生 CUDA 等价：c[idx] = __hadd(a[idx], b[idx]);
    # CuTeDSL 中直接用 + 运算符，编译器自动生成 __hadd
    tC[0] = tA[0] + tB[0]


@cute.jit
def add_f16_scalar(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    n = mA.shape[0]
    num_blocks = n // BLOCK_SIZE
    add_f16_scalar_kernel(mA, mB, mC).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 版本 4: FP16 128-bit 向量化版 — 每个 thread 搬 8 个 half = 128 bits
# =============================================================================
# 原生 CUDA 等价（pack 版本）：
#   __global__ void elementwise_add_f16_pack(half* a, half* b, half* c, int N) {
#       int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
#       half pack_a[8], pack_b[8], pack_c[8];
#       LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);   // 128-bit load
#       LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
#       for (int i = 0; i < 8; i += 2)
#           HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
#       LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);   // 128-bit store
#   }
#
# CuTeDSL 中：num_bits_per_copy = 16 * 8 = 128 → LDG.128，一次搬 8 个 half
# 这一个版本同时覆盖了原生 CUDA 的 vec2、vec2_4、vec2_4_unroll、pack 四个版本

VEC_F16 = 8  # 每线程 8 个 half = 128 bits

@cute.kernel
def add_f16_vec8_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    tv_layout: cute.Layout, tiler: cute.Shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blkA = cute.local_tile(gA, tiler, (bidx,))
    blkB = cute.local_tile(gB, tiler, (bidx,))
    blkC = cute.local_tile(gC, tiler, (bidx,))

    # num_bits_per_copy = 16 × 8 = 128 → LDG.128
    # 原生 CUDA 等价：LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gA.element_type,
        num_bits_per_copy=gA.element_type.width * VEC_F16,  # 16 × 8 = 128 bits
    )
    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
    thr_copy = tiled_copy.get_slice(tidx)

    tAsA = thr_copy.partition_S(blkA)
    tBsB = thr_copy.partition_S(blkB)
    tCsC = thr_copy.partition_D(blkC)

    rA = cute.make_fragment_like(tAsA)
    rB = cute.make_fragment_like(tBsB)
    cute.copy(tiled_copy, tAsA, rA)    # LDG.128
    cute.copy(tiled_copy, tBsB, rB)    # LDG.128

    # 原生 CUDA 等价：HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
    # CuTeDSL 中直接 + 即可，编译器自动选择最优的向量化加法指令
    rC = cute.make_fragment_like(rA)
    for i in range(cute.size(rC)):
        rC[i] = rA[i] + rB[i]

    cute.copy(tiled_copy, rC, tCsC)    # STG.128


@cute.jit
def add_f16_vec8(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    n = mA.shape[0]
    elems_per_block = BLOCK_SIZE * VEC_F16  # 256 × 8 = 2048
    num_blocks = n // elems_per_block

    tv_layout = cute.make_layout(
        (BLOCK_SIZE, VEC_F16),
        stride=(VEC_F16, 1)
    )
    tiler = (elems_per_block,)

    add_f16_vec8_kernel(mA, mB, mC, tv_layout, tiler).launch(
        grid=(num_blocks, 1, 1), block=(BLOCK_SIZE, 1, 1))


# =============================================================================
# 测试 + 性能基准
# =============================================================================
if __name__ == "__main__":
    N = 64 * 1024 * 1024  # 和原生 CUDA 测试一致

    print("=" * 60)
    print("CuTeDSL Elementwise Add — 复现 origin_cuda_kernel/add/add.cu")
    print("=" * 60)

    # ===== FP32 测试 =====
    print(f"\n===== FP32 Elementwise Add (N={N}) =====")
    a_f32 = torch.randn(N, device="cuda", dtype=torch.float32)
    b_f32 = torch.randn(N, device="cuda", dtype=torch.float32)
    ref_f32 = a_f32 + b_f32

    # 版本 1: 标量
    c1 = torch.empty_like(a_f32)
    a1, b1, c1_ = from_dlpack(a_f32, assumed_align=16), from_dlpack(b_f32, assumed_align=16), from_dlpack(c1, assumed_align=16)
    compiled_scalar_f32 = cute.compile(add_f32_scalar, a1, b1, c1_)
    compiled_scalar_f32(a1, b1, c1_)
    assert torch.allclose(c1, ref_f32, atol=1e-5), "FP32 标量版验证失败！"
    print("✅ FP32 标量版 正确")
    t1 = benchmark(compiled_scalar_f32, kernel_arguments=JitArguments(a1, b1, c1_))
    print(f"   耗时: {t1:.2f} µs")

    # 版本 2: 向量化 float4
    c2 = torch.empty_like(a_f32)
    c2_ = from_dlpack(c2, assumed_align=16)
    compiled_vec4_f32 = cute.compile(add_f32_vec4, a1, b1, c2_)
    compiled_vec4_f32(a1, b1, c2_)
    assert torch.allclose(c2, ref_f32, atol=1e-5), "FP32 向量化版验证失败！"
    print("✅ FP32 向量化版 (float4 = 128-bit) 正确")
    t2 = benchmark(compiled_vec4_f32, kernel_arguments=JitArguments(a1, b1, c2_))
    print(f"   耗时: {t2:.2f} µs")

    # ===== FP16 测试 =====
    print(f"\n===== FP16 Elementwise Add (N={N}) =====")
    a_f16 = torch.randn(N, device="cuda", dtype=torch.float16)
    b_f16 = torch.randn(N, device="cuda", dtype=torch.float16)
    ref_f16 = a_f16 + b_f16

    a16, b16 = from_dlpack(a_f16, assumed_align=16), from_dlpack(b_f16, assumed_align=16)

    # 版本 3: FP16 标量
    c3 = torch.empty_like(a_f16)
    c3_ = from_dlpack(c3, assumed_align=16)
    compiled_scalar_f16 = cute.compile(add_f16_scalar, a16, b16, c3_)
    compiled_scalar_f16(a16, b16, c3_)
    assert torch.allclose(c3, ref_f16, atol=1e-2), "FP16 标量版验证失败！"
    print("✅ FP16 标量版 正确")
    t3 = benchmark(compiled_scalar_f16, kernel_arguments=JitArguments(a16, b16, c3_))
    print(f"   耗时: {t3:.2f} µs")

    # 版本 4: FP16 128-bit pack
    c4 = torch.empty_like(a_f16)
    c4_ = from_dlpack(c4, assumed_align=16)
    compiled_vec8_f16 = cute.compile(add_f16_vec8, a16, b16, c4_)
    compiled_vec8_f16(a16, b16, c4_)
    assert torch.allclose(c4, ref_f16, atol=1e-2), "FP16 128-bit pack 版验证失败！"
    print("✅ FP16 128-bit pack 版 正确")
    t4 = benchmark(compiled_vec8_f16, kernel_arguments=JitArguments(a16, b16, c4_))
    print(f"   耗时: {t4:.2f} µs")

    # 汇总
    print(f"\n{'='*60}")
    print(f"  {'版本':<30} {'耗时(µs)':<12}")
    print(f"  {'-'*42}")
    print(f"  {'FP32 标量 (LDG.32)':<30} {t1:<12.2f}")
    print(f"  {'FP32 向量化 (LDG.128)':<30} {t2:<12.2f}")
    print(f"  {'FP16 标量 (LDG.16)':<30} {t3:<12.2f}")
    print(f"  {'FP16 128-bit pack (LDG.128)':<30} {t4:<12.2f}")
    print(f"{'='*60}")
