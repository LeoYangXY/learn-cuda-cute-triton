# WMMA GEMM 数据流拆解：CuTE vs 原生 CUDA

以 V5（WMMA + 向量化拷贝）为例，完整的数据流是：

```
GMEM ──→ SMEM ──→ Register ──→ Tensor Core ──→ Register ──→ GMEM
     拷贝     LdMatrix      mma.sync        累加结果      写回
```

下面按这个流动顺序，逐段拆解。

---

## 0. 准备工作：声明 MMA 计算计划

在任何数据搬运之前，先声明"怎么算"。这决定了后面每一步搬运的目标格式。

### CuTE

```python
# 选硬件指令：mma.sync.aligned.m16n8k16.f32.f16.f16.f32
mma_op = cute.nvgpu.warp.MmaF16BF16Op(
    ab_dtype=cutlass.Float16, acc_dtype=cutlass.Float32, shape_mnk=(16, 8, 16))

# 安排 4 个 Warp (2×2)，合起来覆盖 C 的 32×32 子块
tiled_mma = cute.make_tiled_mma(
    mma_op, atom_layout_mnk=(2, 2, 1), permutation_mnk=(32, 32, 16))
```

这一步做了两件事：
1. **Warp 级分工**：4 个 Warp 排成 2×2，每个负责 C 的一块 16×16
2. **Lane 级映射**：一个 Warp 内 32 个 lane 各持有 A/B/C 的哪些元素（硬件定死的）

### 原生 CUDA

```cpp
// Warp 分工：简单，两行搞定
int warp_id = threadIdx.x / 32;
int warp_m = warp_id / 2;   // 0 或 1
int warp_n = warp_id % 2;   // 0 或 1

// Lane 映射：痛苦，必须查 PTX ISA 手册的 Figure
// 例如 C 的 16×8 子块中，32 个 lane 各持有 4 个 fp32：
//   lane 0:  C[0,0], C[0,1], C[8,0], C[8,1]
//   lane 1:  C[1,0], C[1,1], C[9,0], C[9,1]
//   ...
//   lane 31: C[7,6], C[7,7], C[15,6], C[15,7]
// 你得记住这个映射，因为后面 LdMatrix 和写回都要用。
int lane_id = threadIdx.x % 32;
int group_id = lane_id >> 2;
int thread_in_group = lane_id & 3;
```

> **核心区别：** CuTE 的 `make_tiled_mma` 把 Lane 级映射（MMA_Traits）封装在内部，
> 后续的 `partition_A/B/C`、`make_tiled_copy_A`、`retile` 全自动基于它推导。
> 原生 CUDA 你得自己查手册、自己记住、自己在每一步手算。

---

## 1. GMEM → SMEM：把数据从显存搬到片上

每个 Block 负责 C 的一个 128×128 tile。沿 K 方向循环，每次取 A 的 128×64 和 B 的 128×64 到 SMEM。

### CuTE

```python
# ────── host 端：声明搬运计划 ──────

# ① 每次搬 64 bits = 4 个 fp16
atom_copy = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(), Float16, num_bits_per_copy=64)

# ② 128 个线程排成 (8行, 16列)，每线程搬 (1, 4) 个连续值
#    一轮覆盖 8行 × 64列 = 512 个 fp16
#    128 行 / 8 行 = 16 轮（CuTE 自动算）
tA = cute.make_layout((8, 16), stride=(16, 1))    # 线程排布
vA = cute.make_layout((1, 4), stride=(0, 1))      # 每线程搬的形状
tiled_copy = cute.make_tiled_copy_tv(atom_copy, tA, vA)

# ────── kernel 端：执行搬运 ──────

# SMEM 分配（padding 避免 bank conflict）
sA_layout = cute.make_layout((128, 64), stride=(64 + 8, 1))
sA = allocator.allocate_tensor(cutlass.Float16, sA_layout)

# CTA 级分块：从大矩阵切出当前 Block 的 tile
gA = cute.local_tile(mA, tiler=(128,128,64), coord=(bidx, bidy, None), proj=(1, None, 1))
# gA shape: (128, 64, K//64)

# 绑定线程 + 自动分区
thr_copy = tiled_copy.get_slice(tid)
tAgA = thr_copy.partition_S(gA)    # 这个线程负责读 GMEM 的哪些元素
tAsA = thr_copy.partition_D(sA)    # 这个线程负责写 SMEM 的哪些位置

# 搬！（内部自动循环 16 轮，每轮 64-bit load）
cute.copy(tiled_copy, tAgA[None, None, None, kidx], tAsA[None, None, None])
cute.arch.sync_threads()
```

### 原生 CUDA 等价

```cpp
__shared__ half sA[128][64 + 8];   // padding 避免 bank conflict
__shared__ half sB[128][64 + 8];

int tid = threadIdx.x;             // 0~127
int col_tid = tid % 16;            // 列方向 thread id (0~15)
int row_tid = tid / 16;            // 行方向 thread id (0~7)

// 每个 K-tile 迭代：
int k_offset = k_iter * 64;

for (int round = 0; round < 16; round++) {          // 16 轮，手动写
    int row = round * 8 + row_tid;                   // 这轮这个线程负责的行
    int col = col_tid * 4;                           // 这个线程负责的起始列
    int gA_row = blockIdx.x * 128 + row;             // 全局行号
    int gA_col = k_offset + col;                     // 全局列号

    // 64-bit load：一次搬 4 个 fp16（手动强转指针）
    *(uint2*)&sA[row][col] = *(uint2*)&A[gA_row * K + gA_col];
}
// B 同理，16 轮...
__syncthreads();
```

### 对比

| | CuTE | 原生 CUDA |
|---|---|---|
| 线程分工 | `make_layout((8,16))` 声明 | 手算 `tid%16`, `tid/16` |
| 向量化 | `num_bits_per_copy=64` 声明 | 手动 `*(uint2*)` 强转指针 |
| 循环轮数 | `partition` 自动算，`copy` 自动循环 | 手动 `for (round=0; round<16; ...)` |
| 全局地址 | `local_tile` + `partition_S` 自动 | 手算 `blockIdx.x*128 + round*8 + ...` |
| Bank conflict | `stride=(64+8, 1)` 声明 padding | `sA[128][64+8]` 手动加 padding |

---

## 2. SMEM → Register：LdMatrix 加载到寄存器

Tensor Core 要求数据在寄存器里有**特定排布**。普通的 shared memory load (`lds`) 搬过来排布不对。
必须用 `ldmatrix` 指令：32 个 lane 各提供一个 SMEM 地址，硬件把数据重新分发到各 lane 的寄存器，
使得排布恰好符合 `mma.sync` 的要求。

### CuTE

```python
# 声明 ldmatrix 原子：一次加载 4 个 8×8 矩阵 (16-bit 元素)
atom_s2r = cute.make_copy_atom(
    cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), Float16)

# 关键！让 CuTE 根据 tiled_mma 自动推导线程排布
# make_tiled_copy_A 知道 MMA 要求什么布局，自动生成匹配的搬运计划
tiled_s2r = cute.make_tiled_copy_A(atom_s2r, tiled_mma)

# 绑定线程
thr_s2r = tiled_s2r.get_slice(tid)
tCsA_view = thr_s2r.partition_S(sA)    # SMEM 端：每个 lane 从哪读
tCrA_view = thr_s2r.retile(tCrA)       # Register 端：重排列视图
# retile 不移动数据，只是让 copy 和 MMA 用不同视角看同一块寄存器

# 搬！
cute.copy(tiled_s2r, tCsA_view, tCrA_view)
```

### 原生 CUDA 等价

```cpp
uint32_t a_frag[4];   // 4 个 32-bit 寄存器 = 8 个 fp16
int lane_id = threadIdx.x % 32;

// 每个 lane 要从 sA 的哪一行读？查 PTX 手册！
// lane 0 读 row 0, lane 1 读 row 1, ..., lane 15 读 row 15
// lane 16 读 row 0 (另一个 8×8 块), ...
int smem_row = lane_id % 16;
int smem_col_base = (lane_id / 16) * 8;  // 第 0 或第 1 个 8×8 块

// 算 SMEM 地址（还要考虑 padding！）
uint32_t smem_addr = __cvta_generic_to_shared(
    &sA[smem_row][smem_col_base]);

// 内联 PTX：ldmatrix 指令
asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
    : "=r"(a_frag[0]), "=r"(a_frag[1]),
      "=r"(a_frag[2]), "=r"(a_frag[3])
    : "r"(smem_addr)
);

// 现在 a_frag 里的数据排布符合 mma.sync 的要求
// 但具体 a_frag[0] 存的是 A 的哪些元素？还是要查手册：
//   lane 0 的 a_frag[0] = {A[0,0], A[0,1]}   (2 个 fp16 pack 成 1 个 uint32)
//   lane 0 的 a_frag[1] = {A[8,0], A[8,1]}
//   lane 0 的 a_frag[2] = {A[0,8], A[0,9]}
//   lane 0 的 a_frag[3] = {A[8,8], A[8,9]}
//   lane 1 的 a_frag[0] = {A[1,0], A[1,1]}
//   ...
```

### 对比

| | CuTE | 原生 CUDA |
|---|---|---|
| 指令 | `LdMatrix8x8x16bOp(num_matrices=4)` | 手写 `asm("ldmatrix.sync...")` |
| SMEM 地址 | `partition_S(sA)` 自动算 | 手算 `lane_id % 16`、考虑 padding |
| 寄存器布局 | `retile(tCrA)` 自动对齐 MMA 要求 | 你得知道 `a_frag[i]` 对应 A 的哪些元素 |
| 和 MMA 的关联 | `make_tiled_copy_A(atom, tiled_mma)` 自动推导 | 你得自己保证 ldmatrix 的读取模式和 mma.sync 的输入布局匹配 |

> **关键点：** `make_tiled_copy_A` 接收 `tiled_mma` 作为参数，
> 因为它需要知道 MMA 指令要求什么寄存器布局，才能反推出 SMEM 应该怎么读。
> 原生 CUDA 里这两步是解耦的——你得自己保证 ldmatrix 和 mma.sync 的数据布局匹配。

---

## 3. Register 排布 → Tensor Core 计算

数据已经在寄存器里了，且排布正确（LdMatrix 保证的）。现在发射 MMA 指令。

一条 `mma.sync.m16n8k16` 只算 C 的 16×8。4 个 Warp 覆盖 32×32。
但 CTA tile 是 128×128，所以还需要 (128/32)×(128/32) = 4×4 = 16 组 MMA。

### CuTE

```python
cute.gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC)
# 内部自动：
#   for mma_m in range(4):       ← 128/32
#     for mma_n in range(4):     ← 128/32
#       发射 mma.sync 指令（4 个 Warp 各自执行）
```

### 原生 CUDA 等价

```cpp
// 你得自己写双重循环 + 内联 PTX
for (int mma_m = 0; mma_m < 4; mma_m++) {
    for (int mma_n = 0; mma_n < 4; mma_n++) {
        // 切换到对应子块的 a_frag / b_frag
        // （怎么切换？取决于你的寄存器排列方式，自己算偏移）

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, "       // D: 4 个 fp32 输出
            "{%4,%5,%6,%7}, "       // A: 4 个 uint32 (8 个 fp16)
            "{%8,%9}, "             // B: 2 个 uint32 (4 个 fp16)
            "{%10,%11,%12,%13};"    // C: 4 个 fp32 累加器
            : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
            :  "r"(a[off_a+0]), "r"(a[off_a+1]),
               "r"(a[off_a+2]), "r"(a[off_a+3]),
               "r"(b[off_b+0]), "r"(b[off_b+1]),
               "f"(c[off_c+0]), "f"(c[off_c+1]),
               "f"(c[off_c+2]), "f"(c[off_c+3])
        );
    }
}
```

### 对比

| | CuTE | 原生 CUDA |
|---|---|---|
| 指令 | `cute.gemm(tiled_mma, ...)` 一行 | 手写 `asm("mma.sync...")` |
| 循环 | 自动算 4×4=16 次 | 手写 `for (mma_m) for (mma_n)` |
| 寄存器偏移 | 自动（partition 已算好） | 手算 `off_a`, `off_b`, `off_c` |

---

## 4. Register → GMEM：写回结果

MMA 完成后，每个 lane 的寄存器里有若干 fp32 累加值。要转成 fp16 写回 GMEM。

问题：lane 0 的 `c_frag[0]` 对应 C 的哪个 `(m, n)`？

### CuTE

```python
# tCgC 在之前 partition_C(gC) 时已经记录了映射
atom_store = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type)
tCrC_out = cute.make_fragment_like(tCrC, dtype=cutlass.Float16)

for i in range(cute.size(tCrC_out)):
    tCrC_out[i] = cutlass.Float16(tCrC[i])    # fp32 → fp16

cute.copy(atom_store, tCrC_out, tCgC)          # 自动写到正确的 GMEM 位置
# tCgC 知道每个寄存器对应 C 的哪个 (m,n)，直接用
```

### 原生 CUDA 等价

```cpp
int lane_id = threadIdx.x % 32;
int group_id = lane_id >> 2;              // lane_id / 4
int thread_in_group = lane_id & 3;        // lane_id % 4

// 查 PTX 手册得到的映射（以 16×8 子块为例）：
//   c_frag[0] → C[group_id,     thread_in_group * 2]
//   c_frag[1] → C[group_id,     thread_in_group * 2 + 1]
//   c_frag[2] → C[group_id + 8, thread_in_group * 2]
//   c_frag[3] → C[group_id + 8, thread_in_group * 2 + 1]

// 还要加上 Warp 偏移和 MMA 循环偏移：
int base_m = blockIdx.x * 128 + warp_m * 16 + mma_m * 32;
int base_n = blockIdx.y * 128 + warp_n * 16 + mma_n * 32;

for (int reg = 0; reg < 4; reg++) {
    int m_offset, n_offset;
    switch (reg) {
        case 0: m_offset = group_id;     n_offset = thread_in_group * 2;     break;
        case 1: m_offset = group_id;     n_offset = thread_in_group * 2 + 1; break;
        case 2: m_offset = group_id + 8; n_offset = thread_in_group * 2;     break;
        case 3: m_offset = group_id + 8; n_offset = thread_in_group * 2 + 1; break;
    }
    C[(base_m + m_offset) * N + (base_n + n_offset)] = __float2half(c_frag[reg]);
}
```

### 对比

| | CuTE | 原生 CUDA |
|---|---|---|
| (m,n) 映射 | `partition_C(gC)` 已自动记录 | 手查 PTX 手册 Figure，手写 switch-case |
| Warp/MMA 偏移 | 自动叠加在 partition 中 | 手算 `base_m = blockIdx*128 + warp_m*16 + mma_m*32 + ...` |
| 写回 | `cute.copy(atom, src, dst)` 一行 | 手算地址 `C[(m)*N + (n)]` |

---

## 完整流程总结

```
                    CuTE                                原生 CUDA
                    ────                                ────────
准备          make_tiled_mma(mma_op,              查 PTX 手册 Figure
              atom_layout, permutation)            记住 lane→数据 映射
              └→ 自动封装 Lane 级映射               手算 group_id, thread_in_group
                                                   ↓ 后面每步都要用这个映射

GMEM→SMEM    make_tiled_copy_tv(atom, tA, vA)     手算 tid%16, tid/16
              cute.copy(tiled_copy, src, dst)       for 循环 + *(uint2*) 强转
              └→ 自动 16 轮、64-bit load            手写 16 轮循环

SMEM→Reg     make_tiled_copy_A(atom, tiled_mma)   asm("ldmatrix.sync...")
              cute.copy(tiled_s2r, sA, rA)          手算 lane→SMEM 地址
              └→ 自动匹配 MMA 需要的布局             自己保证布局匹配

MMA 计算     cute.gemm(tiled_mma, D, A, B, C)     asm("mma.sync...")
              └→ 自动 4×4=16 次循环                  手写双重循环 + PTX

写回 GMEM    cute.copy(atom, fragment, tCgC)       手查映射 + 手算全局地址
              └→ partition_C 记录了映射               switch(reg) + base_m + ...
```

**核心洞察：** 原生 CUDA 里，你在第 0 步查手册得到的 Lane 映射，要在第 2、3、4 步反复使用。
CuTE 在第 0 步 `make_tiled_mma` 里把映射封装一次，后面每步通过 `partition` / `make_tiled_copy_A` / `retile` 自动传递。

---

## 性能演进 (M=N=K=4096, fp16)

```
V2  Tiled+SMEM+REG (fp32)          39,938 µs    3.44 TFLOPS
V3  CuTE MmaUniversalOp (fp32)    310,220 µs    0.44 TFLOPS  ← 标量 FMA，仅教学用途
V4  WMMA TC (手动拷贝)              16,469 µs    8.35 TFLOPS  ← Tensor Core 加速
V5  WMMA + 向量化拷贝                5,329 µs   25.79 TFLOPS  ← 64-bit load 加速搬运
V6  WMMA + TMA (SM90+)              4,771 µs   28.81 TFLOPS  ← TMA 硬件搬运
PyTorch torch.matmul                4,670 µs   29.43 TFLOPS
```

关键跳跃点：
- **V2→V4**：Tensor Core 替代标量 FMA（计算瓶颈突破）
- **V4→V5**：向量化拷贝替代逐元素拷贝（搬运瓶颈突破，3x 加速）
- **V5→V6**：TMA 硬件搬运（线程解放，不占 register/ALU）
