# PTX 内联汇编学习

参考 DeepGEMM 的风格：直接用内联 PTX 控制硬件指令，避免依赖高层抽象。

## 目录结构

```
ptx_asm/
├── 01_basics/              — PTX 基础: 特殊寄存器、向量化访存、barrier、atomic、shuffle
├── 02_cp_async/            — cp.async 异步数据搬运 (GMEM → SMEM, double buffering)
├── 03_ldmatrix/            — ldmatrix 指令 (SMEM → Register, Tensor Core 专用布局)
├── 04_mma/                 — mma.sync Tensor Core MMA (m16n8k16, f16/f32 accumulator)
├── 05_hgemm_mma/           — 完整 HGEMM: cp.async + ldmatrix + mma pipeline
├── 06_hopper_wgmma_tma/    — Hopper sm_90: wgmma, TMA, setmaxnreg, mbarrier, cluster
├── 07_type_convert/        — 数据类型转换: FP32↔FP16↔BF16↔FP8, 舍入模式
├── 08_memory_control/      — 访存控制: cache 修饰符, prefetch, red, selp, fma
└── README.md
```

## 学习路径

1. **basics** — 掌握 `asm volatile` 语法、约束修饰符 (`"r"`, `"l"`, `"f"`, `"n"`)
2. **cp_async** — 理解异步数据流: commit_group / wait_group
3. **ldmatrix** — 理解 Tensor Core 的数据布局要求 (m8n8 fragment)
4. **mma** — 实际发射 Tensor Core 计算指令
5. **wgmma** — Hopper 架构新的 warp-group 级别 MMA
6. **tma** — Hopper 硬件 DMA 引擎
7. **hgemm_mma** — 把前面所有串起来: 一个完整的高性能 HGEMM kernel

## PTX 内联汇编语法速查

```cuda
asm volatile(
    "ptx_instruction operands;\n"
    : "=约束"(output_var)          // 输出操作数
    : "约束"(input_var)            // 输入操作数
    : "memory"                     // clobbers (可选)
);
```

| 约束 | 含义 | PTX 类型 |
|------|------|----------|
| `"r"` | 32-bit 整数寄存器 | `.u32` / `.s32` |
| `"l"` | 64-bit 整数寄存器 | `.u64` |
| `"f"` | 32-bit 浮点寄存器 | `.f32` |
| `"d"` | 64-bit 浮点寄存器 | `.f64` |
| `"h"` | 16-bit 寄存器 | `.u16` / `.f16` |
| `"n"` | 编译时常量 | 立即数 |

## 与 DeepGEMM 的关系

DeepGEMM 的核心思想:
- 不依赖 CUTLASS 模板地狱，直接用内联 PTX 控制关键路径
- JIT 编译 → PTX → ptxas → SASS，可 dump 中间产物
- 两级累积解决 FP8 精度: 先在 register 中 FP8→FP32 累积小块，再写回

本目录的目标是逐步掌握这些底层指令，最终能读懂和写出类似 DeepGEMM 级别的内核。
