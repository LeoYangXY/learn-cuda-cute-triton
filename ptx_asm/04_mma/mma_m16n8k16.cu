#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

// ==================== mma.sync — Tensor Core MMA ====================
//
// mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// 这是 Ampere/Ada (sm_80/sm_89) 上最常用的 MMA 形状
//
// 语义: D = A × B + C
//   A: m16×k16 (FP16, row major)
//   B: k16×n8  (FP16, col major)
//   C: m16×n8  (FP16 或 FP32, row major)
//   D: m16×n8  (同 C)
//
// 【一个 warp (32 threads) 协作完成整个 m16n8k16 MMA】
//
// Register 分配 (per thread):
//   A fragment: 4 × uint32 (RA0-RA3) = 8 half values
//   B fragment: 2 × uint32 (RB0-RB1) = 4 half values
//   C/D fragment (f16): 2 × uint32 (RD0-RD1) = 4 half values
//   C/D fragment (f32): 4 × uint32 (RD0-RD3) = 4 float values
//
// 【数据在 lane 中的分布 (m16n8k16, A matrix)】
//
//   lane 0-3:   A[0:2, 0:8]   (行 0-1, 列 0-7)
//   lane 4-7:   A[2:4, 0:8]   (行 2-3, 列 0-7)
//   ...
//   lane 28-31: A[14:16, 0:8] (行 14-15, 列 0-7)
//   (第二组 k=8-15 在 RA2, RA3 中)
//
// 【DeepGEMM 中的使用】
// DeepGEMM 用多个 mma.sync 组成大 tile:
//   - 外层 for k_tile: cp.async 加载 → ldmatrix 分发 → 多个 mma.sync 累积
//   - 两级累积: FP8 → FP32 (用 f32 accumulator 版本)

// ============================================================
// MMA 指令封装
// ============================================================

// mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// D (f16) = A (f16) × B (f16) + C (f16)
// Inplace update: C 和 D 是同一组寄存器
__device__ __forceinline__ void mma_m16n8k16_f16_f16(
    uint32_t* RD0, uint32_t* RD1,         // D (output, 2 regs = 4 half)
    uint32_t* RA0, uint32_t* RA1,         // A (4 regs = 8 half)
    uint32_t* RA2, uint32_t* RA3,
    uint32_t* RB0, uint32_t* RB1          // B (2 regs = 4 half)
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0, %1}, "
        "{%2, %3, %4, %5}, "
        "{%6, %7}, "
        "{%8, %9};\n"
        : "=r"(RD0[0]), "=r"(RD1[0])
        : "r"(RA0[0]), "r"(RA1[0]), "r"(RA2[0]), "r"(RA3[0]),
          "r"(RB0[0]), "r"(RB1[0]),
          "r"(RD0[0]), "r"(RD1[0])  // C = D (inplace accumulate)
    );
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// D (f32) = A (f16) × B (f16) + C (f32)
// FP32 accumulator 版本 (精度更高, DeepGEMM 必用)
__device__ __forceinline__ void mma_m16n8k16_f16_f32(
    uint32_t* RD0, uint32_t* RD1,         // D (output, 4 regs = 4 float)
    uint32_t* RD2, uint32_t* RD3,
    uint32_t* RA0, uint32_t* RA1,         // A (4 regs)
    uint32_t* RA2, uint32_t* RA3,
    uint32_t* RB0, uint32_t* RB1          // B (2 regs)
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3}, "
        "{%4,  %5,  %6,  %7}, "
        "{%8,  %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(RD0[0]), "=r"(RD1[0]), "=r"(RD2[0]), "=r"(RD3[0])
        : "r"(RA0[0]), "r"(RA1[0]), "r"(RA2[0]), "r"(RA3[0]),
          "r"(RB0[0]), "r"(RB1[0]),
          "r"(RD0[0]), "r"(RD1[0]), "r"(RD2[0]), "r"(RD3[0])
    );
}

// ============================================================
// ldmatrix (搭配 MMA 使用)
// ============================================================

__device__ __forceinline__ uint32_t cvta_to_shared(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr)
        : "l"(ptr)
    );
    return addr;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& R0, uint32_t& R1,
                                             uint32_t& R2, uint32_t& R3,
                                             uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& R0, uint32_t& R1,
                                                   uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(R0), "=r"(R1)
        : "r"(addr)
    );
}

// ============================================================
// 完整示例: 16×8 HGEMM tile 用 mma.sync
// ============================================================
// A: [16, 16] half (row major) in SMEM
// B: [16, 8]  half (col major) in SMEM
// C: [16, 8]  half (output)
// 一个 warp 完成整个计算

__global__ void mma_demo_kernel(const half* A_global, const half* B_global,
                                 half* C_global) {
    __shared__ half smem_A[16 * 16];  // A: 16 rows × 16 cols
    __shared__ half smem_B[16 * 8];   // B: 16 rows × 8 cols (col major → 8 cols × 16 rows)

    int tid = threadIdx.x;
    int lane = tid % 32;

    // 协作加载 A 和 B 到 SMEM
    for (int i = tid; i < 16 * 16; i += 32) smem_A[i] = A_global[i];
    for (int i = tid; i < 16 * 8; i += 32)  smem_B[i] = B_global[i];
    __syncthreads();

    // ===== ldmatrix 加载 A fragment (x4) =====
    // 每个 lane 提供一个地址; ldmatrix 从 32 个地址读 32 行 (每行 8 half = 16B)
    // 对于 m16n8k16: A 有 16 行 × 16 列 = 需要 x4 (4 个 m8n8 = 32 行)
    // lane i 的地址: row = i % 16, col_start = (i / 16) * 8
    int a_row = lane % 16;
    int a_col = (lane / 16) * 8;
    uint32_t a_addr = cvta_to_shared(&smem_A[a_row * 16 + a_col]);

    uint32_t RA[4];
    ldmatrix_x4(RA[0], RA[1], RA[2], RA[3], a_addr);

    // ===== ldmatrix.trans 加载 B fragment (x2) =====
    // B 是 col major: B[k][n], 需要 trans 来适配 mma 的 col 布局
    // m16n8k16 的 B: k16×n8, 每个 lane 需要 2 个 uint32
    int b_row = lane % 16;
    uint32_t b_addr = cvta_to_shared(&smem_B[b_row * 8]);

    uint32_t RB[2];
    ldmatrix_x2_trans(RB[0], RB[1], b_addr);

    // ===== 初始化 accumulator (D = 0) =====
    uint32_t RD[2] = {0, 0};

    // ===== 执行 MMA =====
    mma_m16n8k16_f16_f16(&RD[0], &RD[1],
                          &RA[0], &RA[1], &RA[2], &RA[3],
                          &RB[0], &RB[1]);

    // ===== 写回结果 =====
    // D fragment 分布: 每个 thread 持有 C[16,8] 中的某些元素
    // lane i → row = (i / 4) * 2 + (i % 2), col = (i % 4) / 2 * ... (复杂映射)
    // 简化: 直接 store 到 global (实际应用中用 stmatrix 或手动分发)
    // 每个 thread 有 2 个 uint32 = 4 个 half
    half2* out_ptr = reinterpret_cast<half2*>(C_global);
    int out_offset = lane * 2;  // 简化映射
    if (out_offset + 1 < 16 * 8 / 2) {
        out_ptr[out_offset] = *reinterpret_cast<half2*>(&RD[0]);
        out_ptr[out_offset + 1] = *reinterpret_cast<half2*>(&RD[1]);
    }
}

// ============================================================
// Host
// ============================================================
int main() {
    const int M = 16, K = 16, N = 8;
    half *h_A = new half[M * K], *h_B = new half[K * N], *h_C = new half[M * N];

    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(1.0f);

    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(half));
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(half));

    mma_demo_kernel<<<1, 32>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost);
    printf("C[0][0] = %.1f (expected 16.0 = dot product of 16 ones)\n",
           __half2float(h_C[0]));

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
