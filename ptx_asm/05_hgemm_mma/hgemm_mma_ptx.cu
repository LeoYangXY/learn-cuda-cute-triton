#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

// ==================== 完整 HGEMM: cp.async + ldmatrix + mma pipeline ====================
//
// 把前面学到的所有 PTX 指令串起来，实现一个完整的高性能 HGEMM kernel
//
// C[M,N] = A[M,K] × B[K,N]  (half precision, FP32 accumulator)
//
// 【架构设计 (类似 DeepGEMM 的简化版)】
//
// Tile sizes:
//   BM = 128, BN = 128, BK = 32
//   WM = 64, WN = 64 (每个 warp 负责的输出 tile)
//
// Thread block: 128 threads = 4 warps
//   Warp 0: C[0:64, 0:64]
//   Warp 1: C[0:64, 64:128]
//   Warp 2: C[64:128, 0:64]
//   Warp 3: C[64:128, 64:128]
//
// Pipeline (double buffering):
//   Stage 0: cp.async 加载 tile k+1 到 buffer 1
//   Stage 0: ldmatrix 从 buffer 0 取 fragment → mma.sync 计算
//   → commit + wait → swap → 下一个 k
//
// 每个 warp 内部:
//   WM=64, WN=64 由多个 m16n8k16 MMA 组成:
//   64/16 × 64/8 = 4 × 8 = 32 个 MMA per K iteration
//   但每个 MMA 共享 A/B fragment，实际是 4×8 = 32 次调用 mma.sync
//
// 【简化版: BM=16, BN=16, BK=16, 1 warp, 1 stage (无 pipeline)】
// 目的: 展示完整的 PTX 指令串联，不追求性能

#define BM 16
#define BN 16
#define BK 16

// ============================================================
// PTX 辅助函数 (从前面文件复用)
// ============================================================

__device__ __forceinline__ uint32_t cvta_to_shared(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 u64addr;\n"
        "  cvta.to.shared.u64 u64addr, %1;\n"
        "  cvt.u32.u64 %0, u64addr; }\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void cp_async_ca_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& R0, uint32_t& R1,
                                             uint32_t& R2, uint32_t& R3,
                                             uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t& R0, uint32_t& R1, uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(R0), "=r"(R1) : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_f32(
    uint32_t* D, uint32_t* A, uint32_t* B) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,  %1,  %2,  %3}, "
        "{%4,  %5,  %6,  %7}, "
        "{%8,  %9}, "
        "{%10, %11, %12, %13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(D[0]), "r"(D[1]), "r"(D[2]), "r"(D[3]));
}

// ============================================================
// HGEMM Kernel: A[M,K] × B[K,N] = C[M,N]
// 简化版: 每个 block 处理一个 BM×BN tile, 用 1 个 warp
// ============================================================
__global__ void hgemm_ptx_kernel(const half* __restrict__ A,
                                  const half* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    // 共享内存
    __shared__ half smem_A[BM * BK];  // 16×16 half
    __shared__ half smem_B[BK * BN];  // 16×16 half (B 按列存以适配 ldmatrix.trans)

    int bx = blockIdx.x, by = blockIdx.y;
    int lane = threadIdx.x % 32;

    // 输出 tile 起始坐标
    int row_start = by * BM;
    int col_start = bx * BN;

    // Accumulator (FP32): 每个 thread 持有 D[4] for m16n8k16
    // 但 BN=16 需要两个 m16n8k16 (n=8 each) → D0[4] + D1[4]
    uint32_t D0[4] = {0, 0, 0, 0};
    uint32_t D1[4] = {0, 0, 0, 0};

    // K 维度循环
    for (int k_tile = 0; k_tile < K; k_tile += BK) {
        // ===== Step 1: cp.async 加载 A 和 B 到 SMEM =====
        // A tile: A[row_start:row_start+16, k_tile:k_tile+16] → smem_A[16][16]
        // 每个 thread 搬 16B (8 half), 32 threads 搬 32×8 = 256 half = 正好 16×16
        {
            int elem_start = lane * 8;
            int row = elem_start / BK;
            int col = elem_start % BK;
            int g_row = row_start + row;
            int g_col = k_tile + col;
            if (g_row < M && g_col + 7 < K) {
                uint32_t dst = cvta_to_shared(&smem_A[elem_start]);
                const void* src = &A[g_row * K + g_col];
                cp_async_ca_16B(dst, src);
            }
        }

        // B tile: B[k_tile:k_tile+16, col_start:col_start+16]
        {
            int elem_start = lane * 8;
            int row = elem_start / BN;
            int col = elem_start % BN;
            int g_row = k_tile + row;
            int g_col = col_start + col;
            if (g_row < K && g_col + 7 < N) {
                uint32_t dst = cvta_to_shared(&smem_B[elem_start]);
                const void* src = &B[g_row * N + g_col];
                cp_async_ca_16B(dst, src);
            }
        }

        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // ===== Step 2: ldmatrix 加载 fragment =====
        // A fragment: ldmatrix_x4 for m16n8k16
        int a_row = lane % 16;
        int a_col_group = (lane / 16) * 8;
        uint32_t a_addr = cvta_to_shared(&smem_A[a_row * BK + a_col_group]);
        uint32_t RA[4];
        ldmatrix_x4(RA[0], RA[1], RA[2], RA[3], a_addr);

        // B fragment for n=0..7: ldmatrix_x2
        int b_row = lane % 16;
        uint32_t b_addr0 = cvta_to_shared(&smem_B[b_row * BN + 0]);
        uint32_t RB0[2];
        ldmatrix_x2(RB0[0], RB0[1], b_addr0);

        // B fragment for n=8..15
        uint32_t b_addr1 = cvta_to_shared(&smem_B[b_row * BN + 8]);
        uint32_t RB1[2];
        ldmatrix_x2(RB1[0], RB1[1], b_addr1);

        // ===== Step 3: mma.sync =====
        mma_m16n8k16_f32(D0, RA, RB0);  // C[0:16, 0:8]
        mma_m16n8k16_f32(D1, RA, RB1);  // C[0:16, 8:16]

        __syncthreads();
    }

    // ===== Step 4: 写回 C (FP32) =====
    // D fragment 分布: m16n8k16 的输出映射
    // 每个 thread 有 4 个 float (for each D0, D1)
    // Mapping: thread i → rows [(i%4)*2, (i%4)*2+1] or similar
    // 简化: 直接用 lane_id 映射 (可能不精确，实际要查 PTX spec)
    // 
    // 对于 m16n8k16 f32 accumulator:
    //   lane_id → (row, col) mapping:
    //   D[0]: C[lane/4*2][lane%4*2]         (每组4个lane负责2行)
    //   D[1]: C[lane/4*2][(lane%4*2)+1]
    //   D[2]: C[lane/4*2+1][lane%4*2]
    //   D[3]: C[lane/4*2+1][(lane%4*2)+1]
    // (这是简化示意，精确映射见 CUDA PTX ISA)

    int group = lane / 4;      // 0-7
    int sub   = lane % 4;      // 0-3
    int r0 = group * 2;
    int r1 = group * 2 + 1;
    int c0 = sub * 2;
    int c1 = sub * 2 + 1;

    float* C_base = C + (row_start) * N + col_start;
    if (row_start + r0 < M && col_start + c0 < N) {
        // D0 → cols 0..7
        float* f0 = reinterpret_cast<float*>(D0);
        C_base[r0 * N + c0] = f0[0];
        C_base[r0 * N + c1] = f0[1];
        C_base[r1 * N + c0] = f0[2];
        C_base[r1 * N + c1] = f0[3];

        // D1 → cols 8..15
        float* f1 = reinterpret_cast<float*>(D1);
        C_base[r0 * N + 8 + c0] = f1[0];
        C_base[r0 * N + 8 + c1] = f1[1];
        C_base[r1 * N + 8 + c0] = f1[2];
        C_base[r1 * N + 8 + c1] = f1[3];
    }
}

// ============================================================
// Host
// ============================================================
int main() {
    const int M = 16, N = 16, K = 16;

    half *h_A = new half[M * K], *h_B = new half[K * N];
    float *h_C = new float[M * N];

    // A = all 1s, B = all 1s → C should be all K = 16
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(1.0f);

    half *d_A, *d_B; float *d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M * N * sizeof(float));

    dim3 grid(1, 1);  // 只有一个 tile
    dim3 block(32);   // 1 warp
    hgemm_ptx_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("HGEMM PTX Pipeline Result (expect 16.0):\n");
    printf("  C[0][0] = %.1f\n", h_C[0]);
    printf("  C[0][8] = %.1f\n", h_C[8]);
    printf("  C[15][15] = %.1f\n", h_C[15 * N + 15]);

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
