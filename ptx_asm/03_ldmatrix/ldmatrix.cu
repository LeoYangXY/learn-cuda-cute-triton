#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

// ==================== ldmatrix 指令 ====================
//
// ldmatrix 是 Tensor Core 专用的 SMEM → Register 加载指令
// 它把 shared memory 中的数据按 Tensor Core 需要的布局分发到各 lane 的寄存器中
//
// 【为什么不能用普通 load？】
// Tensor Core 的 mma.sync.m16n8k16 要求:
//   - A 矩阵 fragment: 每个 thread 持有 4 个 uint32 (= 8 个 half)
//   - B 矩阵 fragment: 每个 thread 持有 2 个 uint32 (= 4 个 half)
//   - 数据在 32 个 lane 中的分布必须满足特定的 m8n8 布局
//
// ldmatrix 一条指令完成:
//   1. 从 SMEM 读取数据
//   2. 按 m8n8 pattern 分发到 warp 的 32 个 lane
//   3. 支持可选的 transpose (.trans)
//
// 【变体】
// ldmatrix.sync.aligned.x1.m8n8.shared.b16 — 加载 1 个 8×8 矩阵 (128B)
// ldmatrix.sync.aligned.x2.m8n8.shared.b16 — 加载 2 个 8×8 矩阵 (256B)
// ldmatrix.sync.aligned.x4.m8n8.shared.b16 — 加载 4 个 8×8 矩阵 (512B)
//
// 【寻址规则】
// warp 中的每个 lane 提供一个 SMEM 地址 → 读出那一行 (8 个 half = 16B)
// 但不是每个 lane 的地址都被使用:
//   x1: lane 0-7 的地址被使用 (8 行)
//   x2: lane 0-15 的地址被使用 (16 行)
//   x4: lane 0-31 的地址全被使用 (32 行)

// ============================================================
// ldmatrix 封装
// ============================================================

// x1: 加载 1 个 m8n8 fragment → 1 个 uint32 register
__device__ __forceinline__ void ldmatrix_x1(uint32_t& R, uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
        : "=r"(R)
        : "r"(smem_addr)
    );
}

// x2: 加载 2 个 m8n8 fragment → 2 个 uint32 register
__device__ __forceinline__ void ldmatrix_x2(uint32_t& R0, uint32_t& R1, uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(R0), "=r"(R1)
        : "r"(smem_addr)
    );
}

// x4: 加载 4 个 m8n8 fragment → 4 个 uint32 register
__device__ __forceinline__ void ldmatrix_x4(uint32_t& R0, uint32_t& R1,
                                             uint32_t& R2, uint32_t& R3,
                                             uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
        : "r"(smem_addr)
    );
}

// 带 transpose 的版本 (.trans):
// 将列主序 (column-major) 的数据转置为行主序分发
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& R0, uint32_t& R1,
                                                   uint32_t& R2, uint32_t& R3,
                                                   uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
        : "r"(smem_addr)
    );
}

// ============================================================
// 辅助函数
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

// ============================================================
// 示例: ldmatrix_x4 加载 A 矩阵 fragment
// ============================================================
// 场景: A 是 16×16 的 half 矩阵, 已存入 shared memory (行主序)
// 使用 ldmatrix_x4 加载为 mma.m16n8k16 所需的 A fragment
//
// 布局:
//   A[16][16] half → 每行 16 half = 32 bytes
//   ldmatrix_x4 需要每个 lane 提供一行(8 half=16B)的起始地址
//   32 个 lane → 32 行 → 但 A 只有 16 行
//   实际上 lane 0-7 → A 的前 8 行, lane 8-15 → A 的后 8 行
//   lane 16-31 → A 的前/后 8 行的后半部分 (col 8-15)

__global__ void ldmatrix_demo_kernel(const half* A_global, uint32_t* regs_out) {
    __shared__ half smem_A[16 * 16];  // 16 rows × 16 cols

    int tid = threadIdx.x;  // 0-31 (一个 warp)
    int lane = tid % 32;

    // 加载 A 到 shared memory (协作加载)
    // 16×16 = 256 halfs, 32 threads, 每 thread 搬 8 个 half
    for (int i = lane * 8; i < 256 && i < (lane + 1) * 8; i++) {
        smem_A[i] = A_global[i];
    }
    __syncthreads();

    // ldmatrix_x4 寻址:
    // 每个 lane 提供自己要读的那一行的地址
    // lane 0-7:  row 0-7, col 0-7  (前 8 行, 左半)
    // lane 8-15: row 8-15, col 0-7 (后 8 行, 左半)
    // lane 16-23: row 0-7, col 8-15 (前 8 行, 右半)
    // lane 24-31: row 8-15, col 8-15 (后 8 行, 右半)
    int row = lane % 16;
    int col_group = (lane / 16) * 8;  // 0 or 8
    uint32_t smem_addr = cvta_to_shared(&smem_A[row * 16 + col_group]);

    uint32_t R0, R1, R2, R3;
    ldmatrix_x4(R0, R1, R2, R3, smem_addr);

    // 输出到 global memory 查看分布
    regs_out[lane * 4 + 0] = R0;
    regs_out[lane * 4 + 1] = R1;
    regs_out[lane * 4 + 2] = R2;
    regs_out[lane * 4 + 3] = R3;
}

// ============================================================
// Host 测试
// ============================================================
int main() {
    const int size = 16 * 16;
    half* h_A = new half[size];
    for (int i = 0; i < size; i++) h_A[i] = __float2half((float)i);

    half* d_A;
    uint32_t* d_regs;
    cudaMalloc(&d_A, size * sizeof(half));
    cudaMalloc(&d_regs, 32 * 4 * sizeof(uint32_t));
    cudaMemcpy(d_A, h_A, size * sizeof(half), cudaMemcpyHostToDevice);

    ldmatrix_demo_kernel<<<1, 32>>>(d_A, d_regs);
    cudaDeviceSynchronize();

    uint32_t h_regs[128];
    cudaMemcpy(h_regs, d_regs, 128 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 打印 lane 0 的 4 个 register (每个 = 2 个 half)
    printf("Lane 0 registers:\n");
    for (int i = 0; i < 4; i++) {
        half2 v = *reinterpret_cast<half2*>(&h_regs[i]);
        printf("  R%d: [%.1f, %.1f]\n", i,
               __half2float(v.x), __half2float(v.y));
    }

    delete[] h_A;
    cudaFree(d_A);
    cudaFree(d_regs);
    return 0;
}
