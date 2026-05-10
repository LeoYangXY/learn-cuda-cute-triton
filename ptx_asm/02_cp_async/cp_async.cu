#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

// ==================== cp.async 异步数据搬运 ====================
//
// cp.async 是 Ampere (sm_80+) 引入的硬件指令:
//   GMEM → SMEM 的异步拷贝，绕过 register file
//
// 【为什么需要它？】
// 传统方式: GMEM → Register → SMEM (两步，占用 register)
// cp.async:  GMEM → SMEM (一步，不经过 register，由硬件 DMA 完成)
//
// 【面试关键点】
// 1. cp.async 允许计算和数据搬运重叠 (software pipelining)
// 2. commit_group / wait_group 实现多级流水线
// 3. 支持 4B / 8B / 16B 三种粒度 (.ca 支持 4/8/16, .cg 只支持 16)
// 4. 必须配合 __syncthreads() 或 cp.async.wait 才能保证数据可见
//
// 【DeepGEMM 中的使用】
// DeepGEMM 用 cp.async 实现 double/triple buffering:
//   Stage 0: 计算当前 tile (数据已在 SMEM)
//   Stage 1: 异步加载下一个 tile (cp.async in flight)
//   → commit → wait → swap buffers → 下一轮

// ============================================================
// cp.async 基础指令封装
// ============================================================

// 提交一组异步拷贝
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// 等待所有异步拷贝完成
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

// 等待直到最多还有 N 组未完成
// 例如 wait_group<1> = 等待直到只剩最后 1 组还在飞 (即前面的都完成了)
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ============================================================
// cp.async.ca — cache all (支持 4/8/16 字节)
// ============================================================
// dst: shared memory 地址 (uint32_t, 由 cvta.to.shared 得到)
// src: global memory 地址 (uint64_t 指针)
// bytes: 4, 8, 或 16

// 16 字节版本 (最常用: 一次搬 128 bits)
__device__ __forceinline__ void cp_async_ca_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

// 8 字节版本
__device__ __forceinline__ void cp_async_ca_8B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global.L2::128B [%0], [%1], 8;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

// 4 字节版本
__device__ __forceinline__ void cp_async_ca_4B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

// ============================================================
// cp.async.cg — cache global (只 L2, 跳过 L1, 只支持 16B)
// ============================================================
// 适合流式数据（只用一次就扔掉的）
__device__ __forceinline__ void cp_async_cg_16B(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr)
        : "memory"
    );
}

// ============================================================
// 辅助: smem 地址转换
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
// 示例: 用 cp.async 实现 GMEM → SMEM 异步拷贝 + double buffering
// ============================================================
// 简化的 SGEMM 数据加载阶段 (只演示 cp.async pipeline, 不做计算)

#define BLOCK_M 128
#define BLOCK_K 32
#define NUM_STAGES 2  // double buffering

__global__ void cp_async_demo_kernel(const float* A, float* smem_debug,
                                      int M, int K) {
    // 每个 block 负责 A 的 BLOCK_M 行
    extern __shared__ float smem[];  // [NUM_STAGES][BLOCK_M][BLOCK_K]

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row_start = bid * BLOCK_M;

    // 每个 thread 负责搬运多少字节: 每次 16B = 4 floats
    // 总共需要搬: BLOCK_M * BLOCK_K * 4 bytes = 128 * 32 * 4 = 16384 bytes
    // 每个 thread 每次搬 16 bytes, 256 threads → 每轮搬 256*16 = 4096 bytes
    // 需要 16384/4096 = 4 轮

    const int elems_per_thread = 4;  // 16B / 4B = 4 floats
    const int total_elems = BLOCK_M * BLOCK_K;  // per stage
    const int iters_per_stage = total_elems / (blockDim.x * elems_per_thread);

    // ===== Stage 0: 发射第一个 tile 的 cp.async =====
    float* smem_stage0 = smem;
    for (int iter = 0; iter < iters_per_stage; ++iter) {
        int elem_idx = (iter * blockDim.x + tid) * elems_per_thread;
        int row = elem_idx / BLOCK_K;
        int col = elem_idx % BLOCK_K;

        if (row_start + row < M && col < K) {
            uint32_t dst = cvta_to_shared(&smem_stage0[elem_idx]);
            const void* src = &A[(row_start + row) * K + col];
            cp_async_ca_16B(dst, src);
        }
    }
    cp_async_commit_group();

    // ===== Stage 1: 可以发射下一个 tile (如果有) =====
    // ... (省略, 实际中在这里发射下一个 K-tile 的加载)

    // ===== 等待 stage 0 完成 =====
    cp_async_wait_all();
    asm volatile("bar.sync 0;\n" ::: "memory");

    // 现在 smem_stage0 中的数据已就绪，可以安全读取
    // 简单验证: 将 smem 内容写回 global memory
    for (int i = tid; i < total_elems; i += blockDim.x) {
        if (i < total_elems) {
            smem_debug[bid * total_elems + i] = smem_stage0[i];
        }
    }
}

// ============================================================
// Host 测试
// ============================================================
int main() {
    const int M = 128, K = 32;
    float* h_A = new float[M * K];
    for (int i = 0; i < M * K; i++) h_A[i] = (float)i;

    float *d_A, *d_debug;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_debug, M * K * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);

    // smem size = NUM_STAGES * BLOCK_M * BLOCK_K * sizeof(float)
    size_t smem_size = NUM_STAGES * BLOCK_M * BLOCK_K * sizeof(float);
    cp_async_demo_kernel<<<1, 256, smem_size>>>(d_A, d_debug, M, K);
    cudaDeviceSynchronize();

    // 验证
    float* h_debug = new float[M * K];
    cudaMemcpy(h_debug, d_debug, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    bool pass = true;
    for (int i = 0; i < M * K; i++) {
        if (h_debug[i] != h_A[i]) { pass = false; break; }
    }
    printf("cp.async test: %s\n", pass ? "PASSED" : "FAILED");

    delete[] h_A;
    delete[] h_debug;
    cudaFree(d_A);
    cudaFree(d_debug);
    return 0;
}
