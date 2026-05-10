#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// ==================== PTX 内联汇编基础 ====================
//
// 本文件覆盖:
// 1. 基本语法和约束修饰符
// 2. 特殊寄存器读取 (%tid, %ctaid, %nctaid, %clock)
// 3. 向量化 load/store (ld.global.v4, st.global.v4)
// 4. 共享内存地址转换 (cvta.to.shared)
// 5. 内存屏障 (bar.sync, fence)
// 6. 原子操作 (atom.global.add)
//
// 【面试价值】
// - 理解 PTX 是 CUDA 的中间表示，类似 LLVM IR
// - 知道什么时候需要嵌 PTX: 编译器做不好的事（cp.async、mma、ldmatrix）
// - 读懂别人的 PTX 代码（DeepGEMM、FlashAttention 都大量使用）

// ============================================================
// 1. 读取特殊寄存器
// ============================================================
// PTX 中 %tid.x 对应 threadIdx.x，但有时你需要直接读它来避免编译器多余的 mov

__device__ __forceinline__ uint32_t get_thread_id_x() {
    uint32_t tid;
    asm volatile("mov.u32 %0, %tid.x;\n" : "=r"(tid));
    return tid;
}

__device__ __forceinline__ uint32_t get_block_id_x() {
    uint32_t bid;
    asm volatile("mov.u32 %0, %ctaid.x;\n" : "=r"(bid));
    return bid;
}

__device__ __forceinline__ uint32_t get_grid_dim_x() {
    uint32_t gdim;
    asm volatile("mov.u32 %0, %nctaid.x;\n" : "=r"(gdim));
    return gdim;
}

// 读取 GPU 时钟 (用于 kernel 内部 profiling)
__device__ __forceinline__ uint32_t get_clock() {
    uint32_t clk;
    asm volatile("mov.u32 %0, %clock;\n" : "=r"(clk));
    return clk;
}

__device__ __forceinline__ uint64_t get_clock64() {
    uint64_t clk;
    asm volatile("mov.u64 %0, %clock64;\n" : "=l"(clk));
    return clk;
}

// 读取 SM ID (判断当前在哪个 SM 上运行)
__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(smid));
    return smid;
}

// 读取 warp ID
__device__ __forceinline__ uint32_t get_warpid() {
    uint32_t warpid;
    asm volatile("mov.u32 %0, %warpid;\n" : "=r"(warpid));
    return warpid;
}

// 读取 lane ID (warp 内的线程编号，0-31)
__device__ __forceinline__ uint32_t get_laneid() {
    uint32_t laneid;
    asm volatile("mov.u32 %0, %laneid;\n" : "=r"(laneid));
    return laneid;
}

// ============================================================
// 2. 共享内存地址转换
// ============================================================
// PTX 的 shared memory 指令需要 .shared 地址空间的指针
// 从 generic pointer 转为 shared pointer: cvta.to.shared

__device__ __forceinline__ uint32_t smem_ptr(const void* ptr) {
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
// 3. 向量化全局内存访问
// ============================================================
// ld.global.v4.f32 → 一次读 128 bits (4 个 float)
// 等价于 float4 指针解引用，但保证了特定的 cache 行为

__device__ __forceinline__ void ld_global_v4_f32(float& r0, float& r1, float& r2, float& r3,
                                                  const void* ptr) {
    asm volatile(
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "l"(ptr)
    );
}

__device__ __forceinline__ void st_global_v4_f32(void* ptr,
                                                  float r0, float r1, float r2, float r3) {
    asm volatile(
        "st.global.v4.f32 [%0], {%1, %2, %3, %4};\n"
        :: "l"(ptr), "f"(r0), "f"(r1), "f"(r2), "f"(r3)
    );
}

// Cache 修饰符版本:
// .ca = cache at all levels (L1 + L2)
// .cg = cache at global level (只 L2, 跳过 L1)
// .cs = cache streaming (暗示不会重用，给 eviction 优先级低)
__device__ __forceinline__ void ld_global_cg_v4_f32(float& r0, float& r1, float& r2, float& r3,
                                                     const void* ptr) {
    asm volatile(
        "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "l"(ptr)
    );
}

// ============================================================
// 4. 内存屏障和同步
// ============================================================

// bar.sync 0 等价于 __syncthreads()
__device__ __forceinline__ void bar_sync() {
    asm volatile("bar.sync 0;\n" ::: "memory");
}

// 命名 barrier (barrier ID 可选 0-15)
__device__ __forceinline__ void bar_sync_id(int id) {
    asm volatile("bar.sync %0;\n" :: "r"(id) : "memory");
}

// fence.sc.gpu — 全局内存 fence (sequentially consistent)
__device__ __forceinline__ void fence_sc_gpu() {
    asm volatile("fence.sc.gpu;\n" ::: "memory");
}

// membar.gl — global memory barrier (保证之前的写对其他 SM 可见)
__device__ __forceinline__ void membar_gl() {
    asm volatile("membar.gl;\n" ::: "memory");
}

// ============================================================
// 5. 原子操作
// ============================================================

// atom.global.add.f32 — 全局内存浮点原子加
__device__ __forceinline__ float atomic_add_f32(float* addr, float val) {
    float ret;
    asm volatile(
        "atom.global.add.f32 %0, [%1], %2;\n"
        : "=f"(ret)
        : "l"(addr), "f"(val)
        : "memory"
    );
    return ret;
}

// atom.global.min.s32 — 全局内存整数原子最小值
__device__ __forceinline__ int atomic_min_s32(int* addr, int val) {
    int ret;
    asm volatile(
        "atom.global.min.s32 %0, [%1], %2;\n"
        : "=r"(ret)
        : "l"(addr), "r"(val)
        : "memory"
    );
    return ret;
}

// ============================================================
// 6. Warp 投票和 shuffle
// ============================================================

// vote.sync.ballot — 收集 warp 内所有线程的 predicate
__device__ __forceinline__ uint32_t ballot_sync(uint32_t mask, int pred) {
    uint32_t result;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.u32 p, %1, 0;\n"
        "  vote.sync.ballot.b32 %0, p, %2;\n"
        "}\n"
        : "=r"(result)
        : "r"(pred), "r"(mask)
    );
    return result;
}

// shfl.sync.bfly — butterfly shuffle (等价于 __shfl_xor_sync)
__device__ __forceinline__ float shfl_xor_f32(float val, int lane_mask, uint32_t mask = 0xffffffff) {
    float result;
    asm volatile(
        "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, %3;\n"
        : "=f"(result)
        : "f"(val), "r"(lane_mask), "r"(mask)
    );
    return result;
}

// shfl.sync.down — 向下 shuffle (等价于 __shfl_down_sync)
__device__ __forceinline__ float shfl_down_f32(float val, int offset, uint32_t mask = 0xffffffff) {
    float result;
    asm volatile(
        "shfl.sync.down.b32 %0, %1, %2, 0x1f, %3;\n"
        : "=f"(result)
        : "f"(val), "r"(offset), "r"(mask)
    );
    return result;
}

// ============================================================
// 测试 Kernel
// ============================================================

__global__ void ptx_basics_test_kernel(const float* input, float* output, int N) {
    // 用 PTX 读 tid 和 bid
    uint32_t tid = get_thread_id_x();
    uint32_t bid = get_block_id_x();
    uint32_t lane = get_laneid();
    uint32_t sm = get_smid();

    int idx = bid * blockDim.x + tid;
    if (idx >= N) return;

    // 计时
    uint32_t start = get_clock();

    // 读取数据
    float val = input[idx];

    // Warp reduce sum (用 PTX shuffle)
    float sum = val;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += shfl_xor_f32(sum, offset);
    }

    // 只有 lane 0 写出 warp sum
    if (lane == 0) {
        atomic_add_f32(&output[bid], sum);
    }

    uint32_t end = get_clock();

    // Thread 0 打印信息
    if (idx == 0) {
        printf("SM=%u, warp_sum=%.2f, cycles=%u\n", sm, sum, end - start);
    }
}

// ============================================================
// Host 入口
// ============================================================
int main() {
    const int N = 256;
    float h_input[N], h_output[1] = {0.0f};

    for (int i = 0; i < N; i++) h_input[i] = 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));

    ptx_basics_test_kernel<<<1, 256>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %.2f (expected 256.00)\n", h_output[0]);

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
