#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>

// ==================== 访存控制 & 性能调优 PTX 指令 ====================
//
// 控制 cache 行为、预取、原子 reduction 等
// 这些指令单个看不起眼，但在高性能 kernel 中对带宽利用至关重要

// ============================================================
// 1. Load 的 cache 修饰符
// ============================================================
// .ca  = Cache at All levels (L1 + L2) — 默认
// .cg  = Cache at Global (只 L2, 跳过 L1) — streaming data
// .cs  = Cache Streaming (低优先级, 不会驱逐常用数据)
// .nc  = Non-Coherent (只读, 走 texture/constant cache path)
// .lu  = Last Use (告诉 cache 这是最后一次用, 可以尽早驱逐)

// [炫技] .ca 是默认行为，等价于普通 *ptr 解引用
__device__ __forceinline__ float ld_global_ca(const float* ptr) {
    float val;
    asm volatile("ld.global.ca.f32 %0, [%1];\n" : "=f"(val) : "l"(ptr));
    return val;
}

// ld.global.cg — 只走 L2 (适合大数据流式访问, 不污染 L1)
__device__ __forceinline__ float ld_global_cg(const float* ptr) {
    float val;
    asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(val) : "l"(ptr));
    return val;
}

// ld.global.cs — Streaming (不缓存, 适合只用一次的数据)
__device__ __forceinline__ float ld_global_cs(const float* ptr) {
    float val;
    asm volatile("ld.global.cs.f32 %0, [%1];\n" : "=f"(val) : "l"(ptr));
    return val;
}

// ld.global.nc — Non-coherent (只读数据, 走 ncache/texture path, 更高带宽)
__device__ __forceinline__ float ld_global_nc(const float* ptr) {
    float val;
    asm volatile("ld.global.nc.f32 %0, [%1];\n" : "=f"(val) : "l"(ptr));
    return val;
}

// ld.global.lu — Last Use (可以提前驱逐)
__device__ __forceinline__ float ld_global_lu(const float* ptr) {
    float val;
    asm volatile("ld.global.lu.f32 %0, [%1];\n" : "=f"(val) : "l"(ptr));
    return val;
}

// 向量化版本: ld.global.cg.v4.f32 (128-bit, 只走 L2)
__device__ __forceinline__ void ld_global_cg_v4(float& r0, float& r1, float& r2, float& r3,
                                                  const void* ptr) {
    asm volatile(
        "ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "l"(ptr)
    );
}

// ============================================================
// 2. Store 的 cache 修饰符
// ============================================================
// .wb = Write-Back (写到 cache, 延迟刷到 DRAM) — 默认
// .wt = Write-Through (同时写 cache 和 DRAM)
// .cs = Streaming (不缓存写入)

// [炫技] .wb 是默认行为，等价于普通 *ptr = val
__device__ __forceinline__ void st_global_wb(float* ptr, float val) {
    asm volatile("st.global.wb.f32 [%0], %1;\n" :: "l"(ptr), "f"(val) : "memory");
}

__device__ __forceinline__ void st_global_wt(float* ptr, float val) {
    asm volatile("st.global.wt.f32 [%0], %1;\n" :: "l"(ptr), "f"(val) : "memory");
}

__device__ __forceinline__ void st_global_cs(float* ptr, float val) {
    asm volatile("st.global.cs.f32 [%0], %1;\n" :: "l"(ptr), "f"(val) : "memory");
}

// ============================================================
// 3. Prefetch — L2 预取
// ============================================================
// 在需要数据之前发出 prefetch, 让数据提前进入 cache
// 适合: 你知道下一轮循环要用哪些数据

// prefetch 到 L1
__device__ __forceinline__ void prefetch_l1(const void* ptr) {
    asm volatile("prefetch.global.L1 [%0];\n" :: "l"(ptr));
}

// prefetch 到 L2
__device__ __forceinline__ void prefetch_l2(const void* ptr) {
    asm volatile("prefetch.global.L2 [%0];\n" :: "l"(ptr));
}

// ============================================================
// 4. Reduction (red.) — 比 atom 更快的原子操作
// ============================================================
// atom.add 返回旧值 (需要 read-modify-write)
// red.add 不返回旧值 (只做 fire-and-forget reduction)
// → red 可以被硬件合并优化, 比 atom 更高效

// red.global.add.f32 — 全局浮点 reduction (不返回旧值)
__device__ __forceinline__ void red_global_add_f32(float* addr, float val) {
    asm volatile(
        "red.global.add.f32 [%0], %1;\n"
        :: "l"(addr), "f"(val)
        : "memory"
    );
}

// red.global.add.s32 — 全局整数 reduction
__device__ __forceinline__ void red_global_add_s32(int* addr, int val) {
    asm volatile(
        "red.global.add.s32 [%0], %1;\n"
        :: "l"(addr), "r"(val)
        : "memory"
    );
}

// red.global.min.s32
__device__ __forceinline__ void red_global_min_s32(int* addr, int val) {
    asm volatile(
        "red.global.min.s32 [%0], %1;\n"
        :: "l"(addr), "r"(val)
        : "memory"
    );
}

// red.global.max.s32
__device__ __forceinline__ void red_global_max_s32(int* addr, int val) {
    asm volatile(
        "red.global.max.s32 [%0], %1;\n"
        :: "l"(addr), "r"(val)
        : "memory"
    );
}

// ============================================================
// 5. 条件选择 (selp) — 无分支 ternary
// ============================================================
// selp.f32 %d, %a, %b, %p → d = p ? a : b
// 避免 warp divergence: 用 selp 替代 if-else

// [炫技] 等价于三元运算符 pred ? a : b，编译器自动生成 selp
__device__ __forceinline__ float selp_f32(float a, float b, bool pred) {
    float result;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.u32 p, %3, 0;\n"
        "  selp.f32 %0, %1, %2, p;\n"
        "}\n"
        : "=f"(result)
        : "f"(a), "f"(b), "r"((int)pred)
    );
    return result;
}

// [炫技] 等价于三元运算符 pred ? a : b，编译器自动生成 selp
__device__ __forceinline__ int selp_s32(int a, int b, bool pred) {
    int result;
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.u32 p, %3, 0;\n"
        "  selp.s32 %0, %1, %2, p;\n"
        "}\n"
        : "=r"(result)
        : "r"(a), "r"(b), "r"((int)pred)
    );
    return result;
}

// ============================================================
// 6. FMA (fused multiply-add) — 控制精度
// ============================================================
// fma.rn.f32 d, a, b, c → d = a*b + c (单次舍入, 更精确)
// mad.f32 不保证是 fused 的

// [炫技] 等价于 fmaf(a, b, c) 或 a*b+c (--fmad=true 时编译器自动 fuse)
__device__ __forceinline__ float fma_rn_f32(float a, float b, float c) {
    float result;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;\n"
        : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

// ============================================================
// 测试 Kernel
// ============================================================
__global__ void memctrl_test_kernel(const float* input, float* output,
                                     float* reduce_out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // 用 .nc load (只读优化)
    float val = ld_global_nc(&input[idx]);

    // 预取下一个 cacheline
    if (idx + 32 < N) {
        prefetch_l2(&input[idx + 32]);
    }

    // selp 替代 if-else (无分支)
    float processed = selp_f32(val * 2.0f, val * 0.5f, val > 0.0f);

    // FMA
    float result = fma_rn_f32(processed, 1.5f, 0.1f);

    // 写入 (streaming, 不缓存)
    st_global_cs(&output[idx], result);

    // 用 red (无返回值 atomic) 累加
    red_global_add_f32(reduce_out, result);
}

// ============================================================
// Host
// ============================================================
int main() {
    const int N = 256;
    float h_input[N], h_output[N];
    float h_reduce = 0.0f;

    for (int i = 0; i < N; i++) h_input[i] = (float)(i - 128);

    float *d_input, *d_output, *d_reduce;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMalloc(&d_reduce, sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_reduce, 0, sizeof(float));

    memctrl_test_kernel<<<1, 256>>>(d_input, d_output, d_reduce, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_reduce, d_reduce, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Memory control PTX test:\n");
    printf("  input[0]=%6.1f → output[0]=%6.2f\n", h_input[0], h_output[0]);
    printf("  input[128]=%6.1f → output[128]=%6.2f\n", h_input[128], h_output[128]);
    printf("  input[255]=%6.1f → output[255]=%6.2f\n", h_input[255], h_output[255]);
    printf("  reduce_sum = %.2f\n", h_reduce);

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_reduce);
    return 0;
}
