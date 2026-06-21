#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdint.h>

// ==================== 数据类型转换 PTX 指令 ====================
//
// 量化/混合精度中必用: FP32 ↔ FP16 ↔ BF16 ↔ FP8
// 编译器通常会自动生成这些, 但手写 PTX 可以:
//   1. 控制舍入模式 (.rn/.rz/.rm/.rp)
//   2. 控制饱和行为 (.satfinite)
//   3. 批量转换时减少指令数

// ============================================================
// 1. FP32 ↔ FP16
// ============================================================

// [炫技] 等价于 __float2half_rn(val)
__device__ __forceinline__ uint16_t cvt_f32_to_f16_rn(float val) {
    uint16_t result;
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(result) : "f"(val));
    return result;
}

// [炫技] 等价于 __half2float(val)
__device__ __forceinline__ float cvt_f16_to_f32(uint16_t val) {
    float result;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(result) : "h"(val));
    return result;
}

// 2个 FP32 打包成 1个 packed FP16x2 (uint32)
__device__ __forceinline__ uint32_t cvt_2xf32_to_f16x2_rn(float a, float b) {
    uint32_t result;
    asm volatile(
        "{\n"
        "  .reg .f16 h0, h1;\n"
        "  cvt.rn.f16.f32 h0, %1;\n"
        "  cvt.rn.f16.f32 h1, %2;\n"
        "  mov.b32 %0, {h0, h1};\n"
        "}\n"
        : "=r"(result)
        : "f"(a), "f"(b)
    );
    return result;
}

// packed FP16x2 解包为 2个 FP32
__device__ __forceinline__ void cvt_f16x2_to_2xf32(uint32_t packed, float& a, float& b) {
    asm volatile(
        "{\n"
        "  .reg .f16 h0, h1;\n"
        "  mov.b32 {h0, h1}, %2;\n"
        "  cvt.f32.f16 %0, h0;\n"
        "  cvt.f32.f16 %1, h1;\n"
        "}\n"
        : "=f"(a), "=f"(b)
        : "r"(packed)
    );
}

// ============================================================
// 2. FP32 ↔ BF16
// ============================================================

// [炫技] 等价于 __float2bfloat16_rn(val)
__device__ __forceinline__ uint16_t cvt_f32_to_bf16_rn(float val) {
    uint16_t result;
    asm volatile("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(result) : "f"(val));
    return result;
}

// [炫技] 等价于 __bfloat162float(val)
__device__ __forceinline__ float cvt_bf16_to_f32(uint16_t val) {
    float result;
    asm volatile("cvt.f32.bf16 %0, %1;\n" : "=f"(result) : "h"(val));
    return result;
}

// 2个 FP32 打包成 BF16x2
__device__ __forceinline__ uint32_t cvt_2xf32_to_bf16x2_rn(float a, float b) {
    uint32_t result;
    asm volatile(
        "{\n"
        "  .reg .b16 h0, h1;\n"
        "  cvt.rn.bf16.f32 h0, %1;\n"
        "  cvt.rn.bf16.f32 h1, %2;\n"
        "  mov.b32 %0, {h0, h1};\n"
        "}\n"
        : "=r"(result)
        : "f"(a), "f"(b)
    );
    return result;
}

// ============================================================
// 3. FP32 → FP8 (E4M3 / E5M2) — DeepGEMM 核心
// ============================================================
// FP8 E4M3: 1 sign + 4 exp + 3 mantissa, max = 448, 用于 forward
// FP8 E5M2: 1 sign + 5 exp + 2 mantissa, max = 57344, 用于 backward
// .satfinite: 超出范围时饱和到 max (而非 INF/NAN)
//
// 注: 这些指令需要 sm_89+ (Ada/Hopper)

// FP32 → FP8 E4M3 (饱和, round to nearest)
__device__ __forceinline__ uint8_t cvt_f32_to_e4m3_rn(float val) {
    uint16_t result;
    asm volatile(
        "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;\n"
        : "=h"(result) : "f"(val)
    );
    return (uint8_t)(result & 0xFF);
}

// FP32 → FP8 E5M2 (饱和, round to nearest)
__device__ __forceinline__ uint8_t cvt_f32_to_e5m2_rn(float val) {
    uint16_t result;
    asm volatile(
        "cvt.rn.satfinite.e5m2x2.f32 %0, %1, %1;\n"
        : "=h"(result) : "f"(val)
    );
    return (uint8_t)(result & 0xFF);
}

// 2个 FP32 → packed E4M3x2 (一个 uint16 装 2 个 FP8)
__device__ __forceinline__ uint16_t cvt_2xf32_to_e4m3x2_rn(float a, float b) {
    uint16_t result;
    asm volatile(
        "cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
        : "=h"(result) : "f"(b), "f"(a)  // 注意: 高位先
    );
    return result;
}

// 2个 FP32 → packed E5M2x2
__device__ __forceinline__ uint16_t cvt_2xf32_to_e5m2x2_rn(float a, float b) {
    uint16_t result;
    asm volatile(
        "cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;\n"
        : "=h"(result) : "f"(b), "f"(a)
    );
    return result;
}

// FP8 E4M3 → FP32 (先转 FP16 再转 FP32 是常见做法)
// 直接转 sm_90 才有, 这里用两步:
__device__ __forceinline__ float cvt_e4m3_to_f32(uint8_t val) {
    // 将 FP8 放到 uint16 的低 8 bit, 用 f16 解释后转 f32
    // 注: 这是近似做法, 精确做法需要查硬件支持
    uint16_t packed = (uint16_t)val;
    float result;
    asm volatile(
        "{\n"
        "  .reg .b16 tmp;\n"
        "  mov.b16 tmp, %1;\n"
        "  cvt.f32.e4m3 %0, tmp;\n"
        "}\n"
        : "=f"(result)
        : "h"(packed)
    );
    return result;
}

// ============================================================
// 4. 舍入模式对比
// ============================================================
// .rn = round to nearest (默认, 最常用)
// .rz = round toward zero (截断)
// .rm = round toward minus infinity (向下)
// .rp = round toward plus infinity (向上)

__device__ __forceinline__ uint16_t cvt_f32_to_f16_rz(float val) {
    uint16_t result;
    asm volatile("cvt.rz.f16.f32 %0, %1;\n" : "=h"(result) : "f"(val));
    return result;
}

__device__ __forceinline__ uint16_t cvt_f32_to_f16_rm(float val) {
    uint16_t result;
    asm volatile("cvt.rm.f16.f32 %0, %1;\n" : "=h"(result) : "f"(val));
    return result;
}

__device__ __forceinline__ uint16_t cvt_f32_to_f16_rp(float val) {
    uint16_t result;
    asm volatile("cvt.rp.f16.f32 %0, %1;\n" : "=h"(result) : "f"(val));
    return result;
}

// ============================================================
// 5. 整数 ↔ 浮点
// ============================================================

// [炫技] 等价于 (float)val，编译器自动生成 cvt 指令
__device__ __forceinline__ float cvt_s32_to_f32(int val) {
    float result;
    asm volatile("cvt.rn.f32.s32 %0, %1;\n" : "=f"(result) : "r"(val));
    return result;
}

// [炫技] 等价于 (int)val 或 __float2int_rz(val)
__device__ __forceinline__ int cvt_f32_to_s32_rz(float val) {
    int result;
    asm volatile("cvt.rzi.s32.f32 %0, %1;\n" : "=r"(result) : "f"(val));
    return result;
}

// ============================================================
// 测试 Kernel
// ============================================================
__global__ void type_convert_test_kernel(const float* input, half* out_f16,
                                          __nv_bfloat16* out_bf16, float* out_back,
                                          int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = input[idx];

    // FP32 → FP16 (PTX)
    uint16_t f16_val = cvt_f32_to_f16_rn(val);
    out_f16[idx] = *reinterpret_cast<half*>(&f16_val);

    // FP32 → BF16 (PTX)
    uint16_t bf16_val = cvt_f32_to_bf16_rn(val);
    out_bf16[idx] = *reinterpret_cast<__nv_bfloat16*>(&bf16_val);

    // FP16 → FP32 round-trip
    float back = cvt_f16_to_f32(f16_val);
    out_back[idx] = back;
}

// ============================================================
// Host
// ============================================================
int main() {
    const int N = 8;
    float h_input[] = {1.0f, 2.5f, 3.14f, -1.0f, 0.0f, 65504.0f, 1e-4f, 0.333f};
    float h_back[N];
    half h_f16[N];

    float *d_input, *d_back;
    half *d_f16;
    __nv_bfloat16 *d_bf16;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_f16, N * sizeof(half));
    cudaMalloc(&d_bf16, N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_back, N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    type_convert_test_kernel<<<1, N>>>(d_input, d_f16, d_bf16, d_back, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_back, d_back, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("FP32 → FP16 → FP32 round-trip:\n");
    for (int i = 0; i < N; i++) {
        printf("  %.6f → %.6f (err=%.2e)\n", h_input[i], h_back[i],
               fabsf(h_input[i] - h_back[i]));
    }

    cudaFree(d_input); cudaFree(d_f16); cudaFree(d_bf16); cudaFree(d_back);
    return 0;
}
