#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <torch/types.h>
#include <torch/extension.h>

#define CEIL(a, b) ((a + b - 1) / (b))

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
  if (((T).options().dtype() != (th_type))) {                \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);    \
  }

// ==================== Quantization Kernels ====================
// MXFP8 Quantization / Dequantization / GEMM
// MXFP4 Quantization / Dequantization / GEMM
// NVFP4 Quantization / Dequantization / GEMV / GEMM
//
// 【任务划分逻辑】
//
// ▸ Block-scaled Quantization (MXFP8/MXFP4/NVFP4):
//   - 输入按 block_size (通常32) 分组，每组共享一个 scale
//   - 每个 thread 处理一个 block: 先求 amax，再计算 scale = amax / max_representable
//   - 量化: q = round(x / scale), 钳位到 [qmin, qmax]
//   - grid = ceil(num_blocks / blockDim.x)
//
// ▸ Dequantization:
//   - 每个 thread 处理一个 block: dequant[i] = q[i] * scale
//   - 完全 elementwise (加上 block scale 索引)
//
// ▸ Quantized GEMM:
//   - 先量化 A 和 B → 用整数/低精度做 tiled GEMM → 反量化结果
//   - 简化实现: 量化 + FP32 累积 + scale 修正
//   - Tiled: 每个 block 处理 C 的一个 tile
//
// ▸ NVFP4 特点:
//   - 4-bit 量化: 每个字节存 2 个值 (pack/unpack)
//   - block_size=32, scale 用 FP8 (E4M3) 表示
//   - GEMV: matrix [M,N] (quantized) × vector [N] → output [M]

// ==================== MXFP8 ====================
// E4M3 format: 1 sign + 4 exponent + 3 mantissa, max = 448.0

#define MXFP8_BLOCK_SIZE 32
#define MXFP8_MAX_VAL 448.0f

// ---- MXFP8 Quantization ----
// input: [N] float32, output: [N] int8 (quantized), scales: [N/block_size] float32
__global__ void mxfp8_quantize_kernel(const float* input, int8_t* output,
                                       float* scales, int N) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = CEIL(N, MXFP8_BLOCK_SIZE);
    if (block_id >= num_blocks) return;

    int start = block_id * MXFP8_BLOCK_SIZE;
    int end = min(start + MXFP8_BLOCK_SIZE, N);

    // Step 1: Find amax in this block
    float amax = 0.0f;
    for (int i = start; i < end; ++i) {
        amax = fmaxf(amax, fabsf(input[i]));
    }

    // Step 2: Compute scale
    float scale = amax / MXFP8_MAX_VAL;
    if (scale == 0.0f) scale = 1.0f;  // avoid div by zero
    scales[block_id] = scale;

    // Step 3: Quantize
    float inv_scale = 1.0f / scale;
    for (int i = start; i < end; ++i) {
        float val = input[i] * inv_scale;
        val = fminf(MXFP8_MAX_VAL, fmaxf(-MXFP8_MAX_VAL, val));
        output[i] = (int8_t)rintf(val);
    }
}

// ---- MXFP8 Dequantization ----
__global__ void mxfp8_dequantize_kernel(const int8_t* input, const float* scales,
                                         float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int block_id = idx / MXFP8_BLOCK_SIZE;
    float scale = scales[block_id];
    output[idx] = (float)input[idx] * scale;
}

// ---- MXFP8 GEMM ----
// Simulated: quantize A and B per-block, do FP32 matmul with scale correction
// A: [M, K], B: [K, N] -> C: [M, N]
#define GEMM_TILE 32

__global__ void mxfp8_gemm_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[GEMM_TILE][GEMM_TILE];
    __shared__ float Bs[GEMM_TILE][GEMM_TILE];

    int row = blockIdx.y * GEMM_TILE + threadIdx.y;
    int col = blockIdx.x * GEMM_TILE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + GEMM_TILE - 1) / GEMM_TILE; ++t) {
        int a_col = t * GEMM_TILE + threadIdx.x;
        int b_row = t * GEMM_TILE + threadIdx.y;

        float a_val = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        float b_val = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Simulate FP8 quantization: round to limited precision
        // E4M3: 3 mantissa bits -> 8 representable values per binade
        a_val = rintf(a_val * 8.0f) / 8.0f;  // simulate 3-bit mantissa
        b_val = rintf(b_val * 8.0f) / 8.0f;

        As[threadIdx.y][threadIdx.x] = a_val;
        Bs[threadIdx.y][threadIdx.x] = b_val;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==================== MXFP4 ====================
// 4-bit format: 1 sign + 2 exponent + 1 mantissa (or E2M1)
// Representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6

#define MXFP4_BLOCK_SIZE 32
#define MXFP4_MAX_VAL 6.0f

// MXFP4 lookup table for dequantization (unsigned 4-bit values 0-7)
__device__ __constant__ float mxfp4_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ---- MXFP4 Quantization ----
// Pack 2 values per byte: low nibble = even index, high nibble = odd index
__global__ void mxfp4_quantize_kernel(const float* input, uint8_t* output,
                                       float* scales, int N) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = CEIL(N, MXFP4_BLOCK_SIZE);
    if (block_id >= num_blocks) return;

    int start = block_id * MXFP4_BLOCK_SIZE;
    int end = min(start + MXFP4_BLOCK_SIZE, N);

    // Find amax
    float amax = 0.0f;
    for (int i = start; i < end; ++i) {
        amax = fmaxf(amax, fabsf(input[i]));
    }

    float scale = amax / MXFP4_MAX_VAL;
    if (scale == 0.0f) scale = 1.0f;
    scales[block_id] = scale;

    // Quantize: find nearest in LUT
    float inv_scale = 1.0f / scale;
    for (int i = start; i < end; i += 2) {
        uint8_t packed = 0;
        for (int j = 0; j < 2 && (i + j) < end; ++j) {
            float val = input[i + j] * inv_scale;
            int sign = (val < 0.0f) ? 1 : 0;
            float abs_val = fabsf(val);

            // Find nearest LUT value
            int best_idx = 0;
            float best_dist = fabsf(abs_val - mxfp4_lut[0]);
            for (int l = 1; l < 8; ++l) {
                float dist = fabsf(abs_val - mxfp4_lut[l]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = l;
                }
            }
            uint8_t q = (sign << 3) | (best_idx & 0x7);
            if (j == 0)
                packed |= (q & 0xF);
            else
                packed |= ((q & 0xF) << 4);
        }
        output[(i - start) / 2 + (start / 2)] = packed;
    }
}

// ---- MXFP4 Dequantization ----
__global__ void mxfp4_dequantize_kernel(const uint8_t* input, const float* scales,
                                         float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int block_id = idx / MXFP4_BLOCK_SIZE;
    float scale = scales[block_id];

    int byte_idx = idx / 2;
    uint8_t packed = input[byte_idx];
    uint8_t q;
    if (idx % 2 == 0)
        q = packed & 0xF;
    else
        q = (packed >> 4) & 0xF;

    int sign = (q >> 3) & 1;
    int mag_idx = q & 0x7;
    float val = mxfp4_lut[mag_idx] * scale;
    output[idx] = sign ? -val : val;
}

// ---- MXFP4 GEMM ----
__global__ void mxfp4_gemm_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[GEMM_TILE][GEMM_TILE];
    __shared__ float Bs[GEMM_TILE][GEMM_TILE];

    int row = blockIdx.y * GEMM_TILE + threadIdx.y;
    int col = blockIdx.x * GEMM_TILE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (K + GEMM_TILE - 1) / GEMM_TILE; ++t) {
        int a_col = t * GEMM_TILE + threadIdx.x;
        int b_row = t * GEMM_TILE + threadIdx.y;

        float a_val = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        float b_val = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Simulate FP4 quantization: round to nearest LUT value
        auto quantize_fp4 = [](float x) -> float {
            float sign = (x < 0.0f) ? -1.0f : 1.0f;
            float abs_x = fabsf(x);
            // Snap to nearest: 0, 0.5, 1, 1.5, 2, 3, 4, 6
            if (abs_x <= 0.25f) return 0.0f;
            if (abs_x <= 0.75f) return sign * 0.5f;
            if (abs_x <= 1.25f) return sign * 1.0f;
            if (abs_x <= 1.75f) return sign * 1.5f;
            if (abs_x <= 2.5f) return sign * 2.0f;
            if (abs_x <= 3.5f) return sign * 3.0f;
            if (abs_x <= 5.0f) return sign * 4.0f;
            return sign * 6.0f;
        };

        As[threadIdx.y][threadIdx.x] = quantize_fp4(a_val);
        Bs[threadIdx.y][threadIdx.x] = quantize_fp4(b_val);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==================== NVFP4 ====================
// NVIDIA FP4: E2M1 format with block scaling (block_size=32, scale in E4M3)
// Same representable magnitudes as MXFP4 but different scale format

#define NVFP4_BLOCK_SIZE 32
#define NVFP4_MAX_VAL 6.0f

// ---- NVFP4 Quantization ----
__global__ void nvfp4_quantize_kernel(const float* input, uint8_t* output,
                                       float* scales, int N) {
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = CEIL(N, NVFP4_BLOCK_SIZE);
    if (block_id >= num_blocks) return;

    int start = block_id * NVFP4_BLOCK_SIZE;
    int end = min(start + NVFP4_BLOCK_SIZE, N);

    // Find amax
    float amax = 0.0f;
    for (int i = start; i < end; ++i) {
        amax = fmaxf(amax, fabsf(input[i]));
    }

    // Scale in E4M3 range (max 448)
    float scale = amax / NVFP4_MAX_VAL;
    if (scale == 0.0f) scale = 1.0f;
    scales[block_id] = scale;

    // Quantize to 4-bit E2M1
    float inv_scale = 1.0f / scale;
    for (int i = start; i < end; i += 2) {
        uint8_t packed = 0;
        for (int j = 0; j < 2 && (i + j) < end; ++j) {
            float val = input[i + j] * inv_scale;
            int sign = (val < 0.0f) ? 1 : 0;
            float abs_val = fabsf(val);

            // Map to nearest E2M1 value
            int best_idx = 0;
            float best_dist = abs_val;  // distance to 0
            float lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
            for (int l = 1; l < 8; ++l) {
                float dist = fabsf(abs_val - lut_vals[l]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = l;
                }
            }
            uint8_t q = (sign << 3) | (best_idx & 0x7);
            if (j == 0)
                packed |= (q & 0xF);
            else
                packed |= ((q & 0xF) << 4);
        }
        output[(i - start) / 2 + (start / 2)] = packed;
    }
}

// ---- NVFP4 Dequantization ----
__global__ void nvfp4_dequantize_kernel(const uint8_t* input, const float* scales,
                                         float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int block_id = idx / NVFP4_BLOCK_SIZE;
    float scale = scales[block_id];

    int byte_idx = idx / 2;
    uint8_t packed = input[byte_idx];
    uint8_t q;
    if (idx % 2 == 0)
        q = packed & 0xF;
    else
        q = (packed >> 4) & 0xF;

    int sign = (q >> 3) & 1;
    int mag_idx = q & 0x7;
    float lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float val = lut_vals[mag_idx] * scale;
    output[idx] = sign ? -val : val;
}

// ---- NVFP4 GEMV ----
// matrix: [M, N] (quantized as NVFP4), vector: [N], output: [M]
// Each thread computes one row's dot product
__global__ void nvfp4_gemv_kernel(const uint8_t* matrix_q, const float* mat_scales,
                                   const float* vector, float* output,
                                   int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    float lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    int blocks_per_row = CEIL(N, NVFP4_BLOCK_SIZE);

    for (int col = 0; col < N; ++col) {
        // Dequantize matrix element
        int global_idx = row * N + col;
        int block_id = row * blocks_per_row + col / NVFP4_BLOCK_SIZE;
        float scale = mat_scales[block_id];

        int byte_idx = global_idx / 2;
        uint8_t packed = matrix_q[byte_idx];
        uint8_t q = (global_idx % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);

        int sign = (q >> 3) & 1;
        int mag_idx = q & 0x7;
        float mat_val = lut_vals[mag_idx] * scale;
        if (sign) mat_val = -mat_val;

        sum += mat_val * vector[col];
    }
    output[row] = sum;
}

// ---- NVFP4 GEMM ----
__global__ void nvfp4_gemm_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[GEMM_TILE][GEMM_TILE];
    __shared__ float Bs[GEMM_TILE][GEMM_TILE];

    int row = blockIdx.y * GEMM_TILE + threadIdx.y;
    int col = blockIdx.x * GEMM_TILE + threadIdx.x;

    float sum = 0.0f;
    float lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

    for (int t = 0; t < (K + GEMM_TILE - 1) / GEMM_TILE; ++t) {
        int a_col = t * GEMM_TILE + threadIdx.x;
        int b_row = t * GEMM_TILE + threadIdx.y;

        float a_val = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        float b_val = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Simulate NVFP4 quantization
        auto quantize_nvfp4 = [&lut_vals](float x) -> float {
            float sign = (x < 0.0f) ? -1.0f : 1.0f;
            float abs_x = fabsf(x);
            int best_idx = 0;
            float best_dist = abs_x;
            for (int l = 1; l < 8; ++l) {
                float dist = fabsf(abs_x - lut_vals[l]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = l;
                }
            }
            return sign * lut_vals[best_idx];
        };

        As[threadIdx.y][threadIdx.x] = quantize_nvfp4(a_val);
        Bs[threadIdx.y][threadIdx.x] = quantize_nvfp4(b_val);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < GEMM_TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// ==================== Torch Bindings ====================

// MXFP8
torch::Tensor torch_mxfp8_quantize(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    int num_blocks = CEIL(N, MXFP8_BLOCK_SIZE);
    auto output = torch::empty({N}, torch::dtype(torch::kInt8).device(input.device()));
    auto scales = torch::empty({num_blocks}, input.options());
    int block = 256;
    int grid = CEIL(num_blocks, block);
    mxfp8_quantize_kernel<<<grid, block>>>(input.data_ptr<float>(),
        output.data_ptr<int8_t>(), scales.data_ptr<float>(), N);
    return scales;  // Return scales; quantized data in output
}

torch::Tensor torch_mxfp8_dequantize(torch::Tensor input, torch::Tensor scales) {
    int N = input.numel();
    auto output = torch::empty({N}, scales.options());
    int block = 256;
    int grid = CEIL(N, block);
    mxfp8_dequantize_kernel<<<grid, block>>>(input.data_ptr<int8_t>(),
        scales.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

torch::Tensor torch_mxfp8_gemm(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(GEMM_TILE, GEMM_TILE);
    dim3 grid(CEIL(N, GEMM_TILE), CEIL(M, GEMM_TILE));
    mxfp8_gemm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                         C.data_ptr<float>(), M, N, K);
    return C;
}

// MXFP4
torch::Tensor torch_mxfp4_quantize(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    int num_blocks = CEIL(N, MXFP4_BLOCK_SIZE);
    auto output = torch::empty({CEIL(N, 2)}, torch::dtype(torch::kUInt8).device(input.device()));
    auto scales = torch::empty({num_blocks}, input.options());
    int block = 256;
    int grid = CEIL(num_blocks, block);
    mxfp4_quantize_kernel<<<grid, block>>>(input.data_ptr<float>(),
        output.data_ptr<uint8_t>(), scales.data_ptr<float>(), N);
    return scales;
}

torch::Tensor torch_mxfp4_dequantize(torch::Tensor input, torch::Tensor scales, int N) {
    auto output = torch::empty({N}, scales.options());
    int block = 256;
    int grid = CEIL(N, block);
    mxfp4_dequantize_kernel<<<grid, block>>>(input.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

torch::Tensor torch_mxfp4_gemm(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(GEMM_TILE, GEMM_TILE);
    dim3 grid(CEIL(N, GEMM_TILE), CEIL(M, GEMM_TILE));
    mxfp4_gemm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                         C.data_ptr<float>(), M, N, K);
    return C;
}

// NVFP4
torch::Tensor torch_nvfp4_quantize(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    int num_blocks = CEIL(N, NVFP4_BLOCK_SIZE);
    auto output = torch::empty({CEIL(N, 2)}, torch::dtype(torch::kUInt8).device(input.device()));
    auto scales = torch::empty({num_blocks}, input.options());
    int block = 256;
    int grid = CEIL(num_blocks, block);
    nvfp4_quantize_kernel<<<grid, block>>>(input.data_ptr<float>(),
        output.data_ptr<uint8_t>(), scales.data_ptr<float>(), N);
    return scales;
}

torch::Tensor torch_nvfp4_dequantize(torch::Tensor input, torch::Tensor scales, int N) {
    auto output = torch::empty({N}, scales.options());
    int block = 256;
    int grid = CEIL(N, block);
    nvfp4_dequantize_kernel<<<grid, block>>>(input.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

torch::Tensor torch_nvfp4_gemv(torch::Tensor matrix_q, torch::Tensor mat_scales,
                                torch::Tensor vector, int M, int N) {
    auto output = torch::empty({M}, vector.options());
    int block = 256;
    int grid = CEIL(M, block);
    nvfp4_gemv_kernel<<<grid, block>>>(matrix_q.data_ptr<uint8_t>(),
        mat_scales.data_ptr<float>(), vector.data_ptr<float>(),
        output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_nvfp4_gemm(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(GEMM_TILE, GEMM_TILE);
    dim3 grid(CEIL(N, GEMM_TILE), CEIL(M, GEMM_TILE));
    nvfp4_gemm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                         C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // MXFP8
    TORCH_BINDING_COMMON_EXTENSION(torch_mxfp8_quantize)
    TORCH_BINDING_COMMON_EXTENSION(torch_mxfp8_dequantize)
    TORCH_BINDING_COMMON_EXTENSION(torch_mxfp8_gemm)
    // MXFP4
    TORCH_BINDING_COMMON_EXTENSION(torch_mxfp4_quantize)
    TORCH_BINDING_COMMON_EXTENSION(torch_mxfp4_dequantize)
    TORCH_BINDING_COMMON_EXTENSION(torch_mxfp4_gemm)
    // NVFP4
    TORCH_BINDING_COMMON_EXTENSION(torch_nvfp4_quantize)
    TORCH_BINDING_COMMON_EXTENSION(torch_nvfp4_dequantize)
    TORCH_BINDING_COMMON_EXTENSION(torch_nvfp4_gemv)
    TORCH_BINDING_COMMON_EXTENSION(torch_nvfp4_gemm)
}
