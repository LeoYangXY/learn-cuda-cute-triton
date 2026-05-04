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

// ==================== Fused GEMM Kernels ====================
// GEMM + Bias + ReLU
// GEMM + Swish
// GEMM + Sigmoid + Sum
// GEMM + Element-wise Multiply + LeakyReLU
//
// 【任务划分逻辑】
// 所有 fused kernel 基于相同的 tiled GEMM 框架 + 不同的后处理融合
//
// ▸ Tiled GEMM 框架:
//   - block = (TILE_N=32, TILE_M=32), grid = (ceil(N/32), ceil(M/32))
//   - 每个 thread 负责 C 矩阵中 1 个元素 C[row][col]
//   - K 维度按 TILE_K=32 分块: for(t=0; t<ceil(K/32); t++)
//
//   每个 tile 步骤:
//   ① Load A tile: As[ty][tx] = A[row][t*32+tx] (32×32 子矩阵)
//   ② Load B tile: Bs[ty][tx] = B[(t*32+ty)][col] (32×32 子矩阵)
//   ③ __syncthreads()
//   ④ Compute: sum += Σ As[ty][k] * Bs[k][tx], k∈[0,32)
//   ⑤ __syncthreads()
//
//   关键优化:
//   - Shared memory 使 A/B 的每个元素被 32 个 thread 复用 (减少 32x 全局访问)
//   - #pragma unroll 在 K 内循环展开
//   - 边界检查: 越界元素填 0
//
// ▸ 融合后处理 (在 GEMM 结果写入 global memory 前计算):
//   - Bias+ReLU: sum += bias[col]; output = max(0, sum)
//   - Swish: output = sum * sigmoid(sum)
//   - Sigmoid+Sum: output = sigmoid(sum); atomicAdd to global
//   - Mul+LeakyReLU: val = sum * D[row][col]; output = val>0 ? val : alpha*val
//
// 融合的收益:
//   - 避免 GEMM 结果写回 → 重新读取做后处理的两次全局访存
//   - 后处理在 register 中完成，零额外带宽开销

// ---- Tiled GEMM helper (shared memory) ----
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

// ---- GEMM + Bias + ReLU ----
// C = ReLU(A @ B + bias)
// A: [M, K], B: [K, N], bias: [N], C: [M, N]
__global__ void gemm_bias_relu_kernel(const float* A, const float* B, const float* bias,
                                       float* C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        // Load A tile
        int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile
        int b_row = t * TILE_K + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        // Add bias and apply ReLU
        sum += bias[col];
        C[row * N + col] = fmaxf(0.0f, sum);
    }
}

// ---- GEMM + Swish: C = (A @ B) * sigmoid(A @ B) ----
__global__ void gemm_swish_kernel(const float* A, const float* B,
                                   float* C, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE_K + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        C[row * N + col] = sum * sigmoid_val;
    }
}

// ---- GEMM + Sigmoid + Sum: sum(sigmoid(A @ B)) ----
// 先计算 GEMM，然后对结果矩阵做 sigmoid 再求和
__global__ void gemm_sigmoid_sum_kernel(const float* A, const float* B,
                                         float* partial_sums, int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE_K + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float sigmoid_val = 1.0f / (1.0f + expf(-sum));
        atomicAdd(partial_sums, sigmoid_val);
    }
}

// ---- GEMM + Element-wise Multiply + LeakyReLU ----
// C = LeakyReLU((A @ B) * D)
// A: [M,K], B: [K,N], D: [M,N], C: [M,N]
__global__ void gemm_mul_leaky_relu_kernel(const float* A, const float* B,
                                            const float* D, float* C,
                                            int M, int N, int K, float alpha) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        int a_col = t * TILE_K + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int b_row = t * TILE_K + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float val = sum * D[row * N + col];
        C[row * N + col] = val > 0.0f ? val : alpha * val;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_gemm_bias_relu(torch::Tensor A, torch::Tensor B, torch::Tensor bias) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE_N, TILE_M);
    dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));
    gemm_bias_relu_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                            bias.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor torch_gemm_swish(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE_N, TILE_M);
    dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));
    gemm_swish_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                        C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor torch_gemm_sigmoid_sum(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto result = torch::zeros({1}, A.options());
    dim3 block(TILE_N, TILE_M);
    dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));
    gemm_sigmoid_sum_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                              result.data_ptr<float>(), M, N, K);
    return result;
}

torch::Tensor torch_gemm_mul_leaky_relu(torch::Tensor A, torch::Tensor B,
                                         torch::Tensor D, float alpha) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE_N, TILE_M);
    dim3 grid(CEIL(N, TILE_N), CEIL(M, TILE_M));
    gemm_mul_leaky_relu_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                                 D.data_ptr<float>(), C.data_ptr<float>(),
                                                 M, N, K, alpha);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_gemm_bias_relu)
    TORCH_BINDING_COMMON_EXTENSION(torch_gemm_swish)
    TORCH_BINDING_COMMON_EXTENSION(torch_gemm_sigmoid_sum)
    TORCH_BINDING_COMMON_EXTENSION(torch_gemm_mul_leaky_relu)
}
