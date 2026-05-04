#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
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

// ==================== Tensor-Matrix Multiplication ====================
// 3D Tensor-Matrix: T[B, M, K] @ W[K, N] -> O[B, M, N]
// 4D Tensor-Matrix: T[B, C, M, K] @ W[K, N] -> O[B, C, M, N]
//
// 【任务划分逻辑】
// 本质是 Batched GEMM: 对每个 batch (或 batch*channel) 独立做一次矩阵乘法
//
// ▸ 3D Tensor-Matrix (Batched GEMM):
//   - grid = (ceil(N/TILE), ceil(M/TILE), B): 第 3 维是 batch 索引
//   - block = (TILE=32, TILE=32): 每个 block 输出 C 矩阵的一个 32×32 tile
//   - 每个 thread 负责 1 个 output 元素 C[row][col]
//   - K 维按 TILE 分块: for(t=0; t<ceil(K/32); t++)
//     ① Load As[ty][tx] = T[batch][row][t*32+tx]
//     ② Load Bs[ty][tx] = W[t*32+ty][col]
//     ③ sync → compute sum += As[ty][k]*Bs[k][tx] → sync
//   - batch 维完全独立，体现在 grid.z 上
//
// ▸ 4D Tensor-Matrix:
//   - 将 (B, C) 展平为 BC = B*C, grid.z = BC
//   - 其余与 3D 完全相同
//   - 每个 (b,c) 对独立，共享同一个 W 矩阵
//
// ▸ Square MatMul (M=N=K):
//   - 和通用 GEMM 相同的 tiled 方案，但可以假设所有维度相等
//
// ▸ Upper/Lower Triangular MatMul:
//   - 每个 thread 计算 C[row][col]
//   - 额外判断: if (col >= row) 才做计算 (upper), 否则写 0
//   - 节省约一半的计算 (三角形外的区域直接写 0)
//   - 当前实现未做 shared memory tiling (使用 naive + 条件判断)

#define TILE 32

// ---- 3D Tensor-Matrix Multiplication ----
// Batched GEMM: for each batch b, O[b] = T[b] @ W
__global__ void tensor3d_matmul_kernel(const float* T, const float* W, float* O,
                                        int B, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int batch = blockIdx.z;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    const float* T_batch = T + batch * M * K;
    float* O_batch = O + batch * M * N;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? T_batch[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? W[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        O_batch[row * N + col] = sum;
    }
}

// ---- 4D Tensor-Matrix Multiplication ----
// For each (b, c): O[b,c] = T[b,c] @ W
__global__ void tensor4d_matmul_kernel(const float* T, const float* W, float* O,
                                        int BC, int M, int K, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int bc = blockIdx.z;  // flattened batch*channel index
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    const float* T_slice = T + bc * M * K;
    float* O_slice = O + bc * M * N;

    float sum = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? T_slice[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? W[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        O_slice[row * N + col] = sum;
    }
}

// ---- Square Matrix Multiplication (optimized for M=N=K) ----
__global__ void square_matmul_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ---- Upper Triangular Matrix Multiplication ----
// C = A @ B where only upper triangular part of C is computed
__global__ void upper_tri_matmul_kernel(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        if (col >= row) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

// ---- Lower Triangular Matrix Multiplication ----
__global__ void lower_tri_matmul_kernel(const float* A, const float* B, float* C,
                                         int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        if (col <= row) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_tensor3d_matmul(torch::Tensor T, torch::Tensor W) {
    CHECK_TORCH_TENSOR_DTYPE(T, torch::kFloat32);
    int B = T.size(0), M = T.size(1), K = T.size(2);
    int N = W.size(1);
    auto O = torch::empty({B, M, N}, T.options());
    dim3 block(TILE, TILE);
    dim3 grid(CEIL(N, TILE), CEIL(M, TILE), B);
    tensor3d_matmul_kernel<<<grid, block>>>(T.data_ptr<float>(), W.data_ptr<float>(),
                                             O.data_ptr<float>(), B, M, K, N);
    return O;
}

torch::Tensor torch_tensor4d_matmul(torch::Tensor T, torch::Tensor W) {
    CHECK_TORCH_TENSOR_DTYPE(T, torch::kFloat32);
    int B = T.size(0), C = T.size(1), M = T.size(2), K = T.size(3);
    int N = W.size(1);
    auto O = torch::empty({B, C, M, N}, T.options());
    dim3 block(TILE, TILE);
    dim3 grid(CEIL(N, TILE), CEIL(M, TILE), B * C);
    tensor4d_matmul_kernel<<<grid, block>>>(T.data_ptr<float>(), W.data_ptr<float>(),
                                             O.data_ptr<float>(), B * C, M, K, N);
    return O;
}

torch::Tensor torch_square_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());
    dim3 block(TILE, TILE);
    dim3 grid(CEIL(N, TILE), CEIL(N, TILE));
    square_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                           C.data_ptr<float>(), N);
    return C;
}

torch::Tensor torch_upper_tri_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid(CEIL(N, 16), CEIL(M, 16));
    upper_tri_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                              C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor torch_lower_tri_matmul(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid(CEIL(N, 16), CEIL(M, 16));
    lower_tri_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(),
                                              C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_tensor3d_matmul)
    TORCH_BINDING_COMMON_EXTENSION(torch_tensor4d_matmul)
    TORCH_BINDING_COMMON_EXTENSION(torch_square_matmul)
    TORCH_BINDING_COMMON_EXTENSION(torch_upper_tri_matmul)
    TORCH_BINDING_COMMON_EXTENSION(torch_lower_tri_matmul)
}
