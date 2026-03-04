// gemm_cute_zh.cpp
// 使用 CUTE 编写的 FP32 GEMM Kernel（带中文注释）
// 编译命令（以 sm_89 为例，适用于 RTX 4090）：
// nvcc -std=c++17 -arch=sm_89 -O3 -I${CUDA_HOME}/include gemm_cute_zh.cpp -o gemm_cute

#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <cmath>

// 引入 CUTE 库（CUDA 12.0+ 自带）
#include <cute/tensor.hpp>

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// ========== CUDA 错误检查宏 ==========
#define checkCudaErrors(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA 错误: %s，位置: %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ========== CPU 参考 GEMM 实现（用于验证正确性）==========
void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ========== 验证 GPU 结果是否与 CPU 参考一致 ==========
bool validate(const std::vector<float>& ref, const std::vector<float>& gpu, int size, 
              float atol = 1e-4, float rtol = 1e-5) {
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        // 使用相对误差 + 绝对误差判断
        if (diff > atol + rtol * fabsf(ref[i])) {
            printf("结果不匹配！位置 [%d]: CPU=%f, GPU=%f, 误差=%f\n", i, ref[i], gpu[i], diff);
            return false;
        }
    }
    return true;
}

// ========== 用随机数填充向量 ==========
void fill_random(std::vector<float>& v) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : v) x = dis(gen);
}

// ========== 使用 CUTE 思想编写的 GEMM Kernel（FP32）==========
template <
    int BM = 32, // 每个 Block 处理的 C 矩阵行数
    int BN = 32, // 每个 Block 处理的 C 矩阵列数
    int BK = 32, // 沿 K 维度的分块大小
    int TM = 4,  // 每个线程负责计算的 C 子块行数
    int TN = 4   // 每个线程负责计算的 C 子块列数
>
__global__ void sgemm_cute_kernel(float* A, float* B, float* C, int M, int N, int K) {
    using namespace cute;
    constexpr int THREADS_PER_BLOCK = (BM / TM) * (BN / TN);
    static_assert(THREADS_PER_BLOCK == 64, "期望每个 Block 有 64 个线程");

    Tensor gA = make_tensor(make_gmem_ptr(A),
                            make_layout(make_shape(M, K), make_stride(K, 1)));
    Tensor gB = make_tensor(make_gmem_ptr(B),
                            make_layout(make_shape(K, N), make_stride(N, 1)));
    Tensor gC = make_tensor(make_gmem_ptr(C),
                            make_layout(make_shape(M, N), make_stride(N, 1)));

    __shared__ float smemA[BM][BK + 4];
    __shared__ float smemB[BK][BN + 4];

    Tensor sA = make_tensor(make_smem_ptr(&smemA[0][0]),
                            make_layout(make_shape(Int<BM>{}, Int<BK>{}), make_stride(Int<BK + 4>{}, Int<1>{})));
    Tensor sB = make_tensor(make_smem_ptr(&smemB[0][0]),
                            make_layout(make_shape(Int<BK>{}, Int<BN>{}), make_stride(Int<BN + 4>{}, Int<1>{})));

    int tid = threadIdx.x;
    int thr_row_idx = tid / (BN / TN);
    int thr_col_idx = tid % (BN / TN);

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int thread_compute_row = thr_row_idx * TM;
    int thread_compute_col = thr_col_idx * TN;

    int thread_load_row = thr_row_idx * TM;
    int thread_load_col = thr_col_idx * TN;

    float c_reg[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
        // 无边界检查版本（要求 M, N, K 都是 BM/BN/BK 的倍数）
        for (int i = 0; i < TM; ++i) {
            int s_row = thread_load_row + i;
            FLOAT4(sA(s_row, thread_load_col)) = FLOAT4(gA(block_row + s_row, k0 + thread_load_col));
            FLOAT4(sB(s_row, thread_load_col)) = FLOAT4(gB(k0 + s_row, block_col + thread_load_col));
        }

        __syncthreads();

        for (int kk = 0; kk < BK; ++kk) {
            for (int i = 0; i < TM; ++i) {
                float a_val = sA(thread_compute_row + i, kk);
                for (int j = 0; j < TN; ++j) {
                    float b_val = sB(kk, thread_compute_col + j);
                    c_reg[i][j] += a_val * b_val;
                }
            }
        }

        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        int c_row = block_row + thread_compute_row + i;
        if (c_row >= M) continue;
        for (int j = 0; j < TN; ++j) {
            int c_col = block_col + thread_compute_col + j;
            if (c_col < N) {
                gC(c_row, c_col) = c_reg[i][j];
            }
        }
    }
}


// ==================== Torch bindings ====================
#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
  if (((T).options().dtype() != (th_type))) {                \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);    \
  }

#define CEIL(a,b) ((a+b-1)/(b))

torch::Tensor torch_sgemm_cute(torch::Tensor A, torch::Tensor B) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    constexpr int BM = 32, BN = 32, BK = 32, TM = 4, TN = 4;
    constexpr int THREADS = (BM / TM) * (BN / TN);
    dim3 block(THREADS);
    dim3 grid(CEIL(N, BN), CEIL(M, BM));
    sgemm_cute_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_sgemm_cute)
}
