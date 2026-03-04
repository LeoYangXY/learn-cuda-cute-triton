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
using namespace cute;

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

// ========== 主函数：测试 CUTE GEMM ==========
int main() {
    // 测试矩阵尺寸
    const int M = 128;
    const int N = 256;
    const int K = 384;

    printf("正在测试 CUTE GEMM: C[%dx%d] = A[%dx%d] @ B[%dx%d]\n", M, N, M, K, K, N);

    // 分配主机内存
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(M * N); // CPU 参考结果
    std::vector<float> h_C_gpu(M * N); // GPU 计算结果

    // 填充随机数据
    fill_random(h_A);
    fill_random(h_B);

    // 用 CPU 计算参考结果
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 将数据从主机拷贝到设备
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // 配置 Kernel 启动参数
    constexpr int BM = 32, BN = 32, BK = 32, TM = 4, TN = 4;
    constexpr int THREADS = (BM / TM) * (BN / TN); // 64
    dim3 block(THREADS);           // 一维 block，threadIdx.x 范围 0~63
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM); // 向上取整

    printf("启动配置: Grid=(%d, %d), Block=(%d)\n", grid.x, grid.y, block.x);

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 统一口径计时：warmup + 多轮平均，减少一次性抖动
    constexpr int WARMUP = 20;
    constexpr int ITERS = 200;

    for (int i = 0; i < WARMUP; ++i) {
        sgemm_cute_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        sgemm_cute_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&total_ms, start, stop));
    float ms = total_ms / ITERS;
    double tflops = (2.0 * static_cast<double>(M) * N * K) / (ms * 1.0e-3) / 1.0e12;

    // 将结果拷回主机
    checkCudaErrors(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 验证结果
    bool passed = validate(h_C_ref, h_C_gpu, M * N);
    printf("CUTE GEMM 测试: %s (平均 %.4f 毫秒, %.3f TFLOPS)\n", passed ? "通过 ✅" : "失败 ❌", ms, tflops);

    // 释放资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (passed) {
        printf("\n🎉 所有测试通过！\n");
        return 0;
    } else {
        printf("\n💥 测试失败，请检查代码。\n");
        return 1;
    }
}