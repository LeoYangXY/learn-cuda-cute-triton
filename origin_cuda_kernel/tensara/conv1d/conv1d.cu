#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(a) *(float4*)(&(a))

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
  if (((T).options().dtype() != (th_type))) {                \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);    \
  }

// ==================== 1D Convolution Kernels ====================
//
// 【任务划分逻辑】
// 1D卷积: output[i] = sum_{k=0}^{K-1} input[i+k] * kernel[k]
// 输出长度 = N - K + 1
//
// ▸ Naive 版本:
//   - 每个 thread 负责计算 1 个 output 元素
//   - grid = ceil(out_len / blockDim.x)
//   - thread 计算: output[idx] = Σ input[idx+k] * kernel[k], k∈[0,K)
//   - 问题: kernel 数据被每个 thread 重复从 global memory 读取 → 带宽浪费
//
// ▸ Shared Memory 版本:
//   - 每个 block 产出 blockDim.x 个 output 元素
//   - 需要加载 blockDim.x + K - 1 个 input 元素到 shared memory (含 halo 区域)
//     • 前 blockDim.x 个元素: 每个 thread 加载对应的 1 个
//     • 后 K-1 个 halo 元素: 由前 K-1 个 thread 额外加载
//   - 计算时所有 input 访问走 smem (~100x lower latency than global)
//   - __syncthreads() 确保数据就绪后才开始计算
//
// ▸ Constant Memory 版本:
//   - 卷积核存入 __constant__ memory (硬件广播，对 warp 内读同地址免费)
//   - 适合 kernel 较小的情况 (K <= 1024)
//   - 结合 shared memory 缓存 input tile
//
// ▸ Vec4 版本:
//   - 每个 thread 计算 4 个连续的 output 元素 (减少 thread 总数和调度开销)
//   - 对 kernel weights 的访问被复用 4 次 (register-level reuse)
//   - grid = ceil(out_len / 4 / blockDim.x)

// Naive: 每个thread负责output中的一个元素
__global__ void conv1d_naive_kernel(const float* input, const float* kernel_data,
                                     float* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_len = N - K + 1;
    if (idx < out_len) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += input[idx + k] * kernel_data[k];
        }
        output[idx] = sum;
    }
}

// 使用shared memory: 将input的tile加载到shared memory
// 每个block处理 BLOCK_SIZE 个output元素
// 需要加载 BLOCK_SIZE + K - 1 个input元素到shared memory
__global__ void conv1d_shared_kernel(const float* input, const float* kernel_data,
                                      float* output, int N, int K) {
    extern __shared__ float smem[]; // size = blockDim.x + K - 1

    int tid = threadIdx.x;
    int out_idx = blockIdx.x * blockDim.x + tid;
    int out_len = N - K + 1;

    // 每个thread加载自己对应的input元素
    int input_idx = blockIdx.x * blockDim.x + tid;
    if (input_idx < N) {
        smem[tid] = input[input_idx];
    } else {
        smem[tid] = 0.0f;
    }

    // 额外的线程加载halo部分 (K-1个额外元素)
    if (tid < K - 1) {
        int halo_idx = blockIdx.x * blockDim.x + blockDim.x + tid;
        if (halo_idx < N) {
            smem[blockDim.x + tid] = input[halo_idx];
        } else {
            smem[blockDim.x + tid] = 0.0f;
        }
    }
    __syncthreads();

    if (out_idx < out_len) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += smem[tid + k] * kernel_data[k];
        }
        output[out_idx] = sum;
    }
}

// 使用constant memory存储kernel（适用于小kernel）+ shared memory
__constant__ float d_kernel[1024]; // 支持最大kernel size为1024

__global__ void conv1d_const_kernel(const float* input, float* output, int N, int K) {
    extern __shared__ float smem[];

    int tid = threadIdx.x;
    int out_idx = blockIdx.x * blockDim.x + tid;
    int out_len = N - K + 1;

    int input_idx = blockIdx.x * blockDim.x + tid;
    if (input_idx < N) {
        smem[tid] = input[input_idx];
    } else {
        smem[tid] = 0.0f;
    }

    if (tid < K - 1) {
        int halo_idx = blockIdx.x * blockDim.x + blockDim.x + tid;
        if (halo_idx < N) {
            smem[blockDim.x + tid] = input[halo_idx];
        } else {
            smem[blockDim.x + tid] = 0.0f;
        }
    }
    __syncthreads();

    if (out_idx < out_len) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += smem[tid + k] * d_kernel[k];
        }
        output[out_idx] = sum;
    }
}

// float4向量化版本: 每个thread处理4个output元素
__global__ void conv1d_vec4_kernel(const float* input, const float* kernel_data,
                                    float* output, int N, int K) {
    int out_len = N - K + 1;
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (base_idx + 3 < out_len) {
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        for (int k = 0; k < K; ++k) {
            float w = kernel_data[k];
            sum0 += input[base_idx + k] * w;
            sum1 += input[base_idx + 1 + k] * w;
            sum2 += input[base_idx + 2 + k] * w;
            sum3 += input[base_idx + 3 + k] * w;
        }
        output[base_idx] = sum0;
        output[base_idx + 1] = sum1;
        output[base_idx + 2] = sum2;
        output[base_idx + 3] = sum3;
    } else {
        // 处理尾部
        for (int i = 0; i < 4; ++i) {
            int idx = base_idx + i;
            if (idx < out_len) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += input[idx + k] * kernel_data[k];
                }
                output[idx] = sum;
            }
        }
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_conv1d_naive(torch::Tensor input, torch::Tensor kernel_t) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(kernel_t, torch::kFloat32);
    int N = input.numel();
    int K = kernel_t.numel();
    int out_len = N - K + 1;
    auto output = torch::empty({out_len}, input.options());
    int block = 256;
    int grid = CEIL(out_len, block);
    conv1d_naive_kernel<<<grid, block>>>(
        input.data_ptr<float>(), kernel_t.data_ptr<float>(),
        output.data_ptr<float>(), N, K);
    return output;
}

torch::Tensor torch_conv1d_shared(torch::Tensor input, torch::Tensor kernel_t) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(kernel_t, torch::kFloat32);
    int N = input.numel();
    int K = kernel_t.numel();
    int out_len = N - K + 1;
    auto output = torch::empty({out_len}, input.options());
    int block = 256;
    int grid = CEIL(out_len, block);
    size_t smem_size = (block + K - 1) * sizeof(float);
    conv1d_shared_kernel<<<grid, block, smem_size>>>(
        input.data_ptr<float>(), kernel_t.data_ptr<float>(),
        output.data_ptr<float>(), N, K);
    return output;
}

torch::Tensor torch_conv1d_const(torch::Tensor input, torch::Tensor kernel_t) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(kernel_t, torch::kFloat32);
    int N = input.numel();
    int K = kernel_t.numel();
    int out_len = N - K + 1;
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, kernel_t.data_ptr<float>(), K * sizeof(float));
    auto output = torch::empty({out_len}, input.options());
    int block = 256;
    int grid = CEIL(out_len, block);
    size_t smem_size = (block + K - 1) * sizeof(float);
    conv1d_const_kernel<<<grid, block, smem_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N, K);
    return output;
}

torch::Tensor torch_conv1d_vec4(torch::Tensor input, torch::Tensor kernel_t) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(kernel_t, torch::kFloat32);
    int N = input.numel();
    int K = kernel_t.numel();
    int out_len = N - K + 1;
    auto output = torch::empty({out_len}, input.options());
    int block = 256;
    int grid = CEIL(CEIL(out_len, 4), block);
    conv1d_vec4_kernel<<<grid, block>>>(
        input.data_ptr<float>(), kernel_t.data_ptr<float>(),
        output.data_ptr<float>(), N, K);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_conv1d_naive)
    TORCH_BINDING_COMMON_EXTENSION(torch_conv1d_shared)
    TORCH_BINDING_COMMON_EXTENSION(torch_conv1d_const)
    TORCH_BINDING_COMMON_EXTENSION(torch_conv1d_vec4)
}
