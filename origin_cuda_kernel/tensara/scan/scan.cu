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

// ==================== Scan Kernels ====================
// Cumulative Sum (Prefix Sum) and Cumulative Product
//
// 【任务划分逻辑】
// Prefix scan 有数据依赖 (output[i] 依赖 output[i-1])，不能像 elementwise 那样简单并行
//
// ▸ Naive 版本: 一行一个 thread (完全串行)
//   - grid = M, 每行 1 个 thread 按顺序做 scan
//   - 适合 N 很小或作为 baseline 对比
//
// ▸ Parallel Hillis-Steele Scan (N <= 1024):
//   - 一行一个 block, block_size = next_power_of_2(N)
//   - 使用双缓冲 shared memory (2*N floats)
//   - 每轮 stride *= 2:
//     if (tid >= stride): dst[tid] = src[tid] + src[tid - stride]
//     else:               dst[tid] = src[tid]
//   - O(N*log(N)) 工作量但只需 O(log(N)) 轮，每轮全并行
//   - 双缓冲避免 read-write conflict: 每轮 swap src/dst
//   - 适合 N 在几百到一千的场景
//
// ▸ Large N: fallback 到串行 (TODO: Blelloch work-efficient scan)
//
// ▸ Cumulative Product: 同理串行，因为乘法的 scan 不易并行
//
// ▸ Running Sum (滑动窗口求和):
//   - Naive: 每个 thread 负责 1 个 output, 内循环 K 次
//   - Prefix-based: 先算 prefix sum, 再 output[i] = prefix[i+1] - prefix[i-K+1]
//     将 O(N*K) 变为 O(N) (prefix sum 的代价)

// ---- Cumulative Sum (per row) ----
// 使用 Blelloch scan (work-efficient parallel prefix sum) within each block
// 对于每行数据，一个block处理

// Simple sequential scan per row (baseline)
__global__ void cumsum_naive_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        sum += row_in[i];
        row_out[i] = sum;
    }
}

// Parallel prefix sum using shared memory (Hillis-Steele)
// 适用于 N <= blockDim.x 的情况
__global__ void cumsum_parallel_kernel(const float* input, float* output, int M, int N) {
    extern __shared__ float smem[];  // 2 * N

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float* buf0 = smem;
    float* buf1 = smem + N;

    // Load
    buf0[tid] = (tid < N) ? row_in[tid] : 0.0f;
    __syncthreads();

    // Hillis-Steele scan
    float* src = buf0;
    float* dst = buf1;
    for (int stride = 1; stride < N; stride <<= 1) {
        if (tid < N) {
            if (tid >= stride) {
                dst[tid] = src[tid] + src[tid - stride];
            } else {
                dst[tid] = src[tid];
            }
        }
        __syncthreads();
        // swap
        float* tmp = src;
        src = dst;
        dst = tmp;
        __syncthreads();
    }

    if (tid < N) {
        row_out[tid] = src[tid];
    }
}

// Block-level scan for large N: each block handles a chunk, then fixup
// 使用 grid-stride 的方式处理大 N
__global__ void cumsum_large_kernel(const float* input, float* output, int N) {
    // 每个row由一个block按顺序处理
    // block内的threads协作处理chunks
    int row = blockIdx.x;
    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // 简单实现: thread 0 做 sequential scan (对于大N这是baseline)
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += row_in[i];
            row_out[i] = sum;
        }
    }
}

// ---- Cumulative Product (per row) ----
__global__ void cumprod_kernel(const float* input, float* output, int M, int N) {
    int row = blockIdx.x;
    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    if (threadIdx.x == 0) {
        float prod = 1.0f;
        for (int i = 0; i < N; ++i) {
            prod *= row_in[i];
            row_out[i] = prod;
        }
    }
}

// ---- 1D Running Sum ----
// output[i] = sum(input[max(0,i-K+1)..i])
// 即滑动窗口求和
__global__ void running_sum_kernel(const float* input, float* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        int start = max(0, idx - K + 1);
        for (int i = start; i <= idx; ++i) {
            sum += input[i];
        }
        output[idx] = sum;
    }
}

// 使用前缀和加速 running sum
// output[i] = prefix[i+1] - prefix[max(0, i-K+1)]
__global__ void running_sum_prefix_kernel(const float* prefix_sum, float* output, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int start = max(0, idx - K + 1);
        output[idx] = prefix_sum[idx + 1] - prefix_sum[start];
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_cumsum_naive(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    cumsum_naive_kernel<<<M, 1>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_cumsum_parallel(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    if (N <= 1024) {
        // N fits in one block
        int block = N;
        // round up to power of 2
        int p = 1;
        while (p < block) p <<= 1;
        block = p;
        size_t smem = 2 * block * sizeof(float);
        cumsum_parallel_kernel<<<M, block, smem>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    } else {
        // Fallback to large kernel
        cumsum_large_kernel<<<M, 1>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), N);
    }
    return output;
}

torch::Tensor torch_cumprod(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    cumprod_kernel<<<M, 1>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_running_sum(torch::Tensor input, int K) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(N, block);
    running_sum_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, K);
    return output;
}

torch::Tensor torch_running_sum_prefix(torch::Tensor input, int K) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    // Compute prefix sum first using torch
    auto prefix = torch::zeros({N + 1}, input.options());
    // prefix[1:] = cumsum(input)
    prefix.slice(0, 1, N + 1).copy_(input.cumsum(0));
    
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(N, block);
    running_sum_prefix_kernel<<<grid, block>>>(prefix.data_ptr<float>(), output.data_ptr<float>(), N, K);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_cumsum_naive)
    TORCH_BINDING_COMMON_EXTENSION(torch_cumsum_parallel)
    TORCH_BINDING_COMMON_EXTENSION(torch_cumprod)
    TORCH_BINDING_COMMON_EXTENSION(torch_running_sum)
    TORCH_BINDING_COMMON_EXTENSION(torch_running_sum_prefix)
}
