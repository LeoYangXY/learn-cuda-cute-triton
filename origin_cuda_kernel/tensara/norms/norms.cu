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

// ==================== Normalization Kernels ====================
// RMS Norm, L1 Norm, L2 Norm, Frobenius Norm
//
// 【任务划分逻辑】
// 归一化算子本质是: 先 reduce 求出某种"范数/统计量"，再逐元素 scale。
// 除 Frobenius Norm 外，都是按行操作: input [M, N] → output [M, N]
//
// ▸ Per-row Norm (RMS, L1, L2): 每行一个 block
//   grid = M (行数)，block 内 thread 协作处理一行的 N 个元素
//
//   Pass 1 - Reduce: 计算行统计量 (sum of squares / sum of abs / ...)
//     • 每个 thread: for(i=tid; i<N; i+=blockDim.x) 累积
//     • block_reduce_sum: warp_shuffle → smem[32] → warp_shuffle
//     • thread 0 计算 rsqrt/inv 并存入 __shared__ float s_value
//
//   Pass 2 - Normalize: 逐元素乘以 scale
//     • 每个 thread: for(i=tid; i<N; i+=blockDim.x) output[i] = input[i] * scale * weight[i]
//     • 同一行的 scale 对所有元素相同 → 只需一次 smem 读取
//
// ▸ Frobenius Norm: 对整个矩阵做归一化
//   分两个 kernel:
//   Step 1 - Partial reduce: 多个 block 协作计算 sum of squares
//     • 每个 block 用 grid-stride loop 处理一部分元素
//     • block 内做 warp reduce → 写入 partial_sums[blockIdx.x]
//   Step 2 - Normalize: 在 host 端 sum partial_sums 得到 norm
//     • 第二个 kernel: output[i] = input[i] / norm
//
// 设计选择:
//   - 一个 block 处理一行: 避免跨 block 同步，简化编程
//   - block size = min(1024, N) 且对齐到 32 的倍数: 充分利用 warp

// ---- warp reduce sum ----
template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---- Block reduce sum (using shared memory + warp shuffle) ----
__device__ float block_reduce_sum(float val, float* smem) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// ---- RMS Normalization ----
// rms = sqrt(mean(x^2) + eps)
// output[i] = x[i] / rms * weight[i]
// 每行一个block处理
__global__ void rms_norm_kernel(const float* input, const float* weight,
                                 float* output, int N, float eps) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    // 计算 sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = row_input[i];
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq, smem);
    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();

    float rms_inv = s_rms;
    for (int i = tid; i < N; i += blockDim.x) {
        row_output[i] = row_input[i] * rms_inv * weight[i];
    }
}

// ---- L1 Normalization: output[i] = x[i] / sum(|x|) ----
// 对每行做 L1 normalization
__global__ void l1_norm_kernel(const float* input, float* output, int N) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    float abs_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        abs_sum += fabsf(row_input[i]);
    }

    abs_sum = block_reduce_sum(abs_sum, smem);
    __shared__ float s_norm;
    if (threadIdx.x == 0) {
        s_norm = (abs_sum > 0.0f) ? (1.0f / abs_sum) : 0.0f;
    }
    __syncthreads();

    float norm_inv = s_norm;
    for (int i = tid; i < N; i += blockDim.x) {
        row_output[i] = row_input[i] * norm_inv;
    }
}

// ---- L2 Normalization: output[i] = x[i] / sqrt(sum(x^2)) ----
__global__ void l2_norm_kernel(const float* input, float* output, int N) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = row_input[i];
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq, smem);
    __shared__ float s_norm;
    if (threadIdx.x == 0) {
        s_norm = rsqrtf(sum_sq + 1e-12f);
    }
    __syncthreads();

    float norm_inv = s_norm;
    for (int i = tid; i < N; i += blockDim.x) {
        row_output[i] = row_input[i] * norm_inv;
    }
}

// ---- Frobenius Normalization: output = x / ||x||_F ----
// ||x||_F = sqrt(sum of all x_ij^2)
// 这个是对整个矩阵做归一化，不是按行
__global__ void frobenius_norm_compute_kernel(const float* input, float* partial_sums,
                                              int N) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / 32;
    int lane = tid % 32;

    float sum_sq = 0.0f;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        float val = input[i];
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) smem[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) partial_sums[blockIdx.x] = val;
    }
}

__global__ void frobenius_norm_normalize_kernel(const float* input, float* output,
                                                 float norm_inv, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] * norm_inv;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_rms_norm(torch::Tensor input, torch::Tensor weight, float eps) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    rms_norm_kernel<<<M, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        output.data_ptr<float>(), N, eps);
    return output;
}

torch::Tensor torch_l1_norm(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    l1_norm_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

torch::Tensor torch_l2_norm(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    l2_norm_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

torch::Tensor torch_frobenius_norm(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = min(256, CEIL(N, block));

    // Step 1: compute partial sums
    auto partial = torch::empty({grid}, input.options());
    frobenius_norm_compute_kernel<<<grid, block>>>(
        input.data_ptr<float>(), partial.data_ptr<float>(), N);

    // Step 2: sum partials on CPU and compute norm
    float total_sum = partial.sum().item<float>();
    float norm_inv = 1.0f / sqrtf(total_sum + 1e-12f);

    // Step 3: normalize
    int norm_grid = CEIL(N, block);
    frobenius_norm_normalize_kernel<<<norm_grid, block>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), norm_inv, N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_rms_norm)
    TORCH_BINDING_COMMON_EXTENSION(torch_l1_norm)
    TORCH_BINDING_COMMON_EXTENSION(torch_l2_norm)
    TORCH_BINDING_COMMON_EXTENSION(torch_frobenius_norm)
}
