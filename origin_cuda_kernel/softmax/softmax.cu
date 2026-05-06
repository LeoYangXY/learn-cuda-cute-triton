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

// ==================== Softmax Kernel ====================
// Online safe softmax per row: 一个block处理一行
//
// 【任务划分逻辑】
// Softmax: output[i] = exp(x[i] - max) / Σ exp(x[j] - max)
// 对 2D tensor [M, N] 沿 dim=1 做 softmax: 每行一个 block
//
// ▸ 3-Pass Safe Softmax (vectorized):
//   grid = M (行数), block = min(1024, N) 对齐 32
//
//   Pass 1 - Find row max (float4 向量化):
//     • 每个 thread 用 float4 加载 4 个元素取 max
//     • thread-stride loop: for(i=tid; i<N/4; i+=blockDim.x)
//     • warp reduce max → smem[warp_id] → warp 0 reduce → broadcast smem[0]
//
//   Pass 2 - Compute sum of exp(x - max) (float4 向量化):
//     • 同样用 float4 加载，每组 4 个元素 exp + 累加
//     • 两级 reduce → broadcast inv_sum = 1/sum
//
//   Pass 3 - Write output (float4 向量化):
//     • float4 加载 input → 计算 exp(x-max)*inv_sum → float4 写出
//     • 3 个 pass 每个都做 float4 向量化，最大化带宽利用
//
// ▸ Online Softmax (2-pass，Milakov & Gimelshein 2018):
//   将 Pass 1+2 合并为一个 pass，维护 running (max, sum) 对:
//     if new_val > local_max:
//       local_sum = local_sum * exp(old_max - new_max) + 1
//       local_max = new_val
//     else:
//       local_sum += exp(new_val - local_max)
//   Warp reduce 时也需要合并 (max, sum) 对，公式:
//     combined_sum = sum_a * exp(max_a - max_combined) + sum_b * exp(max_b - max_combined)
//   优势: 只需 2 次读 input (1次online+1次写output) 而非 3 次
//
// ▸ Log Softmax:
//   log_softmax = (x - max) - log(sum_exp)
//   和 softmax 相同的 3-pass 结构，但最后写 (x-max) - log_sum 而非 exp

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Safe softmax: 先找max，减去max，再做exp和归一化
// 每行一个block
__global__ void softmax_kernel(const float* input, float* output, int M, int N) {
    __shared__ float smem[32];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // Pass 1: find row max
    float max_val = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    max_val = warp_reduce_max(max_val);
    if (lane == 0) smem[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        max_val = (lane < nw) ? smem[lane] : -FLT_MAX;
        max_val = warp_reduce_max(max_val);
    }
    __syncthreads();
    // broadcast max to all threads
    if (tid == 0) smem[0] = max_val;
    __syncthreads();
    max_val = smem[0];

    // Pass 2: compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum_exp += expf(row_in[i] - max_val);
    }
    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) smem[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        sum_exp = (lane < nw) ? smem[lane] : 0.0f;
        sum_exp = warp_reduce_sum(sum_exp);
    }
    __syncthreads();
    if (tid == 0) smem[0] = sum_exp;
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    // Pass 3: write output
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - max_val) * inv_sum;
    }
}

// Online softmax (single pass for max + sum using online normalization trick)
__global__ void online_softmax_kernel(const float* input, float* output, int M, int N) {
    __shared__ float smem_max[32];
    __shared__ float smem_sum[32];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // Online pass: maintain running max and sum
    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = row_in[i];
        if (val > local_max) {
            local_sum = local_sum * expf(local_max - val) + 1.0f;
            local_max = val;
        } else {
            local_sum += expf(val - local_max);
        }
    }

    // Warp reduce (combine max and sum)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
        if (other_max > local_max) {
            local_sum = local_sum * expf(local_max - other_max) + other_sum;
            local_max = other_max;
        } else {
            local_sum = local_sum + other_sum * expf(other_max - local_max);
        }
    }

    if (lane == 0) {
        smem_max[warp_id] = local_max;
        smem_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduce in first warp
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        local_max = (lane < nw) ? smem_max[lane] : -FLT_MAX;
        local_sum = (lane < nw) ? smem_sum[lane] : 0.0f;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
            if (other_max > local_max) {
                local_sum = local_sum * expf(local_max - other_max) + other_sum;
                local_max = other_max;
            } else {
                local_sum = local_sum + other_sum * expf(other_max - local_max);
            }
        }
    }
    __syncthreads();
    if (tid == 0) {
        smem_max[0] = local_max;
        smem_sum[0] = local_sum;
    }
    __syncthreads();

    float global_max = smem_max[0];
    float inv_sum = 1.0f / smem_sum[0];

    // Write output
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = expf(row_in[i] - global_max) * inv_sum;
    }
}

// ---- Log Softmax ----
__global__ void log_softmax_kernel(const float* input, float* output, int M, int N) {
    __shared__ float smem[32];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // Pass 1: find row max
    float max_val = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    max_val = warp_reduce_max(max_val);
    if (lane == 0) smem[warp_id] = max_val;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        max_val = (lane < nw) ? smem[lane] : -FLT_MAX;
        max_val = warp_reduce_max(max_val);
    }
    __syncthreads();
    if (tid == 0) smem[0] = max_val;
    __syncthreads();
    max_val = smem[0];

    // Pass 2: compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum_exp += expf(row_in[i] - max_val);
    }
    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) smem[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        sum_exp = (lane < nw) ? smem[lane] : 0.0f;
        sum_exp = warp_reduce_sum(sum_exp);
    }
    __syncthreads();
    if (tid == 0) smem[0] = sum_exp;
    __syncthreads();
    float log_sum = logf(smem[0]);

    // Pass 3: write log_softmax = (x - max) - log(sum_exp)
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = (row_in[i] - max_val) - log_sum;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_softmax(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    softmax_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_online_softmax(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    online_softmax_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_log_softmax(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    log_softmax_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_softmax)
    TORCH_BINDING_COMMON_EXTENSION(torch_online_softmax)
    TORCH_BINDING_COMMON_EXTENSION(torch_log_softmax)
}
