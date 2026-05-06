#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <cuda_fp16.h>
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

// ==================== Fused Add + RMS Norm ====================
// LLM 推理中极其常见的 fused kernel:
// output = RMSNorm(x + residual)
// 即: y = (x + residual) / sqrt(mean((x+residual)^2) + eps) * weight
//
// 【面试考点】
// 1. 为什么要 fuse? 减少 kernel launch + 避免中间结果写回 global memory
// 2. RMS Norm vs Layer Norm? RMS 只算 mean of squares，不减均值
// 3. 两遍 pass vs Welford online? 这里用两遍 (先求 rms，再 normalize)
//
// 【任务划分逻辑】
// - input: [M, N], residual: [M, N], weight: [N]
// - grid = M (每行一个 block)
// - 每个 block 协作处理一行:
//   Pass 1: 每个 thread 用 stride loop 累加 (x+res)^2 → warp reduce → block reduce → 得到 rms
//   Pass 2: 每个 thread normalize 各自负责的元素

#define WARP_SIZE 32

template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_fused(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---- Fused Add + RMS Norm ----
// x: [M, N], residual: [M, N], weight: [N], output: [M, N]
// Also writes back (x + residual) to residual for next layer's use
__global__ void fused_add_rmsnorm_kernel(float* x, float* residual,
                                          const float* weight, float* output,
                                          int N, float eps) {
    __shared__ float smem[32];  // for cross-warp reduce

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    float* x_row = x + row * N;
    float* res_row = residual + row * N;
    float* out_row = output + row * N;

    // Pass 1: Fused add + compute sum of squares
    float sumsq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = x_row[i] + res_row[i];
        res_row[i] = val;  // write back fused result for next layer
        sumsq += val * val;
    }

    // Block reduce sum of squares
    sumsq = warp_reduce_sum_fused(sumsq);
    if (lane_id == 0) smem[warp_id] = sumsq;
    __syncthreads();

    if (warp_id == 0) {
        sumsq = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        sumsq = warp_reduce_sum_fused(sumsq);
    }
    __syncthreads();
    if (tid == 0) smem[0] = sumsq;
    __syncthreads();

    float rms = rsqrtf(smem[0] / (float)N + eps);

    // Pass 2: Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        out_row[i] = res_row[i] * rms * weight[i];
    }
}

// ---- Standalone RMS Norm (for comparison) ----
__global__ void rmsnorm_kernel(const float* input, const float* weight,
                                float* output, int N, float eps) {
    __shared__ float smem[32];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    float sumsq = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = row_in[i];
        sumsq += val * val;
    }

    sumsq = warp_reduce_sum_fused(sumsq);
    if (lane_id == 0) smem[warp_id] = sumsq;
    __syncthreads();

    if (warp_id == 0) {
        sumsq = (lane_id < num_warps) ? smem[lane_id] : 0.0f;
        sumsq = warp_reduce_sum_fused(sumsq);
    }
    __syncthreads();
    if (tid == 0) smem[0] = sumsq;
    __syncthreads();

    float rms = rsqrtf(smem[0] / (float)N + eps);

    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = row_in[i] * rms * weight[i];
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_fused_add_rmsnorm(torch::Tensor x, torch::Tensor residual,
                                       torch::Tensor weight, float eps) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    int M = x.size(0), N = x.size(1);
    auto output = torch::empty_like(x);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    fused_add_rmsnorm_kernel<<<M, block>>>(
        x.data_ptr<float>(), residual.data_ptr<float>(),
        weight.data_ptr<float>(), output.data_ptr<float>(), N, eps);
    return output;
}

torch::Tensor torch_rmsnorm(torch::Tensor input, torch::Tensor weight, float eps) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    rmsnorm_kernel<<<M, block>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(),
        output.data_ptr<float>(), N, eps);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_fused_add_rmsnorm)
    TORCH_BINDING_COMMON_EXTENSION(torch_rmsnorm)
}
