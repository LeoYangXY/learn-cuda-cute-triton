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

// ==================== KL Divergence Loss ====================
// KL(P || Q) = sum(P * log(P / Q)) = sum(P * (log(P) - log(Q)))
// input: log_probs (log Q), target: probs (P)
// reduction: batchmean -> sum(P * (log(P) - log(Q))) / batch_size
// PyTorch convention: input = log Q, target = P
// KL = sum(target * (log(target) - input))
//
// 【任务划分逻辑】
//
// ▸ KL Divergence: 全局 reduce 模式
//   - grid = min(256, ceil(N/block))
//   - 每个 thread 用 grid-stride loop 累积: for(i=global_idx; i<N; i+=stride)
//   - 只对 p>0 的元素计算 (否则 p*log(p) 为 0)
//   - Block 内 warp shuffle reduce → thread 0 atomicAdd 到全局
//
// ▸ Triplet Margin Loss: L = max(0, ||a-p||₂ - ||a-n||₂ + margin)
//   - grid = M (样本数), 每个 block 处理 1 个样本
//   - Block 内 thread 并行计算 D 维的 dist_p 和 dist_n:
//     for(d=tid; d<D; d+=blockDim.x): 累积 (a[d]-p[d])² 和 (a[d]-n[d])²
//   - 两级 reduce (warp + smem) 分别得到 dist_p, dist_n
//   - thread 0: loss = max(0, sqrt(dist_p) - sqrt(dist_n) + margin)

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel(const float* log_probs, const float* target,
                               float* partial_sum, int N) {
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int global_idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        float p = target[i];
        if (p > 0.0f) {
            local_sum += p * (logf(p) - log_probs[i]);
        }
    }

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();

    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        local_sum = (lane < nw) ? smem[lane] : 0.0f;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) atomicAdd(partial_sum, local_sum);
    }
}

// ==================== Triplet Margin Loss ====================
// L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
// d(x, y) = ||x - y||_2
// Per-sample loss, then mean

__global__ void triplet_loss_kernel(const float* anchor, const float* positive,
                                     const float* negative, float* losses,
                                     int M, int D, float margin) {
    __shared__ float smem_dp[32];
    __shared__ float smem_dn[32];
    int sample = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    const float* a = anchor + sample * D;
    const float* p = positive + sample * D;
    const float* n = negative + sample * D;

    float dist_p = 0.0f, dist_n = 0.0f;
    for (int d = tid; d < D; d += blockDim.x) {
        float dp = a[d] - p[d];
        float dn = a[d] - n[d];
        dist_p += dp * dp;
        dist_n += dn * dn;
    }

    dist_p = warp_reduce_sum(dist_p);
    dist_n = warp_reduce_sum(dist_n);
    if (lane == 0) {
        smem_dp[warp_id] = dist_p;
        smem_dn[warp_id] = dist_n;
    }
    __syncthreads();

    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        dist_p = (lane < nw) ? smem_dp[lane] : 0.0f;
        dist_n = (lane < nw) ? smem_dn[lane] : 0.0f;
        dist_p = warp_reduce_sum(dist_p);
        dist_n = warp_reduce_sum(dist_n);
        if (lane == 0) {
            float loss = sqrtf(dist_p) - sqrtf(dist_n) + margin;
            losses[sample] = fmaxf(0.0f, loss);
        }
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_kl_div(torch::Tensor log_probs, torch::Tensor target) {
    CHECK_TORCH_TENSOR_DTYPE(log_probs, torch::kFloat32);
    int N = log_probs.numel();
    auto result = torch::zeros({1}, log_probs.options());
    int block = 256;
    int grid = min(256, CEIL(N, block));
    kl_div_kernel<<<grid, block>>>(log_probs.data_ptr<float>(), target.data_ptr<float>(),
                                    result.data_ptr<float>(), N);
    return result;
}

torch::Tensor torch_triplet_loss(torch::Tensor anchor, torch::Tensor positive,
                                  torch::Tensor negative, float margin) {
    CHECK_TORCH_TENSOR_DTYPE(anchor, torch::kFloat32);
    int M = anchor.size(0), D = anchor.size(1);
    auto losses = torch::empty({M}, anchor.options());
    int block = min(256, D);
    block = ((block + 31) / 32) * 32;
    triplet_loss_kernel<<<M, block>>>(anchor.data_ptr<float>(), positive.data_ptr<float>(),
                                       negative.data_ptr<float>(), losses.data_ptr<float>(),
                                       M, D, margin);
    return losses;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_kl_div)
    TORCH_BINDING_COMMON_EXTENSION(torch_triplet_loss)
}
