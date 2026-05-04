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

// ==================== Scaled Dot-Product Attention ====================
// Q: [B*H, S, D], K: [B*H, S, D], V: [B*H, S, D]
// Attn = softmax(Q @ K^T / sqrt(D)) @ V
// Output: [B*H, S, D]
//
// 【任务划分逻辑】
// Attention 的核心: 对每个 (batch_head, query_row) 计算一行 output
//
// ▸ 任务分配:
//   - grid = (B*H, S): 第一维是 batch*head 索引，第二维是 query 行索引
//   - block = min(256, D) 对齐 32: block 内 thread 协作处理 D 维度
//   - 每个 block 输出一个向量: output[bh][q_row][0..D-1]
//
// ▸ 算法步骤 (每个 block 内):
//
//   Step 1 - 计算 attention scores: scores[j] = dot(Q[row], K[j]) / sqrt(D)
//     • 对每个 j ∈ [0, S):
//       - block 内 thread 并行计算部分 dot product (每个 thread 负责 D/blockDim.x 个维度)
//       - warp reduce sum + smem 跨 warp reduce → 得到完整 dot product
//       - thread 0 将 score 乘以 scale 写入 smem_scores[j]
//     • smem 存储所有 S 个 scores → 动态 shared memory 大小 = S * sizeof(float)
//
//   Step 2 - Softmax over scores:
//     • thread 0 串行计算: find max → exp(x-max) → sum → normalize
//     • (S 通常较小，串行即可；若 S 很大可用并行 reduce)
//
//   Step 3 - 加权求和: output[d] = Σ scores[j] * V[j][d]
//     • 每个 thread 负责若干个 d 维度: for(d=tid; d<D; d+=blockDim.x)
//     • 对每个 d, 遍历所有 j 累积
//
// 性能瓶颈:
//   - Step 1 需要 S 次 reduce (每次遍历 D 维)，复杂度 O(S*D)
//   - 整体复杂度 O(S²*D) per query row
//   - 这是 naive 实现; Flash Attention 通过分块和 online softmax 将复杂度降为 O(S*D) IO

// warp reduce
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

// Naive attention: each thread block computes one row of output for one head
// Grid: [B*H, S], Block: [D]
// 对于每个 (batch_head, query_row):
//   1. 计算 scores[j] = sum_d(Q[row][d] * K[j][d]) / sqrt(D) for j in [0, S)
//   2. softmax(scores)
//   3. output[d] = sum_j(scores[j] * V[j][d]) for d in [0, D)
__global__ void sdpa_naive_kernel(const float* Q, const float* K, const float* V,
                                   float* output, int S, int D, float scale) {
    int bh = blockIdx.x;  // batch*head index
    int q_row = blockIdx.y;  // query row
    int tid = threadIdx.x;

    const float* q_ptr = Q + bh * S * D + q_row * D;
    const float* k_base = K + bh * S * D;
    const float* v_base = V + bh * S * D;
    float* o_ptr = output + bh * S * D + q_row * D;

    // 使用动态shared memory存储scores
    extern __shared__ float smem[];  // size = S

    // Step 1: 计算 attention scores
    // 每个thread计算部分dot product, 然后reduce
    for (int j = 0; j < S; ++j) {
        float dot = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) {
            dot += q_ptr[d] * k_base[j * D + d];
        }
        // Reduce within block
        __shared__ float reduce_buf[32];
        int warp_id = tid / 32;
        int lane = tid % 32;
        dot = warp_reduce_sum(dot);
        if (lane == 0) reduce_buf[warp_id] = dot;
        __syncthreads();
        if (warp_id == 0) {
            int nw = (blockDim.x + 31) / 32;
            dot = (lane < nw) ? reduce_buf[lane] : 0.0f;
            dot = warp_reduce_sum(dot);
        }
        __syncthreads();
        if (tid == 0) smem[j] = dot * scale;
        __syncthreads();
    }

    // Step 2: Softmax over scores
    if (tid == 0) {
        float max_val = -FLT_MAX;
        for (int j = 0; j < S; ++j) {
            max_val = fmaxf(max_val, smem[j]);
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < S; ++j) {
            smem[j] = expf(smem[j] - max_val);
            sum_exp += smem[j];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < S; ++j) {
            smem[j] *= inv_sum;
        }
    }
    __syncthreads();

    // Step 3: 计算 output = scores @ V
    for (int d = tid; d < D; d += blockDim.x) {
        float val = 0.0f;
        for (int j = 0; j < S; ++j) {
            val += smem[j] * v_base[j * D + d];
        }
        o_ptr[d] = val;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_sdpa(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_TORCH_TENSOR_DTYPE(Q, torch::kFloat32);
    // Q, K, V: [B*H, S, D]
    int BH = Q.size(0), S = Q.size(1), D = Q.size(2);
    float scale = 1.0f / sqrtf((float)D);
    auto output = torch::empty_like(Q);

    int block = min(256, D);
    block = ((block + 31) / 32) * 32;
    dim3 grid(BH, S);
    size_t smem_size = S * sizeof(float) + 32 * sizeof(float);  // scores + reduce buffer

    sdpa_naive_kernel<<<grid, block, smem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        output.data_ptr<float>(), S, D, scale);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_sdpa)
}
