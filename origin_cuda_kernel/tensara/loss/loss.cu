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

// ==================== Loss Function Kernels ====================
// Huber Loss, MSE Loss, Hinge Loss, Cosine Similarity
//
// 【任务划分逻辑】
// Loss 函数通常分为两步: ① 逐元素计算 ② reduce 求均值
//
// ▸ Scalar Loss (Huber, MSE, Hinge): 全局 reduce
//   - grid = min(256, ceil(N/block))，使用少量 block 避免 atomicAdd 竞争
//   - 每个 thread 用 grid-stride loop 处理多个元素:
//     for(i = global_idx; i < N; i += blockDim.x * gridDim.x)
//   - Block 内做 warp shuffle reduce → thread 0 atomicAdd 到全局 partial_sum
//   - Host 端最终 / N 得到 mean
//   - 优势: 一次 kernel launch 完成所有工作，无需中间 buffer
//
// ▸ Cosine Similarity: 按行操作
//   - grid = M (样本数)，每行一个 block
//   - 每个 thread 用 thread-stride loop 累积 (dot, norm_a², norm_b²) 三个值
//   - warp reduce + smem 跨 warp reduce (3 组 smem: dot/na/nb)
//   - thread 0 计算 cos_sim = dot / (sqrt(na) * sqrt(nb))
//   - 设计: 3 个独立的 reduce 可以并行在同一个 thread 中累积

// ---- warp/block reduce sum ----
template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float smem[32];
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

// ---- Huber Loss (element-wise, then reduce mean) ----
// L_delta(a) = 0.5 * a^2 if |a| <= delta, else delta * (|a| - 0.5*delta)
// a = pred - target
__global__ void huber_loss_kernel(const float* pred, const float* target,
                                   float* partial_sums, int N, float delta) {
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        float diff = pred[i] - target[i];
        float abs_diff = fabsf(diff);
        if (abs_diff <= delta) {
            local_sum += 0.5f * diff * diff;
        } else {
            local_sum += delta * (abs_diff - 0.5f * delta);
        }
    }

    local_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(partial_sums, local_sum);
    }
}

// ---- MSE Loss ----
__global__ void mse_loss_kernel(const float* pred, const float* target,
                                 float* partial_sums, int N) {
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        float diff = pred[i] - target[i];
        local_sum += diff * diff;
    }

    local_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(partial_sums, local_sum);
    }
}

// ---- Hinge Loss: mean(max(0, 1 - y * pred)) ----
// y should be +1 or -1
__global__ void hinge_loss_kernel(const float* pred, const float* target,
                                   float* partial_sums, int N) {
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    float local_sum = 0.0f;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        float margin = 1.0f - target[i] * pred[i];
        local_sum += fmaxf(0.0f, margin);
    }

    local_sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(partial_sums, local_sum);
    }
}

// ---- Cosine Similarity ----
// cos_sim = dot(a, b) / (||a|| * ||b||)
// 使用一个block处理一对向量
__global__ void cosine_similarity_kernel(const float* a, const float* b,
                                          float* output, int N) {
    __shared__ float smem[32 * 3]; // dot, norm_a, norm_b
    float* smem_dot = smem;
    float* smem_na = smem + 32;
    float* smem_nb = smem + 64;

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    const float* va = a + row * N;
    const float* vb = b + row * N;

    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float ai = va[i], bi = vb[i];
        dot += ai * bi;
        na += ai * ai;
        nb += bi * bi;
    }

    // warp reduce
    dot = warp_reduce_sum(dot);
    na = warp_reduce_sum(na);
    nb = warp_reduce_sum(nb);

    if (lane == 0) {
        smem_dot[warp_id] = dot;
        smem_na[warp_id] = na;
        smem_nb[warp_id] = nb;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        dot = (lane < num_warps) ? smem_dot[lane] : 0.0f;
        na = (lane < num_warps) ? smem_na[lane] : 0.0f;
        nb = (lane < num_warps) ? smem_nb[lane] : 0.0f;
        dot = warp_reduce_sum(dot);
        na = warp_reduce_sum(na);
        nb = warp_reduce_sum(nb);
        if (lane == 0) {
            float denom = sqrtf(na) * sqrtf(nb);
            output[row] = (denom > 1e-8f) ? (dot / denom) : 0.0f;
        }
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_huber_loss(torch::Tensor pred, torch::Tensor target, float delta) {
    CHECK_TORCH_TENSOR_DTYPE(pred, torch::kFloat32);
    int N = pred.numel();
    auto output = torch::zeros({1}, pred.options());
    int block = 256;
    int grid = min(256, CEIL(N, block));
    huber_loss_kernel<<<grid, block>>>(pred.data_ptr<float>(), target.data_ptr<float>(),
                                        output.data_ptr<float>(), N, delta);
    // mean
    output = output / (float)N;
    return output;
}

torch::Tensor torch_mse_loss(torch::Tensor pred, torch::Tensor target) {
    CHECK_TORCH_TENSOR_DTYPE(pred, torch::kFloat32);
    int N = pred.numel();
    auto output = torch::zeros({1}, pred.options());
    int block = 256;
    int grid = min(256, CEIL(N, block));
    mse_loss_kernel<<<grid, block>>>(pred.data_ptr<float>(), target.data_ptr<float>(),
                                      output.data_ptr<float>(), N);
    output = output / (float)N;
    return output;
}

torch::Tensor torch_hinge_loss(torch::Tensor pred, torch::Tensor target) {
    CHECK_TORCH_TENSOR_DTYPE(pred, torch::kFloat32);
    int N = pred.numel();
    auto output = torch::zeros({1}, pred.options());
    int block = 256;
    int grid = min(256, CEIL(N, block));
    hinge_loss_kernel<<<grid, block>>>(pred.data_ptr<float>(), target.data_ptr<float>(),
                                        output.data_ptr<float>(), N);
    output = output / (float)N;
    return output;
}

torch::Tensor torch_cosine_similarity(torch::Tensor a, torch::Tensor b) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32);
    int M = a.size(0), N = a.size(1);
    auto output = torch::empty({M}, a.options());
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    cosine_similarity_kernel<<<M, block>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                            output.data_ptr<float>(), N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_huber_loss)
    TORCH_BINDING_COMMON_EXTENSION(torch_mse_loss)
    TORCH_BINDING_COMMON_EXTENSION(torch_hinge_loss)
    TORCH_BINDING_COMMON_EXTENSION(torch_cosine_similarity)
}
