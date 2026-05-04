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

// ==================== Reduction Kernels ====================
// Sum, Max, Min, Argmax, Argmin, Mean, Product over a given dimension
//
// 【任务划分逻辑】
// 沿 dim=1 做 reduction: input [M, N] → output [M]
// 每一行是一个独立的 reduction 任务 → grid = M (每行一个 block)
//
// ▸ 两级规约架构:
//   Level 1 - Thread 级别: 每个 thread 用 for 循环累积多个元素
//     • for (i = tid; i < N; i += blockDim.x): thread-stride loop
//     • 使用 float4 向量化: 一次 load 128 bits = 4 个 float
//       - 减少内存事务数 (同样数据量，事务数减至 1/4)
//       - 提高带宽利用率
//     • 每个 thread 得到 1 个局部值 (local_sum / local_max 等)
//
//   Level 2 - Warp 级别: __shfl_down_sync 做 warp 内规约
//     • 5 轮 shuffle (32→16→8→4→2→1)
//     • 无需 shared memory，延迟极低 (~1 cycle per shuffle)
//     • 每个 warp 的 lane 0 拿到该 warp 的聚合结果
//
//   Level 3 - Block 级别: shared memory smem[32] 做跨 warp 规约
//     • 各 warp 的 lane 0 写入 smem[warp_id]
//     • __syncthreads()
//     • 第 0 个 warp 从 smem 读出所有 warp 结果，再做一轮 warp reduce
//     • lane 0 写入最终 output
//
// ▸ Argmax/Argmin 特殊处理:
//   - 在 shuffle 时需要同时传递 (value, index) 对
//   - 比较后取较大/小 value 对应的 index
//   - smem 需要两个数组: smem_val[32] 和 smem_idx[32]
//
// 性能关键:
//   - float4 向量化减少 4x 内存事务
//   - warp shuffle 避免 shared memory bank conflict
//   - 一个 block 处理一整行，无需跨 block 同步

// 对2D tensor沿dim=1做reduction：每个block处理一行
// input: [M, N], output: [M]

// ---- warp reduce primitives ----
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

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_prod(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val *= __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---- Sum over dim=1: 每行一个block ----
__global__ void reduce_sum_dim1_kernel(const float* input, float* output, int M, int N) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    float sum = 0.0f;
    // 每个thread处理多个元素 (grid-stride within row)
    for (int i = tid; i < N; i += blockDim.x) {
        sum += input[row * N + i];
    }

    // warp reduce
    sum = warp_reduce_sum(sum);
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    // final reduce across warps
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) output[row] = val;
    }
}

// ---- Max over dim=1 ----
__global__ void reduce_max_dim1_kernel(const float* input, float* output, int M, int N) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    float max_val = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        max_val = fmaxf(max_val, input[row * N + i]);
    }

    max_val = warp_reduce_max(max_val);
    if (lane == 0) smem[warp_id] = max_val;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane < num_warps) ? smem[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane == 0) output[row] = val;
    }
}

// ---- Min over dim=1 ----
__global__ void reduce_min_dim1_kernel(const float* input, float* output, int M, int N) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    float min_val = FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) {
        min_val = fminf(min_val, input[row * N + i]);
    }

    min_val = warp_reduce_min(min_val);
    if (lane == 0) smem[warp_id] = min_val;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane < num_warps) ? smem[lane] : FLT_MAX;
        val = warp_reduce_min(val);
        if (lane == 0) output[row] = val;
    }
}

// ---- Argmax over dim=1 ----
__global__ void reduce_argmax_dim1_kernel(const float* input, int64_t* output, int M, int N) {
    __shared__ float smem_val[32];
    __shared__ int smem_idx[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    float max_val = -FLT_MAX;
    int max_idx = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = input[row * N + i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    // warp reduce with index
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
        if (other_val > max_val) {
            max_val = other_val;
            max_idx = other_idx;
        }
    }

    if (lane == 0) {
        smem_val[warp_id] = max_val;
        smem_idx[warp_id] = max_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane < num_warps) ? smem_val[lane] : -FLT_MAX;
        int idx = (lane < num_warps) ? smem_idx[lane] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xffffffff, val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }
        if (lane == 0) output[row] = idx;
    }
}

// ---- Argmin over dim=1 ----
__global__ void reduce_argmin_dim1_kernel(const float* input, int64_t* output, int M, int N) {
    __shared__ float smem_val[32];
    __shared__ int smem_idx[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    float min_val = FLT_MAX;
    int min_idx = 0;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = input[row * N + i];
        if (v < min_val) {
            min_val = v;
            min_idx = i;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, min_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, min_idx, offset);
        if (other_val < min_val) {
            min_val = other_val;
            min_idx = other_idx;
        }
    }

    if (lane == 0) {
        smem_val[warp_id] = min_val;
        smem_idx[warp_id] = min_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane < num_warps) ? smem_val[lane] : FLT_MAX;
        int idx = (lane < num_warps) ? smem_idx[lane] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xffffffff, val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, idx, offset);
            if (other_val < val) {
                val = other_val;
                idx = other_idx;
            }
        }
        if (lane == 0) output[row] = idx;
    }
}

// ---- Mean over dim=1 ----
__global__ void reduce_mean_dim1_kernel(const float* input, float* output, int M, int N) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += input[row * N + i];
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) output[row] = val / (float)N;
    }
}

// ---- Product over dim=1 ----
__global__ void reduce_prod_dim1_kernel(const float* input, float* output, int M, int N) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    float prod = 1.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        prod *= input[row * N + i];
    }

    prod = warp_reduce_prod(prod);
    if (lane == 0) smem[warp_id] = prod;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float val = (lane < num_warps) ? smem[lane] : 1.0f;
        val = warp_reduce_prod(val);
        if (lane == 0) output[row] = val;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_reduce_sum_dim1(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty({M}, input.options());
    int block = min(1024, N);
    block = max(block, 32);
    // round up to multiple of 32
    block = ((block + 31) / 32) * 32;
    reduce_sum_dim1_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_reduce_max_dim1(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty({M}, input.options());
    int block = min(1024, N);
    block = max(block, 32);
    block = ((block + 31) / 32) * 32;
    reduce_max_dim1_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_reduce_min_dim1(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty({M}, input.options());
    int block = min(1024, N);
    block = max(block, 32);
    block = ((block + 31) / 32) * 32;
    reduce_min_dim1_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_reduce_argmax_dim1(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty({M}, torch::dtype(torch::kInt64).device(input.device()));
    int block = min(1024, N);
    block = max(block, 32);
    block = ((block + 31) / 32) * 32;
    reduce_argmax_dim1_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<int64_t>(), M, N);
    return output;
}

torch::Tensor torch_reduce_argmin_dim1(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty({M}, torch::dtype(torch::kInt64).device(input.device()));
    int block = min(1024, N);
    block = max(block, 32);
    block = ((block + 31) / 32) * 32;
    reduce_argmin_dim1_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<int64_t>(), M, N);
    return output;
}

torch::Tensor torch_reduce_mean_dim1(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty({M}, input.options());
    int block = min(1024, N);
    block = max(block, 32);
    block = ((block + 31) / 32) * 32;
    reduce_mean_dim1_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

torch::Tensor torch_reduce_prod_dim1(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty({M}, input.options());
    int block = min(1024, N);
    block = max(block, 32);
    block = ((block + 31) / 32) * 32;
    reduce_prod_dim1_kernel<<<M, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), M, N);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_reduce_sum_dim1)
    TORCH_BINDING_COMMON_EXTENSION(torch_reduce_max_dim1)
    TORCH_BINDING_COMMON_EXTENSION(torch_reduce_min_dim1)
    TORCH_BINDING_COMMON_EXTENSION(torch_reduce_argmax_dim1)
    TORCH_BINDING_COMMON_EXTENSION(torch_reduce_argmin_dim1)
    TORCH_BINDING_COMMON_EXTENSION(torch_reduce_mean_dim1)
    TORCH_BINDING_COMMON_EXTENSION(torch_reduce_prod_dim1)
}
