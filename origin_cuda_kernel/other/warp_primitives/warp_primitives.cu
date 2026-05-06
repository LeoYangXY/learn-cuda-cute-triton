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

// ==================== Warp-level Primitives ====================
// 面试高频: warp shuffle 实现 reduce / scan / broadcast
//
// 【核心原语】
// __shfl_sync(mask, val, srcLane)        — 从指定 lane 读值 (broadcast)
// __shfl_xor_sync(mask, val, laneMask)   — butterfly exchange
// __shfl_down_sync(mask, val, offset)    — 向下偏移读值 (用于 tree reduce)
// __shfl_up_sync(mask, val, offset)      — 向上偏移读值 (用于 inclusive scan)
//
// 【面试常见问题】
// 1. 用 warp shuffle 实现 sum/max reduce (无 shared memory)
// 2. 用 warp shuffle 实现 prefix sum (inclusive/exclusive scan)
// 3. warp divergence 的性能影响
// 4. __ballot_sync + __popc 实现 warp-level filter/compact

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// ---- Warp Reduce Sum ----
// 经典的 butterfly pattern: 5 步完成 32 个 lane 的 sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---- Warp Reduce Max ----
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// ---- Warp Inclusive Scan (prefix sum) ----
// 使用 shfl_up: lane i 的结果 = sum(val[0..i])
__device__ __forceinline__ float warp_inclusive_scan(float val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        float other = __shfl_up_sync(0xffffffff, val, offset);
        if ((threadIdx.x % WARP_SIZE) >= offset) {
            val += other;
        }
    }
    return val;
}

// ---- Block Reduce Sum (two-level: warp reduce + cross-warp reduce) ----
__global__ void block_reduce_sum_kernel(const float* input, float* output, int N) {
    __shared__ float warp_sums[NUM_WARPS];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Level 1: 每个 thread 从 global memory 加载 (grid-stride loop)
    float val = 0.0f;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        val += input[i];
    }

    // Level 2: warp 内 shuffle reduce
    val = warp_reduce_sum(val);

    // Warp leader 写入 shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Level 3: 第一个 warp 做跨 warp reduce
    if (warp_id == 0) {
        val = (lane_id < NUM_WARPS) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            atomicAdd(output, val);
        }
    }
}

// ---- Block Inclusive Scan (Blelloch-style with warp primitives) ----
// input: [N], output: [N], per-block prefix sum
__global__ void block_scan_kernel(const float* input, float* output, int N) {
    __shared__ float warp_totals[NUM_WARPS];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Load
    float val = (global_idx < N) ? input[global_idx] : 0.0f;

    // Warp-level inclusive scan
    val = warp_inclusive_scan(val);

    // 每个 warp 的最后一个 lane 保存 warp total
    if (lane_id == WARP_SIZE - 1) {
        warp_totals[warp_id] = val;
    }
    __syncthreads();

    // 第一个 warp 对 warp_totals 做 inclusive scan (得到 prefix of warp sums)
    if (warp_id == 0 && lane_id < NUM_WARPS) {
        float wt = warp_totals[lane_id];
        // Scan across warps
        for (int offset = 1; offset < NUM_WARPS; offset <<= 1) {
            float other = __shfl_up_sync(0xffffffff, wt, offset);
            if (lane_id >= offset) wt += other;
        }
        warp_totals[lane_id] = wt;
    }
    __syncthreads();

    // 每个 thread 加上前面 warp 的 total
    if (warp_id > 0) {
        val += warp_totals[warp_id - 1];
    }

    if (global_idx < N) {
        output[global_idx] = val;
    }
}

// ---- Warp-level Filter/Compact (用 ballot + popc) ----
// 从输入中筛选出 > threshold 的元素，紧凑排列到 output
__global__ void warp_filter_kernel(const float* input, float* output,
                                    int* output_count, int N, float threshold) {
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int lane_id = tid % WARP_SIZE;

    float val = (global_idx < N) ? input[global_idx] : -FLT_MAX;
    bool pred = (val > threshold) && (global_idx < N);

    // ballot: 获取 warp 内所有 lane 的 predicate bitmap
    unsigned int mask = __ballot_sync(0xffffffff, pred);

    // popc: 统计 mask 中 bit=1 的个数 → 该 warp 有多少元素通过筛选
    int warp_count = __popc(mask);

    // 该 lane 之前有多少个通过的 (exclusive prefix count)
    unsigned int lower_mask = (1u << lane_id) - 1;
    int lane_prefix = __popc(mask & lower_mask);

    // Warp leader 用 atomicAdd 分配全局写入位置
    __shared__ int warp_offset[NUM_WARPS];
    int warp_id = tid / WARP_SIZE;
    if (lane_id == 0 && warp_count > 0) {
        warp_offset[warp_id] = atomicAdd(output_count, warp_count);
    }
    __syncthreads();

    // 写入
    if (pred) {
        int write_idx = warp_offset[warp_id] + lane_prefix;
        output[write_idx] = val;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_block_reduce_sum(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::zeros({1}, input.options());
    int grid = min(256, CEIL(N, BLOCK_SIZE));
    block_reduce_sum_kernel<<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

torch::Tensor torch_block_scan(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    int grid = CEIL(N, BLOCK_SIZE);
    block_scan_kernel<<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), N);
    return output;
}

std::vector<torch::Tensor> torch_warp_filter(torch::Tensor input, float threshold) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    auto count = torch::zeros({1}, torch::dtype(torch::kInt32).device(input.device()));
    int grid = CEIL(N, BLOCK_SIZE);
    warp_filter_kernel<<<grid, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(),
        count.data_ptr<int>(), N, threshold);
    int actual_count = count.item<int>();
    return {output.slice(0, 0, actual_count), count};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_block_reduce_sum)
    TORCH_BINDING_COMMON_EXTENSION(torch_block_scan)
    TORCH_BINDING_COMMON_EXTENSION(torch_warp_filter)
}
