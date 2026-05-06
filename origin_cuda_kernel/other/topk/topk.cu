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

// ==================== TopK Kernel ====================
// 对每行 [M, N] 求 Top-K 个最大值及其索引
//
// 【任务划分逻辑】
//
// ▸ 方法: Per-row partial sort (selection)
//   - grid = M (每行一个 block)
//   - 每个 block 协作完成一行的 Top-K 选择
//   - 策略: K 轮迭代，每轮用 block-level reduce 找当前最大值
//     然后将已选中的位置标记为 -INF
//   - 适合 K 较小的情况 (K << N)
//
// ▸ 优化版: Radix-based selection (BitFE)
//   - 从最高位到最低位逐 bit 决定阈值
//   - 每轮统计: 当前 bit=1 的元素个数 >= K ? 则保留 bit=1 的
//   - O(32 * N / blockDim.x) 找到第 K 大的值
//   - 最后一遍 pass 收集 >= threshold 的元素

#define TOPK_BLOCK 256

// ---- TopK: K轮选择法 (适合小 K) ----
// input: [M, N], output_val: [M, K], output_idx: [M, K]
__global__ void topk_kernel(const float* input, float* output_val, int64_t* output_idx,
                             int M, int N, int K) {
    extern __shared__ char shared_mem[];
    float* smax = (float*)shared_mem;              // [1] current max
    int* sidx = (int*)(smax + 1);                  // [1] current max idx
    float* svals = (float*)(sidx + 1);             // [TOPK_BLOCK] per-thread max
    int* sindices = (int*)(svals + TOPK_BLOCK);    // [TOPK_BLOCK] per-thread idx

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* row_in = input + row * N;
    float* row_out_val = output_val + row * K;
    int64_t* row_out_idx = output_idx + row * K;

    // 用一个 mask 数组标记已选元素（用 -INF 覆盖）
    // 为节省内存，我们用 output 记录每轮结果，直接修改 shared 值

    for (int k = 0; k < K; ++k) {
        // 每个 thread 找自己负责区间的最大值
        float local_max = -FLT_MAX;
        int local_idx = -1;
        for (int i = tid; i < N; i += blockDim.x) {
            float val = row_in[i];
            // 跳过已选过的（检查是否已在之前的 output 中）
            bool already_picked = false;
            for (int p = 0; p < k; ++p) {
                if ((int)row_out_idx[p] == i) {
                    already_picked = true;
                    break;
                }
            }
            if (!already_picked && val > local_max) {
                local_max = val;
                local_idx = i;
            }
        }
        svals[tid] = local_max;
        sindices[tid] = local_idx;
        __syncthreads();

        // Block-level reduce to find global max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (svals[tid + stride] > svals[tid]) {
                    svals[tid] = svals[tid + stride];
                    sindices[tid] = sindices[tid + stride];
                }
            }
            __syncthreads();
        }

        // Thread 0 writes result
        if (tid == 0) {
            row_out_val[k] = svals[0];
            row_out_idx[k] = sindices[0];
        }
        __syncthreads();
    }
}

// ---- TopK: Radix Selection (适合大 K) ----
// 基于位选择: 从高位到低位确定第 K 大的阈值
__global__ void topk_radix_kernel(const float* input, float* output_val, int64_t* output_idx,
                                   int N, int K) {
    // 单 block 处理一行
    extern __shared__ int shared_count[];
    int tid = threadIdx.x;

    // 将 float 转为可比较的 unsigned (flip sign bit)
    // 这样 unsigned 比较等价于 float 比较
    auto float_to_ordered = [](float f) -> unsigned int {
        unsigned int u = __float_as_uint(f);
        unsigned int mask = -(int)(u >> 31) | 0x80000000u;
        return u ^ mask;
    };

    auto ordered_to_float = [](unsigned int u) -> float {
        unsigned int mask = ((u >> 31) - 1) | 0x80000000u;
        return __uint_as_float(u ^ mask);
    };

    // 逐 bit 从高到低确定阈值
    unsigned int threshold = 0;
    int remaining = K;

    for (int bit = 31; bit >= 0; --bit) {
        // 统计当前 bit=1 的元素个数
        int count = 0;
        for (int i = tid; i < N; i += blockDim.x) {
            unsigned int val = float_to_ordered(input[i]);
            if ((val & ~((1u << bit) - 1)) >= (threshold | (1u << bit))) {
                count++;
            }
        }

        // Block reduce
        shared_count[tid] = count;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shared_count[tid] += shared_count[tid + s];
            __syncthreads();
        }

        int total_above = shared_count[0];
        if (total_above >= remaining) {
            threshold |= (1u << bit);
        } else {
            remaining -= total_above;
        }
        __syncthreads();
    }

    // 收集 >= threshold 的元素
    float thresh_f = ordered_to_float(threshold);
    int write_pos = 0;
    for (int i = 0; i < N && write_pos < K; ++i) {
        if (input[i] >= thresh_f) {
            if (tid == 0) {
                output_val[write_pos] = input[i];
                output_idx[write_pos] = i;
            }
            write_pos++;
        }
    }
}

// ==================== Torch Bindings ====================

std::vector<torch::Tensor> torch_topk(torch::Tensor input, int K) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output_val = torch::empty({M, K}, input.options());
    auto output_idx = torch::empty({M, K}, torch::dtype(torch::kInt64).device(input.device()));

    int block = TOPK_BLOCK;
    size_t smem = sizeof(float) + sizeof(int) + block * sizeof(float) + block * sizeof(int);
    topk_kernel<<<M, block, smem>>>(input.data_ptr<float>(),
        output_val.data_ptr<float>(), output_idx.data_ptr<int64_t>(), M, N, K);
    return {output_val, output_idx};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_topk)
}
