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

// ==================== Sorting Kernel ====================
// Bitonic Sort for GPU
//
// 【任务划分逻辑】
// Bitonic Sort 是一种非常适合 GPU 的排序网络算法：
// 所有比较-交换操作的索引对在编译时确定 → 完全数据无关 → 天然并行
//
// ▸ Shared Memory 版本 (N <= 1024):
//   - 1 个 block, block_size = next_power_of_2(N)
//   - 所有数据加载到 smem，整个排序在 smem 中完成
//   - 外循环 k = 2,4,8,...,N (merge 的步长，从小到大)
//   - 内循环 j = k/2, k/4, ..., 1 (比较距离从大到小)
//   - 每个 thread 负责 1 个位置: 计算 ixj = tid ^ j
//     if ixj > tid: 这对 (tid, ixj) 由 tid 负责比较交换
//     方向由 (tid & k) 决定: == 0 → ascending, != 0 → descending
//   - __syncthreads() 确保每轮所有交换完成后才进行下一轮
//
// ▸ Global Memory 版本 (N > 1024):
//   - 先 pad 到 2 的幂次，填充 FLT_MAX
//   - 每轮 (j, k) 启动一次 kernel:
//     grid = ceil(N / blockDim.x), 每个 thread 负责 1 个位置
//   - 总共 O(log²(N)) 次 kernel launch
//   - 每次 launch 内部完全并行，thread 之间无依赖

// ---- Bitonic Sort (in-place) ----
// Only works for N = power of 2
__global__ void bitonic_sort_step(float* data, int j, int k, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int ixj = idx ^ j;
    if (ixj > idx) {
        if ((idx & k) == 0) {
            // ascending
            if (data[idx] > data[ixj]) {
                float tmp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = tmp;
            }
        } else {
            // descending
            if (data[idx] < data[ixj]) {
                float tmp = data[idx];
                data[idx] = data[ixj];
                data[ixj] = tmp;
            }
        }
    }
}

// Shared memory bitonic sort for small arrays (N <= 1024)
__global__ void bitonic_sort_shared(float* data, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    // Load
    smem[tid] = (tid < N) ? data[tid] : FLT_MAX;
    __syncthreads();

    // Bitonic sort in shared memory
    int n = blockDim.x; // assume power of 2
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (smem[tid] > smem[ixj]) {
                        float tmp = smem[tid];
                        smem[tid] = smem[ixj];
                        smem[ixj] = tmp;
                    }
                } else {
                    if (smem[tid] < smem[ixj]) {
                        float tmp = smem[tid];
                        smem[tid] = smem[ixj];
                        smem[ixj] = tmp;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Store
    if (tid < N) data[tid] = smem[tid];
}

// ==================== Torch Bindings ====================

torch::Tensor torch_bitonic_sort(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = input.clone();

    // Pad to next power of 2 if needed
    int n = 1;
    while (n < N) n <<= 1;

    if (n <= 1024) {
        // Use shared memory version
        bitonic_sort_shared<<<1, n, n * sizeof(float)>>>(output.data_ptr<float>(), N);
    } else {
        // Pad with FLT_MAX
        auto padded = torch::full({n}, FLT_MAX, input.options());
        padded.slice(0, 0, N).copy_(output);

        int block = 256;
        int grid = CEIL(n, block);
        for (int k = 2; k <= n; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                bitonic_sort_step<<<grid, block>>>(padded.data_ptr<float>(), j, k, n);
            }
        }
        output = padded.slice(0, 0, N).contiguous();
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_bitonic_sort)
}
