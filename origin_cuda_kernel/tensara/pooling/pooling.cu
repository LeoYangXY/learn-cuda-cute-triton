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

// ==================== Pooling Kernels ====================
// 1D Average Pooling, 1D Max Pooling, 2D Average Pooling, 2D Max Pooling
//
// 【任务划分逻辑】
// Pooling 和卷积类似：每个 output 元素由 input 的一个滑窗区域计算得出。
//
// ▸ 1D Pooling:
//   - 输出长度 out_len = (N - K) / S + 1
//   - 每个 thread 负责 1 个 output 元素 (标量版) 或 4 个 (vec4 版)
//   - thread 计算: output[idx] = pool(input[idx*S .. idx*S+K-1])
//   - vec4 版本: 每个 thread 依次处理 base+0, base+1, base+2, base+3
//     减少 thread 总数，降低调度开销
//
// ▸ 2D Pooling:
//   - 输出尺寸: OH = (H-KH)/SH+1, OW = (W-KW)/SW+1
//   - 将 output 展平为 1D: total = OH * OW
//   - grid = ceil(total / blockDim.x), 每个 thread 负责 1 个 output 像素
//   - 通过 idx/OW 和 idx%OW 反算 (oh, ow)
//   - 计算 input 起始坐标: h_start=oh*SH, w_start=ow*SW
//   - 双重循环遍历 KH×KW 窗口，做 max 或 avg
//
// 性能特点:
//   - Pooling 是 bandwidth-bound (计算很少，主要是读内存)
//   - 1D 情况下 stride=1 时相邻 thread 的 input 区域高度重叠 → L1/L2 cache 能帮忙
//   - 2D 情况若 kernel 较大可考虑 shared memory tiling (当前实现依赖 L2 cache)

// ---- 1D Average Pooling ----
// input: [N], kernel_size: K, stride: S
// output: [(N - K) / S + 1]
__global__ void avg_pool1d_kernel(const float* input, float* output,
                                   int N, int K, int S, int out_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_len) {
        int start = idx * S;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += input[start + k];
        }
        output[idx] = sum / (float)K;
    }
}

// 向量化版本: 每个thread处理4个output
__global__ void avg_pool1d_vec4_kernel(const float* input, float* output,
                                        int N, int K, int S, int out_len) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float inv_k = 1.0f / (float)K;

    for (int i = 0; i < 4; ++i) {
        int idx = base + i;
        if (idx < out_len) {
            int start = idx * S;
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += input[start + k];
            }
            output[idx] = sum * inv_k;
        }
    }
}

// ---- 1D Max Pooling ----
__global__ void max_pool1d_kernel(const float* input, float* output,
                                   int N, int K, int S, int out_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_len) {
        int start = idx * S;
        float max_val = -FLT_MAX;
        for (int k = 0; k < K; ++k) {
            max_val = fmaxf(max_val, input[start + k]);
        }
        output[idx] = max_val;
    }
}

__global__ void max_pool1d_vec4_kernel(const float* input, float* output,
                                        int N, int K, int S, int out_len) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    for (int i = 0; i < 4; ++i) {
        int idx = base + i;
        if (idx < out_len) {
            int start = idx * S;
            float max_val = -FLT_MAX;
            for (int k = 0; k < K; ++k) {
                max_val = fmaxf(max_val, input[start + k]);
            }
            output[idx] = max_val;
        }
    }
}

// ---- 2D Average Pooling ----
// input: [H, W], kernel: KH x KW, stride: SH x SW
// output: [(H-KH)/SH+1, (W-KW)/SW+1]
__global__ void avg_pool2d_kernel(const float* input, float* output,
                                   int H, int W, int KH, int KW,
                                   int SH, int SW, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OH * OW) {
        int oh = idx / OW;
        int ow = idx % OW;
        int h_start = oh * SH;
        int w_start = ow * SW;

        float sum = 0.0f;
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                sum += input[(h_start + kh) * W + (w_start + kw)];
            }
        }
        output[idx] = sum / (float)(KH * KW);
    }
}

// ---- 2D Max Pooling ----
__global__ void max_pool2d_kernel(const float* input, float* output,
                                   int H, int W, int KH, int KW,
                                   int SH, int SW, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OH * OW) {
        int oh = idx / OW;
        int ow = idx % OW;
        int h_start = oh * SH;
        int w_start = ow * SW;

        float max_val = -FLT_MAX;
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                max_val = fmaxf(max_val, input[(h_start + kh) * W + (w_start + kw)]);
            }
        }
        output[idx] = max_val;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_avg_pool1d(torch::Tensor input, int K, int S) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    int out_len = (N - K) / S + 1;
    auto output = torch::empty({out_len}, input.options());
    int block = 256;
    int grid = CEIL(out_len, block);
    avg_pool1d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, K, S, out_len);
    return output;
}

torch::Tensor torch_avg_pool1d_vec4(torch::Tensor input, int K, int S) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    int out_len = (N - K) / S + 1;
    auto output = torch::empty({out_len}, input.options());
    int block = 256;
    int grid = CEIL(CEIL(out_len, 4), block);
    avg_pool1d_vec4_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, K, S, out_len);
    return output;
}

torch::Tensor torch_max_pool1d(torch::Tensor input, int K, int S) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    int out_len = (N - K) / S + 1;
    auto output = torch::empty({out_len}, input.options());
    int block = 256;
    int grid = CEIL(out_len, block);
    max_pool1d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, K, S, out_len);
    return output;
}

torch::Tensor torch_max_pool1d_vec4(torch::Tensor input, int K, int S) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    int out_len = (N - K) / S + 1;
    auto output = torch::empty({out_len}, input.options());
    int block = 256;
    int grid = CEIL(CEIL(out_len, 4), block);
    max_pool1d_vec4_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, K, S, out_len);
    return output;
}

torch::Tensor torch_avg_pool2d(torch::Tensor input, int KH, int KW, int SH, int SW) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int H = input.size(0), W = input.size(1);
    int OH = (H - KH) / SH + 1;
    int OW = (W - KW) / SW + 1;
    auto output = torch::empty({OH, OW}, input.options());
    int total = OH * OW;
    int block = 256;
    int grid = CEIL(total, block);
    avg_pool2d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(),
                                        H, W, KH, KW, SH, SW, OH, OW);
    return output;
}

torch::Tensor torch_max_pool2d(torch::Tensor input, int KH, int KW, int SH, int SW) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int H = input.size(0), W = input.size(1);
    int OH = (H - KH) / SH + 1;
    int OW = (W - KW) / SW + 1;
    auto output = torch::empty({OH, OW}, input.options());
    int total = OH * OW;
    int block = 256;
    int grid = CEIL(total, block);
    max_pool2d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(),
                                        H, W, KH, KW, SH, SW, OH, OW);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_avg_pool1d)
    TORCH_BINDING_COMMON_EXTENSION(torch_avg_pool1d_vec4)
    TORCH_BINDING_COMMON_EXTENSION(torch_max_pool1d)
    TORCH_BINDING_COMMON_EXTENSION(torch_max_pool1d_vec4)
    TORCH_BINDING_COMMON_EXTENSION(torch_avg_pool2d)
    TORCH_BINDING_COMMON_EXTENSION(torch_max_pool2d)
}
