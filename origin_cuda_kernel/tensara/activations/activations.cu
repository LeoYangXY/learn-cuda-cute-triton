#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(a) *(float4*)(&(a))

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
  if (((T).options().dtype() != (th_type))) {                \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);    \
  }

// ==================== Activation Function Kernels ====================
// 实现: ReLU, LeakyReLU, GELU, Sigmoid, Tanh, ELU, SELU, Softplus, Swish, HardSigmoid
//
// 【任务划分逻辑】
// 激活函数是最典型的 Elementwise 算子：每个输出元素的计算完全独立，只依赖对应位置的输入元素。
// 
// ▸ 标量版本 (_f32)：
//   - grid = ceil(N / blockDim.x)，每个 thread 负责 1 个元素
//   - thread_idx = blockIdx.x * blockDim.x + threadIdx.x → 全局元素索引
//   - 计算: output[idx] = f(input[idx])
//   - 优点: 逻辑简单
//   - 瓶颈: 每次 load/store 只搬 4 bytes，内存事务利用率低
//
// ▸ 向量化版本 (_f32x4)：
//   - grid = ceil(N / (blockDim.x * 4))，每个 thread 负责连续 4 个元素
//   - 使用 float4 (128-bit) 向量化加载/存储
//   - 一个 warp 的 32 个 thread 合并访问 32*16 = 512 bytes，触发更少的内存事务
//   - 优化原理: 减少总线程数 → 降低 register pressure + scheduling overhead
//              同时 128-bit load 能更好利用内存带宽
//   - 尾部处理: 如果 idx+3 >= N，退化为标量逐个处理
//
// 对于所有激活函数，核心思想完全一致：只是 f(x) 不同。
// 这类 bandwidth-bound kernel 的性能关键在于: 向量化访存 > 计算优化

// ---- ReLU: max(0, x) ----
__global__ void relu_f32_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_f32x4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = fmaxf(0.0f, in.x);
        out.y = fmaxf(0.0f, in.y);
        out.z = fmaxf(0.0f, in.z);
        out.w = fmaxf(0.0f, in.w);
        *reinterpret_cast<float4*>(output + idx) = out;
    } else {
        for (int i = 0; i < 4 && idx + i < N; ++i) {
            output[idx + i] = fmaxf(0.0f, input[idx + i]);
        }
    }
}

// ---- Leaky ReLU: x > 0 ? x : alpha * x ----
__global__ void leaky_relu_f32_kernel(const float* input, float* output, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : alpha * x;
    }
}

__global__ void leaky_relu_f32x4_kernel(const float* input, float* output, int N, float alpha) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = in.x > 0.0f ? in.x : alpha * in.x;
        out.y = in.y > 0.0f ? in.y : alpha * in.y;
        out.z = in.z > 0.0f ? in.z : alpha * in.z;
        out.w = in.w > 0.0f ? in.w : alpha * in.w;
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ---- GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) ----
__device__ __forceinline__ float gelu_func(float x) {
    const float c = 0.7978845608f; // sqrt(2/pi)
    const float k = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(c * (x + k * x * x * x)));
}

__global__ void gelu_f32_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = gelu_func(input[idx]);
    }
}

__global__ void gelu_f32x4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = gelu_func(in.x);
        out.y = gelu_func(in.y);
        out.z = gelu_func(in.z);
        out.w = gelu_func(in.w);
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ---- Sigmoid: 1 / (1 + exp(-x)) ----
__global__ void sigmoid_f32_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void sigmoid_f32x4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = 1.0f / (1.0f + expf(-in.x));
        out.y = 1.0f / (1.0f + expf(-in.y));
        out.z = 1.0f / (1.0f + expf(-in.z));
        out.w = 1.0f / (1.0f + expf(-in.w));
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ---- Tanh ----
__global__ void tanh_f32_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void tanh_f32x4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ---- ELU: x > 0 ? x : alpha * (exp(x) - 1) ----
__global__ void elu_f32_kernel(const float* input, float* output, int N, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float x = input[idx];
        output[idx] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
}

__global__ void elu_f32x4_kernel(const float* input, float* output, int N, float alpha) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = in.x > 0.0f ? in.x : alpha * (expf(in.x) - 1.0f);
        out.y = in.y > 0.0f ? in.y : alpha * (expf(in.y) - 1.0f);
        out.z = in.z > 0.0f ? in.z : alpha * (expf(in.z) - 1.0f);
        out.w = in.w > 0.0f ? in.w : alpha * (expf(in.w) - 1.0f);
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ---- SELU: lambda * (x > 0 ? x : alpha * (exp(x) - 1)) ----
// lambda = 1.0507, alpha = 1.67326
__device__ __forceinline__ float selu_func(float x) {
    const float lambda_val = 1.0507009873554804934193349852946f;
    const float alpha_val = 1.6732632423543772848170429916717f;
    return lambda_val * (x > 0.0f ? x : alpha_val * (expf(x) - 1.0f));
}

__global__ void selu_f32_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = selu_func(input[idx]);
    }
}

__global__ void selu_f32x4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = selu_func(in.x);
        out.y = selu_func(in.y);
        out.z = selu_func(in.z);
        out.w = selu_func(in.w);
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ---- Softplus: log(1 + exp(x)) ----
__device__ __forceinline__ float softplus_func(float x) {
    // 数值稳定: 当x很大时直接返回x
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return logf(1.0f + expf(x));
}

__global__ void softplus_f32_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = softplus_func(input[idx]);
    }
}

__global__ void softplus_f32x4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = softplus_func(in.x);
        out.y = softplus_func(in.y);
        out.z = softplus_func(in.z);
        out.w = softplus_func(in.w);
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ---- Swish: x * sigmoid(x) ----
__device__ __forceinline__ float swish_func(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swish_f32_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = swish_func(input[idx]);
    }
}

__global__ void swish_f32x4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = swish_func(in.x);
        out.y = swish_func(in.y);
        out.z = swish_func(in.z);
        out.w = swish_func(in.w);
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ---- Hard Sigmoid: max(0, min(1, (x + 3) / 6)) ----
__device__ __forceinline__ float hard_sigmoid_func(float x) {
    float val = (x + 3.0f) / 6.0f;
    return fminf(1.0f, fmaxf(0.0f, val));
}

__global__ void hard_sigmoid_f32_kernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = hard_sigmoid_func(input[idx]);
    }
}

__global__ void hard_sigmoid_f32x4_kernel(const float* input, float* output, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = hard_sigmoid_func(in.x);
        out.y = hard_sigmoid_func(in.y);
        out.z = hard_sigmoid_func(in.z);
        out.w = hard_sigmoid_func(in.w);
        *reinterpret_cast<float4*>(output + idx) = out;
    }
}

// ==================== Torch Bindings ====================

// Helper macro for elementwise ops
#define LAUNCH_ELEMENTWISE(kernel, input, N) \
    auto output = torch::empty_like(input); \
    int block = 256; \
    int grid = CEIL(N, block); \
    kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N); \
    return output;

#define LAUNCH_ELEMENTWISE_X4(kernel, input, N) \
    auto output = torch::empty_like(input); \
    int block = 256; \
    int grid = CEIL(CEIL(N, 4), block); \
    kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N); \
    return output;

// ReLU
torch::Tensor torch_relu_f32(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE(relu_f32_kernel, input, N);
}
torch::Tensor torch_relu_f32x4(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE_X4(relu_f32x4_kernel, input, N);
}

// Leaky ReLU
torch::Tensor torch_leaky_relu_f32(torch::Tensor input, float alpha) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(N, block);
    leaky_relu_f32_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, alpha);
    return output;
}
torch::Tensor torch_leaky_relu_f32x4(torch::Tensor input, float alpha) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(CEIL(N, 4), block);
    leaky_relu_f32x4_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, alpha);
    return output;
}

// GELU
torch::Tensor torch_gelu_f32(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE(gelu_f32_kernel, input, N);
}
torch::Tensor torch_gelu_f32x4(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE_X4(gelu_f32x4_kernel, input, N);
}

// Sigmoid
torch::Tensor torch_sigmoid_f32(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE(sigmoid_f32_kernel, input, N);
}
torch::Tensor torch_sigmoid_f32x4(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE_X4(sigmoid_f32x4_kernel, input, N);
}

// Tanh
torch::Tensor torch_tanh_f32(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE(tanh_f32_kernel, input, N);
}
torch::Tensor torch_tanh_f32x4(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE_X4(tanh_f32x4_kernel, input, N);
}

// ELU
torch::Tensor torch_elu_f32(torch::Tensor input, float alpha) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(N, block);
    elu_f32_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, alpha);
    return output;
}
torch::Tensor torch_elu_f32x4(torch::Tensor input, float alpha) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(CEIL(N, 4), block);
    elu_f32x4_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, alpha);
    return output;
}

// SELU
torch::Tensor torch_selu_f32(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE(selu_f32_kernel, input, N);
}
torch::Tensor torch_selu_f32x4(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE_X4(selu_f32x4_kernel, input, N);
}

// Softplus
torch::Tensor torch_softplus_f32(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE(softplus_f32_kernel, input, N);
}
torch::Tensor torch_softplus_f32x4(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE_X4(softplus_f32x4_kernel, input, N);
}

// Swish
torch::Tensor torch_swish_f32(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE(swish_f32_kernel, input, N);
}
torch::Tensor torch_swish_f32x4(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE_X4(swish_f32x4_kernel, input, N);
}

// Hard Sigmoid
torch::Tensor torch_hard_sigmoid_f32(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE(hard_sigmoid_f32_kernel, input, N);
}
torch::Tensor torch_hard_sigmoid_f32x4(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    LAUNCH_ELEMENTWISE_X4(hard_sigmoid_f32x4_kernel, input, N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_relu_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_relu_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_leaky_relu_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_leaky_relu_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_gelu_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_gelu_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_sigmoid_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_sigmoid_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_tanh_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_tanh_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_elu_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_elu_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_selu_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_selu_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_softplus_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_softplus_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_swish_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_swish_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(torch_hard_sigmoid_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_hard_sigmoid_f32x4)
}
