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

// ==================== Graphics Kernels ====================
// Grayscale, Image Histogram, Image Thresholding, Box Blur, Edge Detection
//
// 【任务划分逻辑】
//
// ▸ Grayscale (逐像素独立):
//   - 每个 thread 负责 1 个像素: 读 3 个通道值，加权求和写 1 个灰度值
//   - grid = ceil(H*W / blockDim.x)
//   - 纯 bandwidth-bound，类似 elementwise
//
// ▸ Histogram (全局 reduce + 冲突):
//   - 使用 shared memory 局部直方图 local_hist[256] (int)
//   - 每个 block 先在 smem 中累积 (atomicAdd on smem，比 global atomic 快 10x+)
//   - __syncthreads() 后，由前 256 个 thread 将 local_hist atomicAdd 到 global histogram
//   - 两级 atomic: smem 级 (低竞争) + global 级 (block 间竞争)
//
// ▸ Thresholding (逐像素独立):
//   - float4 向量化: 每个 thread 处理 4 个像素
//   - output = (input > threshold) ? 1.0 : 0.0
//   - 纯 bandwidth-bound
//
// ▸ Box Blur (Stencil 计算):
//   - 每个 thread 负责 1 个 output 像素
//   - 遍历 K×K 邻域求平均 (边界处只对有效邻居求平均)
//   - 当前依赖 L1/L2 cache 缓存相邻 thread 的重叠读取
//
// ▸ Edge Detection (Sobel 算子):
//   - 每个 thread 负责 1 个 output 像素
//   - 读取 3×3 邻域，分别用 Gx/Gy 卷积核计算梯度
//   - output = sqrt(Gx² + Gy²)
//   - 边界像素直接输出 0

// ---- Grayscale Conversion ----
// RGB to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
// input: [H, W, 3], output: [H, W]
__global__ void grayscale_kernel(const float* input, float* output, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float r = input[idx * 3];
        float g = input[idx * 3 + 1];
        float b = input[idx * 3 + 2];
        output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// ---- Image Histogram (256 bins, input values 0-255 as int) ----
// input: [N] int values, output: [256] histogram
__global__ void histogram_kernel(const int* input, int* histogram, int N) {
    __shared__ int local_hist[256];

    int tid = threadIdx.x;
    // Initialize shared histogram
    if (tid < 256) local_hist[tid] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + tid;
    if (idx < N) {
        int val = input[idx];
        if (val >= 0 && val < 256) {
            atomicAdd(&local_hist[val], 1);
        }
    }
    __syncthreads();

    // Write local histogram to global
    if (tid < 256) {
        atomicAdd(&histogram[tid], local_hist[tid]);
    }
}

// ---- Image Thresholding ----
// output[i] = (input[i] > threshold) ? 1.0 : 0.0
__global__ void threshold_kernel(const float* input, float* output, int N, float threshold) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 in = *reinterpret_cast<const float4*>(input + idx);
        float4 out;
        out.x = (in.x > threshold) ? 1.0f : 0.0f;
        out.y = (in.y > threshold) ? 1.0f : 0.0f;
        out.z = (in.z > threshold) ? 1.0f : 0.0f;
        out.w = (in.w > threshold) ? 1.0f : 0.0f;
        *reinterpret_cast<float4*>(output + idx) = out;
    } else {
        for (int i = 0; i < 4 && idx + i < N; ++i) {
            output[idx + i] = (input[idx + i] > threshold) ? 1.0f : 0.0f;
        }
    }
}

// ---- Box Blur (3x3) ----
// output[i][j] = average of 3x3 neighborhood
__global__ void box_blur_kernel(const float* input, float* output, int H, int W, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (idx < total) {
        int row = idx / W;
        int col = idx % W;
        int half_k = K / 2;

        float sum = 0.0f;
        int count = 0;
        for (int dr = -half_k; dr <= half_k; ++dr) {
            for (int dc = -half_k; dc <= half_k; ++dc) {
                int r = row + dr;
                int c = col + dc;
                if (r >= 0 && r < H && c >= 0 && c < W) {
                    sum += input[r * W + c];
                    count++;
                }
            }
        }
        output[idx] = sum / (float)count;
    }
}

// ---- Edge Detection (Sobel) ----
// Gx = [[-1,0,1],[-2,0,2],[-1,0,1]]
// Gy = [[-1,-2,-1],[0,0,0],[1,2,1]]
// output = sqrt(Gx^2 + Gy^2)
__global__ void edge_detection_kernel(const float* input, float* output, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * W;
    if (idx < total) {
        int row = idx / W;
        int col = idx % W;

        if (row == 0 || row == H - 1 || col == 0 || col == W - 1) {
            output[idx] = 0.0f;
            return;
        }

        // Sobel Gx
        float gx = -input[(row-1)*W + (col-1)] + input[(row-1)*W + (col+1)]
                   -2.0f*input[row*W + (col-1)] + 2.0f*input[row*W + (col+1)]
                   -input[(row+1)*W + (col-1)] + input[(row+1)*W + (col+1)];

        // Sobel Gy
        float gy = -input[(row-1)*W + (col-1)] - 2.0f*input[(row-1)*W + col] - input[(row-1)*W + (col+1)]
                   +input[(row+1)*W + (col-1)] + 2.0f*input[(row+1)*W + col] + input[(row+1)*W + (col+1)];

        output[idx] = sqrtf(gx * gx + gy * gy);
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_grayscale(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int H = input.size(0), W = input.size(1);
    int total = H * W;
    auto output = torch::empty({H, W}, input.options());
    int block = 256;
    int grid = CEIL(total, block);
    grayscale_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), total);
    return output;
}

torch::Tensor torch_histogram(torch::Tensor input) {
    // input should be int32
    int N = input.numel();
    auto output = torch::zeros({256}, torch::dtype(torch::kInt32).device(input.device()));
    int block = 256;
    int grid = CEIL(N, block);
    histogram_kernel<<<grid, block>>>(input.data_ptr<int>(), output.data_ptr<int>(), N);
    return output;
}

torch::Tensor torch_threshold(torch::Tensor input, float threshold) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int N = input.numel();
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(CEIL(N, 4), block);
    threshold_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), N, threshold);
    return output;
}

torch::Tensor torch_box_blur(torch::Tensor input, int K) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int H = input.size(0), W = input.size(1);
    int total = H * W;
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(total, block);
    box_blur_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), H, W, K);
    return output;
}

torch::Tensor torch_edge_detection(torch::Tensor input) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int H = input.size(0), W = input.size(1);
    int total = H * W;
    auto output = torch::empty_like(input);
    int block = 256;
    int grid = CEIL(total, block);
    edge_detection_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(), H, W);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_grayscale)
    TORCH_BINDING_COMMON_EXTENSION(torch_histogram)
    TORCH_BINDING_COMMON_EXTENSION(torch_threshold)
    TORCH_BINDING_COMMON_EXTENSION(torch_box_blur)
    TORCH_BINDING_COMMON_EXTENSION(torch_edge_detection)
}
