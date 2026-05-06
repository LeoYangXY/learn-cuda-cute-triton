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

// ==================== 2D Convolution Kernel ====================
// input: [H, W], kernel: [KH, KW], output: [OH, OW] where OH=H-KH+1, OW=W-KW+1
//
// 【任务划分逻辑】
// 2D卷积: output[oh][ow] = ΣΣ input[oh+kh][ow+kw] * kernel[kh][kw]
//
// ▸ Naive 版本:
//   - 将 output 展平为 1D, 总共 OH*OW 个元素
//   - grid = ceil(OH*OW / blockDim.x), 每个 thread 负责 1 个 output 元素
//   - 通过 idx / OW 和 idx % OW 反算出 (oh, ow) 坐标
//   - 双重循环访问 input 的 KH×KW 邻域
//   - 问题: input 数据局部性差，相邻 thread 访问的 input 区域大量重叠
//
// ▸ Tiled Shared Memory 版本:
//   - 使用 2D thread block: dim3(TILE_SIZE, TILE_SIZE) = (16, 16)
//   - 每个 block 输出一个 16×16 的 output tile
//   - 需要加载 (TILE_SIZE+KH-1) × (TILE_SIZE+KW-1) 的 input tile 到 smem
//   - 协作加载: 每个 thread 可能加载多个元素 (因为 input tile > output tile)
//     用双层循环 r += TILE_SIZE, c += TILE_SIZE 确保覆盖整个 input tile
//   - 计算时: output[ty][tx] = ΣΣ smem[ty+kh][tx+kw] * kernel[kh][kw]
//   - 优势: input 数据从 global memory 只读一次到 smem，被 tile 内多个 thread 复用
//   - grid = (ceil(OW/TILE), ceil(OH/TILE))

// Naive: 每个thread计算一个output元素
__global__ void conv2d_naive_kernel(const float* input, const float* kernel_data,
                                     float* output, int H, int W, int KH, int KW,
                                     int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < OH * OW) {
        int oh = idx / OW;
        int ow = idx % OW;

        float sum = 0.0f;
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                sum += input[(oh + kh) * W + (ow + kw)] * kernel_data[kh * KW + kw];
            }
        }
        output[oh * OW + ow] = sum;
    }
}

// Tiled with shared memory
// 每个block处理一个output tile, 加载input tile到shared memory
#define TILE_SIZE 16

__global__ void conv2d_shared_kernel(const float* input, const float* kernel_data,
                                      float* output, int H, int W, int KH, int KW,
                                      int OH, int OW) {
    // 每个block的输出tile大小为 TILE_SIZE x TILE_SIZE
    // 输入tile大小为 (TILE_SIZE + KH - 1) x (TILE_SIZE + KW - 1)
    extern __shared__ float smem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int out_row = blockIdx.y * TILE_SIZE + ty;
    int out_col = blockIdx.x * TILE_SIZE + tx;

    int in_tile_h = TILE_SIZE + KH - 1;
    int in_tile_w = TILE_SIZE + KW - 1;

    // Cooperatively load input tile into shared memory
    int in_start_row = blockIdx.y * TILE_SIZE;
    int in_start_col = blockIdx.x * TILE_SIZE;

    // Each thread may need to load multiple elements
    for (int r = ty; r < in_tile_h; r += TILE_SIZE) {
        for (int c = tx; c < in_tile_w; c += TILE_SIZE) {
            int global_r = in_start_row + r;
            int global_c = in_start_col + c;
            if (global_r < H && global_c < W) {
                smem[r * in_tile_w + c] = input[global_r * W + global_c];
            } else {
                smem[r * in_tile_w + c] = 0.0f;
            }
        }
    }
    __syncthreads();

    if (out_row < OH && out_col < OW) {
        float sum = 0.0f;
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                sum += smem[(ty + kh) * in_tile_w + (tx + kw)] * kernel_data[kh * KW + kw];
            }
        }
        output[out_row * OW + out_col] = sum;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_conv2d_naive(torch::Tensor input, torch::Tensor kernel_t) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int H = input.size(0), W = input.size(1);
    int KH = kernel_t.size(0), KW = kernel_t.size(1);
    int OH = H - KH + 1, OW = W - KW + 1;
    auto output = torch::empty({OH, OW}, input.options());
    int total = OH * OW;
    int block = 256;
    int grid = CEIL(total, block);
    conv2d_naive_kernel<<<grid, block>>>(
        input.data_ptr<float>(), kernel_t.data_ptr<float>(),
        output.data_ptr<float>(), H, W, KH, KW, OH, OW);
    return output;
}

torch::Tensor torch_conv2d_shared(torch::Tensor input, torch::Tensor kernel_t) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int H = input.size(0), W = input.size(1);
    int KH = kernel_t.size(0), KW = kernel_t.size(1);
    int OH = H - KH + 1, OW = W - KW + 1;
    auto output = torch::empty({OH, OW}, input.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(CEIL(OW, TILE_SIZE), CEIL(OH, TILE_SIZE));
    int in_tile_h = TILE_SIZE + KH - 1;
    int in_tile_w = TILE_SIZE + KW - 1;
    size_t smem_size = in_tile_h * in_tile_w * sizeof(float);

    conv2d_shared_kernel<<<grid, block, smem_size>>>(
        input.data_ptr<float>(), kernel_t.data_ptr<float>(),
        output.data_ptr<float>(), H, W, KH, KW, OH, OW);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_conv2d_naive)
    TORCH_BINDING_COMMON_EXTENSION(torch_conv2d_shared)
}
