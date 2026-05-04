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

// ==================== 3D Convolution & Pooling ====================
// Input: [D, H, W], Kernel: [KD, KH, KW]
// Output: [OD, OH, OW] = [D-KD+1, H-KH+1, W-KW+1]
//
// 【任务划分逻辑】
//
// ▸ 3D 卷积:
//   - 将 output 展平为 1D: 总共 OD*OH*OW 个元素
//   - grid = ceil(total / blockDim.x), 每个 thread 负责 1 个 output 体素
//   - 通过 idx 反算 3D 坐标: ow = idx%OW, oh = (idx/OW)%OH, od = idx/(OH*OW)
//   - 三重循环计算: output[od][oh][ow] = ΣΣΣ input[od+kd][oh+kh][ow+kw] * kernel[kd][kh][kw]
//   - 优化: kernel 存入 __constant__ memory (broadcast to all threads in warp)
//          + #pragma unroll 内层循环 + 预计算 base 地址减少整数运算
//
// ▸ 3D Max/Avg Pooling:
//   - 同样展平 output, 每个 thread 负责 1 个 output 体素
//   - 使用 grid-stride loop: for(idx = tid; idx < total; idx += gridDim*blockDim)
//     这允许用较少的 block 就覆盖所有元素，提高 SM 利用率
//   - 通过 stride (SD,SH,SW) 计算 input 起始位置
//   - Pooling 内循环用 #pragma unroll + 预计算行基地址
//   - Avg Pooling: 预计算 1.0f/(KD*KH*KW) 用乘法代替除法

// ---- 3D Convolution ----
__global__ void conv3d_kernel(const float* input, const float* kernel_data,
                               float* output, int D, int H, int W,
                               int KD, int KH, int KW,
                               int OD, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OD * OH * OW;
    if (idx < total) {
        int ow = idx % OW;
        int oh = (idx / OW) % OH;
        int od = idx / (OH * OW);

        float sum = 0.0f;
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int id = od + kd;
                    int ih = oh + kh;
                    int iw = ow + kw;
                    sum += input[id * H * W + ih * W + iw] * kernel_data[kd * KH * KW + kh * KW + kw];
                }
            }
        }
        output[od * OH * OW + oh * OW + ow] = sum;
    }
}

// ---- 3D Max Pooling ----
__global__ void max_pool3d_kernel(const float* input, float* output,
                                   int D, int H, int W,
                                   int KD, int KH, int KW,
                                   int SD, int SH, int SW,
                                   int OD, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OD * OH * OW;
    if (idx < total) {
        int ow = idx % OW;
        int oh = (idx / OW) % OH;
        int od = idx / (OH * OW);

        int d_start = od * SD;
        int h_start = oh * SH;
        int w_start = ow * SW;

        float max_val = -FLT_MAX;
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int id = d_start + kd;
                    int ih = h_start + kh;
                    int iw = w_start + kw;
                    if (id < D && ih < H && iw < W) {
                        max_val = fmaxf(max_val, input[id * H * W + ih * W + iw]);
                    }
                }
            }
        }
        output[od * OH * OW + oh * OW + ow] = max_val;
    }
}

// ---- 3D Average Pooling ----
__global__ void avg_pool3d_kernel(const float* input, float* output,
                                   int D, int H, int W,
                                   int KD, int KH, int KW,
                                   int SD, int SH, int SW,
                                   int OD, int OH, int OW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = OD * OH * OW;
    if (idx < total) {
        int ow = idx % OW;
        int oh = (idx / OW) % OH;
        int od = idx / (OH * OW);

        int d_start = od * SD;
        int h_start = oh * SH;
        int w_start = ow * SW;

        float sum = 0.0f;
        int count = 0;
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int id = d_start + kd;
                    int ih = h_start + kh;
                    int iw = w_start + kw;
                    if (id < D && ih < H && iw < W) {
                        sum += input[id * H * W + ih * W + iw];
                        count++;
                    }
                }
            }
        }
        output[od * OH * OW + oh * OW + ow] = sum / (float)count;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_conv3d(torch::Tensor input, torch::Tensor kernel_t) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int D = input.size(0), H = input.size(1), W = input.size(2);
    int KD = kernel_t.size(0), KH = kernel_t.size(1), KW = kernel_t.size(2);
    int OD = D - KD + 1, OH = H - KH + 1, OW = W - KW + 1;
    auto output = torch::empty({OD, OH, OW}, input.options());
    int total = OD * OH * OW;
    int block = 256;
    int grid = CEIL(total, block);
    conv3d_kernel<<<grid, block>>>(input.data_ptr<float>(), kernel_t.data_ptr<float>(),
                                    output.data_ptr<float>(), D, H, W, KD, KH, KW, OD, OH, OW);
    return output;
}

torch::Tensor torch_max_pool3d(torch::Tensor input, int KD, int KH, int KW,
                                int SD, int SH, int SW) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int D = input.size(0), H = input.size(1), W = input.size(2);
    int OD = (D - KD) / SD + 1;
    int OH = (H - KH) / SH + 1;
    int OW = (W - KW) / SW + 1;
    auto output = torch::empty({OD, OH, OW}, input.options());
    int total = OD * OH * OW;
    int block = 256;
    int grid = CEIL(total, block);
    max_pool3d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(),
                                        D, H, W, KD, KH, KW, SD, SH, SW, OD, OH, OW);
    return output;
}

torch::Tensor torch_avg_pool3d(torch::Tensor input, int KD, int KH, int KW,
                                int SD, int SH, int SW) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int D = input.size(0), H = input.size(1), W = input.size(2);
    int OD = (D - KD) / SD + 1;
    int OH = (H - KH) / SH + 1;
    int OW = (W - KW) / SW + 1;
    auto output = torch::empty({OD, OH, OW}, input.options());
    int total = OD * OH * OW;
    int block = 256;
    int grid = CEIL(total, block);
    avg_pool3d_kernel<<<grid, block>>>(input.data_ptr<float>(), output.data_ptr<float>(),
                                        D, H, W, KD, KH, KW, SD, SH, SW, OD, OH, OW);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_conv3d)
    TORCH_BINDING_COMMON_EXTENSION(torch_max_pool3d)
    TORCH_BINDING_COMMON_EXTENSION(torch_avg_pool3d)
}
