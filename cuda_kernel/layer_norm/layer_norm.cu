
#include <algorithm>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>

// 语义：
// 输入：x为(N,K)的矩阵  （N 是 batch×seq_len，K 是 hidden_size）
// 输出：对于每一个元素，按照每一行（即每个样本的 hidden 向量）独立做归一化
// 层归一化（Layer Normalization）——对输入张量（ N*K的矩阵 ）的每一行独立进行归一化。
// 归一化在隐藏维度（K）上进行，每个样本（即每行）使用自己的均值和方差。
//
// 数学公式：
//   LayerNorm(x_i) = γ * (x_i - μ_i) / sqrt(σ_i² + ε) + β
//
// 其中，对于第 i 个样本（即 [N x K] 张量中的第 i 行）：
//   μ_i = (1 / K) * Σ_{j=1}^{K} x_{ij}          // 该行的均值
//   σ_i² = (1 / K) * Σ_{j=1}^{K} (x_{ij} - μ_i)² // 该行的方差
//   γ, β 为可学习的仿射变换参数（缩放和平移）
//   ε = 1e-5 为防止除零的小常数
//
// 注意：本 kernel 中 γ 和 β 被简化为标量（g 和 b），
//       实际应用中通常为长度为 K 的向量（逐通道可学习）。

//我们让一个block负责一行的归一化，一个thread负责一行中的一个或者多个元素，那么其实本质上和reduce的思想很像：
//就是每个thread需要使用这个block集体出力算出来的结果（μ和σ²）
//每个thread传入一个值val，然后这个block的所有thread集体出力，最后每个thread会拿到这个求和之后的结果
//使用两层规约：第一层是warp里面用shuffle去做，第二层是使用shared_memory

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_THREADS = 256;

// FP32
// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// 这个 block_reduce_sum 函数的本质就是：
// 每个线程传入一个thread自己处理好的标量值（比如从 float4 中 reduce 出来的单个 float），或者就是一个thread自己只负责一个元素
// 然后通过两阶段归约（warp 内 shuffle + 跨 warp借助shared memory然后用一个represent warp去做warp内shuffle）求出整个 block 的总和，
// 并广播给所有线程，最后每个thread都能拿到这个 block 协作算出来的那个值
// Block-wide sum: reduce within warp, then reduce warp sums and broadcast.
// grid 1D block 1D, grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum(float val) {
  // always <= 32 warps per block (limited by 1024 threads per block)
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  __shared__ float shared[NUM_WARPS];  // Use non-static shared memory for flexibility

  val = warp_reduce_sum_f32<WARP_SIZE>(val);
  if (lane == 0)
    shared[warp] = val;
  __syncthreads();
  val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
  val = warp_reduce_sum_f32<NUM_WARPS>(val);
  if (warp == 0 && lane == 0) {
   shared[0] = val; // broadcast block sum to all threads
 }
  __syncthreads();
  return shared[0];
}


// Single-block-per-row version: one thread per element (K must be <= blockDim.x).
template<int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float* x, float* y, float g, float b, int N, int K) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= N || col >= K) return;

    float val = x[row * K + col];

    float sum = block_reduce_sum<NUM_THREADS>(val);        // 每个thread获得相同的值

    float mean = sum / K;              
    float var_val = (val - mean) * (val - mean);
    float var_sum = block_reduce_sum<NUM_THREADS>(var_val);
    float inv_std = rsqrtf(var_sum / K + 1e-5f);

    y[row * K + col] = (val - mean) * inv_std * g + b;
}


#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

// Single-block-per-row version: one thread processes 4 floats.
template<int NUM_THREADS = 256/4>  //相比于上面的每个thread处理一个float，这里每个thread处理4个float，因此NUM_THREADS需要除以4
__global__ void layer_norm_float4_kernel(float* x, float* y, float g, float b, int N, int K) {
    int row=blockIdx.x;
    int col_start=threadIdx.x*4;
    if(row>=N || col_start>=K){
        return;
    }

    //每个thread处理4个分量，因为四个分量是独立的，所以其实代码上没有什么本质区别
    float4 v=FLOAT4(x[row*K+col_start]);
    float sum_of_thread=v.x+v.y+v.z+v.w;
    
    float sum_of_block=block_reduce_sum<NUM_THREADS>(sum_of_thread);

    float mean=sum_of_block/K;

    float var_sum = 
        (v.x - mean) * (v.x - mean) +
        (v.y - mean) * (v.y - mean) +
        (v.z - mean) * (v.z - mean) +
        (v.w - mean) * (v.w - mean);

    float total_var = block_reduce_sum<NUM_THREADS>(var_sum);

    float inv_std = rsqrtf(total_var / K + 1e-5f);  

    float4 out;
    out.x = (v.x - mean) * inv_std * g + b;
    out.y = (v.y - mean) * inv_std * g + b;
    out.z = (v.z - mean) * inv_std * g + b;
    out.w = (v.w - mean) * inv_std * g + b;

    FLOAT4(y[row * K + col_start]) = out;

}


// ==================== Torch bindings ====================
#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
  if (((T).options().dtype() != (th_type))) {                \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);    \
  }

// x: [N, K], g: scalar, b: scalar -> y: [N, K]
torch::Tensor torch_layer_norm_f32(torch::Tensor x, float g, float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    int N = x.size(0);
    int K = x.size(1);
    auto y = torch::empty_like(x);
    dim3 grid(N);
    dim3 block(K);  // K must be <= 1024
    layer_norm_f32_kernel<256><<<grid, block>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), g, b, N, K);
    return y;
}

torch::Tensor torch_layer_norm_float4(torch::Tensor x, float g, float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    int N = x.size(0);
    int K = x.size(1);
    auto y = torch::empty_like(x);
    dim3 grid(N);
    dim3 block(K / 4);
    layer_norm_float4_kernel<256/4><<<grid, block>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), g, b, N, K);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_layer_norm_f32)
    TORCH_BINDING_COMMON_EXTENSION(torch_layer_norm_float4)
}
