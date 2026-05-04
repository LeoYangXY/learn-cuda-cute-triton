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

// ==================== Matrix Operations ====================
// Matrix Scalar Multiplication, Diagonal Matrix Multiplication
// Layer Normalization, Batch Normalization
//
// 【任务划分逻辑】
//
// ▸ Matrix Scalar Mul (C = α * A):
//   - 纯 elementwise, float4 向量化: 每个 thread 处理 4 个元素
//   - grid = ceil(N/4 / blockDim.x)
//   - bandwidth-bound, 向量化是唯一有意义的优化
//
// ▸ Diagonal Matrix Mul (C = diag(d) @ A):
//   - C[i][j] = d[i] * A[i][j], 等价于每行乘以一个标量
//   - grid = M (行数), 每行一个 block
//   - 每个 thread 用 thread-stride loop 处理一行的多个元素
//   - thread 0 和其他 thread 读同一个 d[row] → 编译器会放在 register 中复用
//
// ▸ Layer Normalization:
//   - grid = M, 每行一个 block (同 norms 的 per-row 模式)
//   - Pass 1: reduce sum → mean
//   - Pass 2: reduce sum of (x-mean)² → variance
//   - Pass 3: normalize + affine: output[i] = (x[i]-mean) * rsqrt(var+eps) * γ[i] + β[i]
//   - 3 次 global memory 读 (可优化为 2 次，用 Welford 在线算法)
//
// ▸ Batch Normalization (inference):
//   - 使用 running_mean/running_var (预计算好的统计量)
//   - 纯 elementwise: output[i][c] = (x[i][c] - mean[c]) * rsqrt(var[c]+eps) * w[c] + b[c]
//   - grid = ceil(M*C / blockDim.x), 每个 thread 1 个元素
//   - 通过 idx % C 得到 channel 索引来读取对应的 mean/var/weight/bias

// ---- warp reduce sum ----
template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ---- Matrix Scalar Multiplication: C = alpha * A ----
__global__ void mat_scalar_mul_kernel(const float* A, float* C, float alpha, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 a = *reinterpret_cast<const float4*>(A + idx);
        float4 c;
        c.x = alpha * a.x;
        c.y = alpha * a.y;
        c.z = alpha * a.z;
        c.w = alpha * a.w;
        *reinterpret_cast<float4*>(C + idx) = c;
    } else {
        for (int i = 0; i < 4 && idx + i < N; ++i) {
            C[idx + i] = alpha * A[idx + i];
        }
    }
}

// ---- Diagonal Matrix Multiplication: C = diag(d) @ A ----
// d: [M], A: [M, N], C: [M, N]
// C[i][j] = d[i] * A[i][j]
__global__ void diag_matmul_kernel(const float* d, const float* A, float* C,
                                    int M, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float scale = d[row];

    for (int j = tid; j < N; j += blockDim.x) {
        C[row * N + j] = scale * A[row * N + j];
    }
}

// ---- Layer Normalization ----
// input: [M, N], output: [M, N]
// output[i][j] = (input[i][j] - mean_i) / sqrt(var_i + eps) * gamma[j] + beta[j]
__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta,
                                   float* output, int N, float eps) {
    __shared__ float smem[32];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    const float* row_in = input + row * N;
    float* row_out = output + row * N;

    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        sum += row_in[i];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        sum = (lane < nw) ? smem[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    __syncthreads();
    if (tid == 0) smem[0] = sum;
    __syncthreads();
    float mean = smem[0] / (float)N;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = row_in[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = warp_reduce_sum(var_sum);
    if (lane == 0) smem[warp_id] = var_sum;
    __syncthreads();
    if (warp_id == 0) {
        int nw = (blockDim.x + 31) / 32;
        var_sum = (lane < nw) ? smem[lane] : 0.0f;
        var_sum = warp_reduce_sum(var_sum);
    }
    __syncthreads();
    if (tid == 0) smem[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(smem[0] / (float)N + eps);

    // Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// ---- Batch Normalization (inference mode) ----
// input: [N, C, H, W] flattened as [N*H*W, C] -> per-channel normalization
// For simplicity: input: [M, C], normalize along dim=0 for each channel
// output[i][c] = (input[i][c] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
__global__ void batch_norm_kernel(const float* input, const float* running_mean,
                                   const float* running_var, const float* weight,
                                   const float* bias, float* output,
                                   int M, int C, float eps) {
    // Each thread handles one element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * C;
    if (idx < total) {
        int c = idx % C;
        float x = input[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        output[idx] = (x - mean) * inv_std * weight[c] + bias[c];
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_mat_scalar_mul(torch::Tensor A, float alpha) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int N = A.numel();
    auto C = torch::empty_like(A);
    int block = 256;
    int grid = CEIL(CEIL(N, 4), block);
    mat_scalar_mul_kernel<<<grid, block>>>(A.data_ptr<float>(), C.data_ptr<float>(), alpha, N);
    return C;
}

torch::Tensor torch_diag_matmul(torch::Tensor d, torch::Tensor A) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), N = A.size(1);
    auto C = torch::empty_like(A);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    diag_matmul_kernel<<<M, block>>>(d.data_ptr<float>(), A.data_ptr<float>(),
                                      C.data_ptr<float>(), M, N);
    return C;
}

torch::Tensor torch_layer_norm(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), N = input.size(1);
    auto output = torch::empty_like(input);
    int block = min(1024, N);
    block = ((block + 31) / 32) * 32;
    layer_norm_kernel<<<M, block>>>(input.data_ptr<float>(), gamma.data_ptr<float>(),
                                     beta.data_ptr<float>(), output.data_ptr<float>(), N, eps);
    return output;
}

torch::Tensor torch_batch_norm(torch::Tensor input, torch::Tensor running_mean,
                                torch::Tensor running_var, torch::Tensor weight,
                                torch::Tensor bias, float eps) {
    CHECK_TORCH_TENSOR_DTYPE(input, torch::kFloat32);
    int M = input.size(0), C = input.size(1);
    auto output = torch::empty_like(input);
    int total = M * C;
    int block = 256;
    int grid = CEIL(total, block);
    batch_norm_kernel<<<grid, block>>>(
        input.data_ptr<float>(), running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(), weight.data_ptr<float>(),
        bias.data_ptr<float>(), output.data_ptr<float>(), M, C, eps);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_mat_scalar_mul)
    TORCH_BINDING_COMMON_EXTENSION(torch_diag_matmul)
    TORCH_BINDING_COMMON_EXTENSION(torch_layer_norm)
    TORCH_BINDING_COMMON_EXTENSION(torch_batch_norm)
}
