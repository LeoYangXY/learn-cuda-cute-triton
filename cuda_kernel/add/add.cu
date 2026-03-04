#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <iostream>
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>


#define FLOAT4(a) *(float4*)(&(a))
#define CEIL(a,b) ((a+b-1)/(b))
#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};


//下面2个是针对fp32版本的
//按照一个thread负责一个元素的结果值：
__global__ void eltwise_add_scaler(float* a,float* b,float* c,int N){
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<=N){//现在每个thread算完自己的idx之后可能会越过N
        c[idx]=a[idx]+b[idx];
    }
}

//这里的尾块我们没有去处理，因为那个脏数据不影响
__global__ void eltwise_add_vector(float* a,float* b,float* c,int N){
    int start=(blockDim.x*blockIdx.x+threadIdx.x)*4;//每个thread负责start~start+3这段的4个数据
    float4 a_vec = *reinterpret_cast<float4*>(a + start);//float4是nvidia的内置类型，我们需要传入一个重新解释一段内存才可以
    float4 b_vec = *reinterpret_cast<float4*>(b + start);

    float4 c_vec;
    //由于float4没有为其重载加号，因此我们需要使用分量进行相加。
    //其实从这里可以看出：向量化主要是省了那个访存的时间：每个线程使用向量化的方式去访存；但是计算上仍然是scaler方式的计算
    //float4内置了.x  .y这些其帮助我们进行访问
    c_vec.x = a_vec.x + b_vec.x;
    c_vec.y = a_vec.y + b_vec.y;
    c_vec.z = a_vec.z + b_vec.z;
    c_vec.w = a_vec.w + b_vec.w;

    // 向量化存储
    *reinterpret_cast<float4*>(c + start) = c_vec;

}


//针对fp16
//标量化
__global__ void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}

//向量化访存加速，但是实际计算上并没有变快，因为我们每次仍然是使用_hadd这样的标量方式去计算的
__global__ void elementwise_add_f16_vec2(half *a, half *b, half *c, int N){
    int start=(blockDim.x*blockIdx.x+threadIdx.x)*2;
    half2 a_vec = *reinterpret_cast<half2*>(a + start);
    half2 b_vec = *reinterpret_cast<half2*>(b + start);
    half2 c_vec;
    c_vec.x = __hadd(a_vec.x, b_vec.x);
    c_vec.y = __hadd(a_vec.y, b_vec.y);
    *reinterpret_cast<half2*>(c + start) = c_vec;
}

//使用_hadd2这样子做向量化的计算进行加速
__global__ void elementwise_add_f16_vec2_4(half *a, half *b, half *c, int N){
    int start=(blockDim.x*blockIdx.x+threadIdx.x)*8;
    half2 a_vec0=*reinterpret_cast<half2*>(a + start);
    half2 a_vec1=*reinterpret_cast<half2*>(a + start+2);
    half2 a_vec2=*reinterpret_cast<half2*>(a + start+4);
    half2 a_vec3=*reinterpret_cast<half2*>(a + start+6);

    half2 b_vec0=*reinterpret_cast<half2*>(b + start);
    half2 b_vec1=*reinterpret_cast<half2*>(b + start+2);
    half2 b_vec2=*reinterpret_cast<half2*>(b + start+4);
    half2 b_vec3=*reinterpret_cast<half2*>(b + start+6);
    
    half2 c_vec0, c_vec1, c_vec2, c_vec3;
    c_vec0 = __hadd2(a_vec0, b_vec0);
    c_vec1 = __hadd2(a_vec1, b_vec1);
    c_vec2 = __hadd2(a_vec2, b_vec2);
    c_vec3 = __hadd2(a_vec3, b_vec3);
    
    *reinterpret_cast<half2*>(c + start) = c_vec0;
    *reinterpret_cast<half2*>(c + start+2) = c_vec1;
    *reinterpret_cast<half2*>(c + start+4) = c_vec2;
    *reinterpret_cast<half2*>(c + start+6) = c_vec3;

}

//使用循环展开的标识符去书写上面的代码:既保持高性能（避免运行时开销），又提升代码可读性和可维护性
__global__ void elementwise_add_f16_vec2_4_unroll(half *a, half *b, half *c, int N) {
    int start = (blockDim.x * blockIdx.x + threadIdx.x) * 8;

    half2 a_vec[4];
    half2 b_vec[4];
    half2 c_vec[4];

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a_vec[i] = *reinterpret_cast<half2*>(a + start + 2 * i);
        b_vec[i] = *reinterpret_cast<half2*>(b + start + 2 * i);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        c_vec[i] = __hadd2(a_vec[i], b_vec[i]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        *reinterpret_cast<half2*>(c + start + 2 * i) = c_vec[i];
    }
}


#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
__global__ void elementwise_add_f16_pack(half *a,half *b, half *c,int N){
    
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  // temporary register(memory), .local space in ptx, addressable
  half pack_a[8], pack_b[8], pack_c[8]; // 8x16 bits=128 bits.
  // reinterpret as float4 and load 128 bits in 1 memory issue.
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]); // load 128 bits
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]); // load 128 bits

#pragma unroll
  for (int i = 0; i < 8; i += 2) {
    //对于pack_c的填充:
    //我们先用HALF2(x) 这个个宏，把两个连续的 half 视为一个 half2 类型。
    //然后使用__hadd2，同时对两个 half 做加法
    HALF2(pack_c[i]) = __hadd2(HALF2(pack_a[i]), HALF2(pack_b[i]));
  }

  // 把pack_c的结果填充回真正的output
  LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);

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

// ----- FP32 wrappers -----
void torch_eltwise_add_f32_scalar(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32);
    int N = a.numel();
    int block = 256;
    int grid = CEIL(N, block);
    eltwise_add_scaler<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
}

void torch_eltwise_add_f32_vector(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32);
    int N = a.numel();
    int block = 256;
    int threads_needed = CEIL(N, 4);
    int grid = CEIL(threads_needed, block);
    eltwise_add_vector<<<grid, block>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
}

// ----- FP16 wrappers -----
void torch_eltwise_add_f16_scalar(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);
    int N = a.numel();
    int block = 256;
    int grid = CEIL(N, block);
    elementwise_add_f16_kernel<<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()), N);
}

void torch_eltwise_add_f16_vec2(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);
    int N = a.numel();
    int block = 256;
    int threads_needed = CEIL(N, 2);
    int grid = CEIL(threads_needed, block);
    elementwise_add_f16_vec2<<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()), N);
}

void torch_eltwise_add_f16_vec2_4(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);
    int N = a.numel();
    int block = 256;
    int threads_needed = CEIL(N, 8);
    int grid = CEIL(threads_needed, block);
    elementwise_add_f16_vec2_4<<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()), N);
}

void torch_eltwise_add_f16_vec2_4_unroll(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);
    int N = a.numel();
    int block = 256;
    int threads_needed = CEIL(N, 8);
    int grid = CEIL(threads_needed, block);
    elementwise_add_f16_vec2_4_unroll<<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()), N);
}

void torch_eltwise_add_f16_pack(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf);
    int N = a.numel();
    int block = 256;
    int threads_needed = CEIL(N, 8);
    int grid = CEIL(threads_needed, block);
    elementwise_add_f16_pack<<<grid, block>>>(
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(b.data_ptr<at::Half>()),
        reinterpret_cast<half*>(c.data_ptr<at::Half>()), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_eltwise_add_f32_scalar)
    TORCH_BINDING_COMMON_EXTENSION(torch_eltwise_add_f32_vector)
    TORCH_BINDING_COMMON_EXTENSION(torch_eltwise_add_f16_scalar)
    TORCH_BINDING_COMMON_EXTENSION(torch_eltwise_add_f16_vec2)
    TORCH_BINDING_COMMON_EXTENSION(torch_eltwise_add_f16_vec2_4)
    TORCH_BINDING_COMMON_EXTENSION(torch_eltwise_add_f16_vec2_4_unroll)
    TORCH_BINDING_COMMON_EXTENSION(torch_eltwise_add_f16_pack)
}
