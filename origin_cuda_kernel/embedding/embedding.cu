#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>


#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

//对这个宏的理解：注意：指针的值就是指针指向的地址，也就是一个变量的首地址
// float output[1024];  对其调用：LDST128BITS(output[100])
// 那么宏展开后变成：(reinterpret_cast<float4 *>(&(output[100]))[0])
// 步骤 1：&(output[100])，取 output[100] 的地址 → 类型是 float*  假设这个指针指向的地址是 0x1000
// 步骤 2：reinterpret_cast<float4 *>(...) 把 float* 强转为 float4*  因此现在指针类型是 float4*，指针指向的地址仍是 0x1000（这便是那个首地址）
// 步骤 3：[0]（等价于 *ptr） 解引用这个 float4* 指针，相当于读取从 0x1000 开始的 16 字节（4 个 float），并将其解释为一个 float4 结构体
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])


//输入：索引数组 idx（长度为 n），权重表 weight（形状 [vocab_size, emb_size]）
//输出：output（形状 [n, emb_size]）
//对每个 i ∈ [0, n)，把 weight[idx[i]] 这一行（长度 emb_size）复制到 output[i]
__global__ void embedding_float4(const int *idx, float *weight,
                                     float *output, int n, int emb_size) {

    //任务划分：每个block负责填写最终output的一行
    //也就是从idx数组中取一个值然后找到对应的weight中的那一行，一个block中的一个thread负责4个元素

    int i=blockIdx.x;
    int row=idx[i];
    int col = threadIdx.x * 4;
    
    //每个thread负责weight[row][col]到weight[row][col+3]的四个元素
    float4* src=reinterpret_cast<float4*>(weight+row*emb_size+col);
    float4* dst=reinterpret_cast<float4*>(output+i*emb_size+col);
    
    //注意这样的float4指针的写法,去做向量化赋值
    *dst = *src;
}

__global__ void embedding_pack(const int *idx, float *weight, float *output, int n, int emb_size) {
    int i=blockIdx.x;
    int row=idx[i];
    int col = threadIdx.x * 4;
    //和上面的float4本质一样，只是我们可以用此去处理更多的数据类型,比如half
    LDST128BITS(output[i * emb_size + col]) = LDST128BITS(weight[row * emb_size + col]);
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

// idx: [n], int32;  weight: [vocab_size, emb_size], float32;  returns: [n, emb_size], float32
torch::Tensor torch_embedding_float4(torch::Tensor idx, torch::Tensor weight) {
    CHECK_TORCH_TENSOR_DTYPE(weight, torch::kFloat32);
    int n = idx.size(0);
    int emb_size = weight.size(1);
    auto output = torch::empty({n, emb_size}, weight.options());
    dim3 grid(n);
    dim3 block(emb_size / 4);
    embedding_float4<<<grid, block>>>(
        idx.data_ptr<int>(), weight.data_ptr<float>(), output.data_ptr<float>(), n, emb_size);
    return output;
}

torch::Tensor torch_embedding_pack(torch::Tensor idx, torch::Tensor weight) {
    CHECK_TORCH_TENSOR_DTYPE(weight, torch::kFloat32);
    int n = idx.size(0);
    int emb_size = weight.size(1);
    auto output = torch::empty({n, emb_size}, weight.options());
    dim3 grid(n);
    dim3 block(emb_size / 4);
    embedding_pack<<<grid, block>>>(
        idx.data_ptr<int>(), weight.data_ptr<float>(), output.data_ptr<float>(), n, emb_size);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_embedding_float4)
    TORCH_BINDING_COMMON_EXTENSION(torch_embedding_pack)
}
