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








// CPU 参考实现
void embedding_cpu(const std::vector<int>& h_idx,
                   const std::vector<float>& h_weight,
                   std::vector<float>& h_output,
                   int n, int vocab_size, int emb_size) {
    for (int i = 0; i < n; ++i) {
        int row = h_idx[i];
        if (row < 0 || row >= vocab_size) {
            fprintf(stderr, "Invalid index at %d: %d\n", i, row);
            exit(1);
        }
        for (int j = 0; j < emb_size; ++j) {
            h_output[i * emb_size + j] = h_weight[row * emb_size + j];
        }
    }
}

// 验证结果是否正确
bool verify(const std::vector<float>& ref, const std::vector<float>& test, float tol = 1e-5f) {
    if (ref.size() != test.size()) return false;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > tol) {
            printf("Mismatch at %zu: ref=%f, test=%f\n", i, ref[i], test[i]);
            return false;
        }
    }
    return true;
}

// 性能测试函数
void benchmark(const char* name, dim3 grid, dim3 block,
               const int* d_idx, float* d_weight, float* d_output,
               int n, int emb_size, int iterations = 100) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        if (strcmp(name, "float4") == 0) {
            embedding_float4<<<grid, block>>>(d_idx, d_weight, d_output, n, emb_size);
        } else {
            embedding_pack<<<grid, block>>>(d_idx, d_weight, d_output, n, emb_size);
        }
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        if (strcmp(name, "float4") == 0) {
            embedding_float4<<<grid, block>>>(d_idx, d_weight, d_output, n, emb_size);
        } else {
            embedding_pack<<<grid, block>>>(d_idx, d_weight, d_output, n, emb_size);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double avg_time_us = milliseconds * 1000.0 / iterations;
    double bandwidth = (double(n) * emb_size * sizeof(float)) / (avg_time_us * 1e-6) / (1e9); // GB/s

    printf("[%s] Avg time: %.3f us, Bandwidth: %.2f GB/s\n", name, avg_time_us, bandwidth);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // 参数设置（可调整）
    const int n = 1024;          // batch size
    const int vocab_size = 32768; // 词表大小
    const int emb_size = 1024;   // embedding 维度（必须是 4 的倍数！）

    if (emb_size % 4 != 0) {
        fprintf(stderr, "Error: emb_size must be divisible by 4!\n");
        return 1;
    }

    printf("Testing embedding kernels with n=%d, vocab_size=%d, emb_size=%d\n",
           n, vocab_size, emb_size);

    // === 1. 分配主机内存 ===
    std::vector<int> h_idx(n);
    std::vector<float> h_weight(vocab_size * emb_size);
    std::vector<float> h_output_ref(n * emb_size);
    std::vector<float> h_output_test(n * emb_size);

    // 初始化随机数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> idx_dist(0, vocab_size - 1);
    std::uniform_real_distribution<float> weight_dist(-1.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        h_idx[i] = idx_dist(gen);
    }
    for (size_t i = 0; i < h_weight.size(); ++i) {
        h_weight[i] = weight_dist(gen);
    }

    // === 2. CPU 参考结果 ===
    embedding_cpu(h_idx, h_weight, h_output_ref, n, vocab_size, emb_size);

    // === 3. 分配设备内存 ===
    int *d_idx = nullptr;
    float *d_weight = nullptr, *d_output = nullptr;

    cudaMalloc(&d_idx, n * sizeof(int));
    cudaMalloc(&d_weight, vocab_size * emb_size * sizeof(float));
    cudaMalloc(&d_output, n * emb_size * sizeof(float));

    cudaMemcpy(d_idx, h_idx.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), vocab_size * emb_size * sizeof(float), cudaMemcpyHostToDevice);

    // === 4. 启动配置 ===
    dim3 grid(n);
    dim3 block(emb_size / 4); // 每个线程处理 4 个 float

    // === 5. 测试 embedding_float4 ===
    embedding_float4<<<grid, block>>>(d_idx, d_weight, d_output, n, emb_size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_test.data(), d_output, n * emb_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (!verify(h_output_ref, h_output_test)) {
        printf("❌ embedding_float4 failed!\n");
        return 1;
    } else {
        printf("✅ embedding_float4 passed!\n");
    }

    // === 6. 测试 embedding_pack ===
    embedding_pack<<<grid, block>>>(d_idx, d_weight, d_output, n, emb_size);
    cudaDeviceSynchronize();
    cudaMemcpy(h_output_test.data(), d_output, n * emb_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (!verify(h_output_ref, h_output_test)) {
        printf("❌ embedding_pack failed!\n");
        return 1;
    } else {
        printf("✅ embedding_pack passed!\n");
    }

    // === 7. 性能 benchmark ===
    printf("\n--- Performance Benchmark (100 iterations) ---\n");
    benchmark("float4", grid, block, d_idx, d_weight, d_output, n, emb_size);
    benchmark("pack",   grid, block, d_idx, d_weight, d_output, n, emb_size);

    // === 8. 清理 ===
    cudaFree(d_idx);
    cudaFree(d_weight);
    cudaFree(d_output);

    printf("\n🎉 All tests passed!\n");
    return 0;
}