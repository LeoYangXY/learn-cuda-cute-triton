#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include <cuda_fp16.h>


void max_cpu(float* input, float* output, int N) {
    *output =  *(std::max_element(input, input + N));  // 计算输入数组的最大值
}

// int atomicCAS(int* address, int compare, int val) {
//     int old = *address;               // 原子地读取当前值
//     if (old == compare) {             // 如果等于 compare
//         *address = val;               // 就原子地写入 val
//     }
//     return old;                       // 总是返回操作前的旧值
// }
// 直观理解：
// atomicCAS(address, A, B) 的作用是：
// 偷偷看一眼 *address 的值：如果它等于 A，就立刻把它改成 B；
// 不过最后不管改没改，都把「看到*address的那个旧值」返回给你。”


//实现原子操作，因为我们是要在一个内存位置上放最大值，因此需要原子操作
__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;  // address转为int指针
    int old = *address_as_i;  // address中的旧值，用int解码
    int assumed;
    do {
        assumed = old;  // assumed存储旧值
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

//每个thread负责一个元素，然后把自己负责的那个元素看看能不能放到最终的结果位置
__global__ void max_kernel_naive(float* input, float* output, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        atomicMax(output, input[idx]);
    }
}

//使用shared memory
// 整个数组 input[0..N-1] 被划分成多个 线程块（block）。
// 每个 block 内部先用 shared memory 找出自己 block 内的最大值（这叫“局部最大值”）。
// 然后所有 block 的局部最大值通过 atomicMax 竞争写入同一个全局变量 output[0]，最终得到全局最大值
__global__ void max_kernel_shared_only(float* input, float* output, int N) {
    extern __shared__ float sdata[];  // 动态 shared memory

    int global_idx=blockDim.x*blockIdx.x+threadIdx.x;
    int idx_in_block=threadIdx.x;
    
    sdata[idx_in_block]=(global_idx<N)?input[global_idx]:(-FLT_MAX);//使用三元运算符加速，注意越界的数据不要填进去
    __syncthreads();//等待所有线程都把数据写入shared memory

    //下面我们希望高效地利用 block 内的 shared memory，找出该 block 所有线程数据中的最大值
    // 如果我们使用串行计算的话：
    // 在一个 线程块（block） 中，假设有 8 个线程（blockDim.x = 8），每个线程手里有一个数：
    // sdata = [3, 7, 2, 9, 1, 5, 8, 4]
    // 我们要在 shared memory 里 快速算出这 8 个数的最大值 → 应该是 9。
    // 如果让一个线程从头扫到尾，要 8 次比较，这样就浪费了其他 7 个线程！
    // 而“树形规约”让所有线程并行参与计算，只用 3 轮（= log₂8） 就搞定！
    // 树形规约（Tree Reduction）在 shared memory 中找最大值

    // 目标：让 block 内所有线程协作，在 O(log(blockDim.x)) 轮内找出局部最大值
    // 示例：假设 blockDim.x = 8，初始 sdata = [3, 7, 2, 9, 1, 5, 8, 4]
    //
    // 第 1 轮：stride = 8 / 2 = 4
    //   只有 tid < 4 的线程参与（tid=0,1,2,3）
    //   每个线程比较 sdata[tid] 和 sdata[tid + 4]，保留较大者
    //     tid=0: max(3, 1) → 3
    //     tid=1: max(7, 5) → 7
    //     tid=2: max(2, 8) → 8
    //     tid=3: max(9, 4) → 9
    //   结果：sdata = [3, 7, 8, 9, 1, 5, 8, 4]（这个时候前4个已包含全部信息，后面的四个数据可以不用管了）
    //
    // 第 2 轮：stride = 4 / 2 = 2
    //   只有 tid < 2 的线程参与（tid=0,1）
    //     tid=0: max(sdata[0], sdata[2]) = max(3, 8) = 8
    //     tid=1: max(sdata[1], sdata[3]) = max(7, 9) = 9
    //   结果：sdata = [8, 9, 8, 9, ...] （这个时候前2个已包含全部信息，后面的那些数据可以不用管了）
    //
    // 第 3 轮：stride = 2 / 2 = 1
    //   只有 tid=0 参与
    //     tid=0: max(sdata[0], sdata[1]) = max(8, 9) = 9
    //   最终：sdata[0] = 9 ← 即本 block 的最大值
    //
    // 整个过程共 log2(8) = 3 轮，每轮后有效数据减半，最终 sdata[0] 存放结果。
    // 这种结构类似一棵倒置的二叉树（树形规约）：
    //         [9]
    //        /   \
    //      [8]   [9]
    //     / \   / \
    //   [3] [8] [7] [9]
    //  / \ / \ / \ / \
    // 3  1 2 8 7 5 9  4
    //
    // 优势：充分利用 shared memory 的高速和线程并行性，避免串行扫描。

    //这样子规约需要让blockSize为2的幂次，比如是128
    //在host侧分配的时候我们就做到了：每个block是128个thread的大小，然后尾块block因为要处理的有效数据不足128个，所以我们使用-FLT_MAX这样的东西来填充，避免影响我们这里的比较逻辑
    for(int stride=blockDim.x/2;stride>0;stride>>=1){//可以使用>>1去替代除以2
        //每个thread负责比较两个元素
        if(idx_in_block<stride){
            sdata[idx_in_block]=fmaxf(sdata[idx_in_block],sdata[idx_in_block+stride]);
        }
    }
    __syncthreads();

    //现在我们每个block的最大值都存储在sdata[0]中，因此我们需要把这个值写入全局内存中
    //我们只需要拿block中的一个thread去负责当前这个block的最大值的写入
    if(idx_in_block==0){
        atomicMax(output,sdata[0]);
    }
}



// 三种shfl函数讲解：
// T __shfl_xor_sync(
//     unsigned mask,    // 参与线程的位掩码 (通常0xffffffff)
//     T value,          // 要交换的值 (int/float)
//     int lane_mask,    // 异或操作的掩码 (通常为warp_size/2递减)
//     int width=32      // 实际参与线程数 (默认32)
// );
// 伪代码实现:
// int target_lane = thread_lane_id ^ lane_mask;
// return (target_lane < width) ? 
//        target_thread.value : 
//        undefined; // 通常返回原值
//注意：这个操作是原子的，不会出现thread0读取了thread16的值，写完得到新的之后，再做thread16去读thread0的值这样的

// T __shfl_down_sync(
//     unsigned mask,    // 参与线程的位掩码
//     T value,          // 要传递的值
//     unsigned delta,   // 向下移动的偏移量
//     int width=32      // 实际参与线程数
// );
// 伪代码实现:
// int target_lane = thread_lane_id + delta;
// return (target_lane < width) ? 
//        target_thread.value : 
//        undefined; // 通常返回原值


// T __shfl_up_sync(
//     unsigned mask,    // 参与线程的位掩码
//     T value,          // 要传递的值
//     unsigned delta,   // 向上移动的偏移量
//     int width=32      // 实际参与线程数
// );
// 伪代码实现:
// int target_lane = thread_lane_id - delta;
// return (target_lane >= 0) ? 
//        target_thread.value : 
//        undefined; // 通常返回0或原值


//cuda编程的时候我们的视角是thread
// 使用向下广播：第一轮：lane 0 读 lane 16，lane 1 读 lane 17，...，lane 15 读 lane 31
// 最终 lane 0 汇聚整个 warp 的最大值
template<const int kWarpSize = 32>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int offset = kWarpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));//注意：cuda编程的时候我们的视角是thread：因此是去看每一轮中，当前这个thread会拿到xxx
    }
    return val;
}

//每个thread负责读入一个数据，然后通过2层规约来实现
__global__ void max_kernel_shuffle(float* input, float* output, int N) {
    constexpr int kWarpSize = 32;
    __shared__ float sdata[32];//我们假设一个block中的thread数量小于32*32，因此最多放32个warp

    //第一层：每个warp内进行规约，然后把数据写到sdata
    int global_idx=blockDim.x*blockIdx.x+threadIdx.x;
    int idx_in_block=threadIdx.x;
    int warp_idx=threadIdx.x/kWarpSize;
    int idx_in_warp=threadIdx.x%kWarpSize;

    float val=(global_idx<N)?input[global_idx]:(-FLT_MAX);//每个thread先保存好自己负责的那个数据
    val=warp_reduce_max_f32(val);
    if(idx_in_warp==0){
        sdata[warp_idx]=val;//一个warp的结果由第0个thread负责写入即可
    }
    __syncthreads();//做完同步之后：这个时候sdata中已经存放了每个warp的最大值

    //现在sdata整个容量为32，有效数据<=32个，那么我们只需要使用1个warp中的32个thread即可完成规约，不需要使用整个warp中的所有thread
    if(warp_idx==0){
        float val=sdata[idx_in_warp];
        val=warp_reduce_max_f32(val);
        if(idx_in_warp==0){//也是去拿第0个thread去负责写入全局内存
            atomicMax(output,val);
        }
    }
}


// ========================================================================
// GPU 全局内存访问与 float4 优化原理详解
// ========================================================================
//
// 🧠 内存访问基本机制：
// - GPU 以 **warp（32 线程）** 为单位执行内存加载指令。
// - 硬件会将一个 warp 中所有线程的地址请求 **合并（coalesce）** 成尽可能少的内存事务。事务数由“总访问字节数”和“地址连续性”决定
// - 每个事务的最小粒度在现代 GPU（如 Ampere/Hopper）上通常为 **128 字节**（通过 L2 cache），
// - 即使单个线程只读 4 字节（1 个 float），只要整个 warp 的 32 个线程访问的是
//   **连续且对齐的地址**（如 a[0] ～ a[31]），硬件就能将其合并为 **1 次 128B 事务**，
//   实现带宽的高效利用。
//
// eg:
// - 每个线程一次性读取 16 字节（例如 8 个 FP16 元素）；
// - 线程 0 读取字节区间 [0, 15]
// - 线程 1 读取字节区间 [16, 31]
// - ...
// - 线程 31 读取字节区间 [496, 511]
// → 整个 warp 访问的是一个连续的 512 字节区域（32 × 16B = 512B）。
//
// GPU 内存子系统会将该连续区域合并为最少数量的 128 字节事务：
//   • 事务 0: bytes 0–127
//   • 事务 1: bytes 128–255
//   • 事务 2: bytes 256–383
//   • 事务 3: bytes 384–511
// 因此，整个 warp 仅触发 4 次全局内存事务
//
//
// 🚀 优化策略：每个线程处理多个元素（如使用 float4 一次读 4 个 float）
// 虽然总数据量不变（因此总内存事务数大致相同），但该策略带来三大核心优势：
//
// 📊 完整示例：N = 1024，求 input[0..1023] 的最大值
//   • 方案 A（1 线程 = 1 float）：
//       - 需 1024 线程（如 4 blocks × 256 threads）
//       - 触发 1024 / 32 = 32 次 128B 事务
//       - 使用 32 warps，高寄存器占用，4 次 __syncthreads()
//   • 方案 B（1 线程 = 4 floats via float4）：
//       - 仅需 256 线程（4 blocks × 64 threads）
//       - 总数据量仍为 4096B → 同样触发 4096 / 128 = 32 次事务
//       - 但仅用 8 warps（64 线程/block → 2 warps/block）
//
// ✅ 核心优势（非“计算变多”，而是“开销变少”）：
//    【线程与资源开销大幅降低】
//      - 线程数减少至 1/4 → 寄存器总占用下降，shared memory 压力减小；
//      - 更易达到高 occupancy（SM 可调度更多 block 并行执行）。
//

//每个thread负责读取4个元素，然后再做2层规约
__global__ void max_kernel_shuffle_float4(float* input, float* output, int N) {
    constexpr int kWarpSize = 32;
    __shared__ float sdata[32]; // 最多 32 warps

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_start = (bid * blockDim.x + tid) * 4; // 每个线程负责 4 个 float

    int warp_idx = tid / kWarpSize;
    int lane = tid % kWarpSize;

    // Step 1: Load 4 floats as float4 (128-bit load)
    float4 vec= *reinterpret_cast<float4*>(input + global_start);

    // Step 2: 求这 4 个值的最大值
    float local_max = fmaxf(fmaxf(vec.x, vec.y), fmaxf(vec.z, vec.w));

    // Step 3: Warp 内规约
    local_max = warp_reduce_max_f32(local_max);

    // Step 4: 每个 warp 的 leader 写入 shared memory
    if (lane == 0) {
        sdata[warp_idx] = local_max;
    }
    __syncthreads();

    // Step 5: 第一个 warp 对 sdata 做最终规约
    if (warp_idx == 0) {
        float val = (lane < (blockDim.x + kWarpSize - 1) / kWarpSize) ? sdata[lane] : -FLT_MAX;
        val = warp_reduce_max_f32(val);
        if (lane == 0) {
            atomicMax(output, val);
        }
    }
}


//对于别的数据类型，我们也可以用上面那个float4类似的思想，只是我们这里要做一次pack：
// 辅助宏：128-bit load/store via float4
#define LDST128BITS(x) (*(reinterpret_cast<float4*>(&(x))))

__global__ void max_kernel_shuffle_pack_half(half* input, float* output, int N) {
    constexpr int kWarpSize = 32;
    constexpr int kElementsPerThread = 8;
    __shared__ float sdata[32];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int global_start = (bid * blockDim.x + tid) * 8; // 每个线程处理 8 个 half

    int warp_idx = tid / kWarpSize;
    int lane = tid % kWarpSize;

    // Step 1: Load 8 halfs as 128-bit vector
    half local_pack[8];
    //这一步是语法糖，达到的效果就是把&input这个指针指向的地址，从global_start开始的128bits加载到&local_pack这个指针指向的地址开始的128bits
    LDST128BITS(local_pack[0]) = LDST128BITS(input[global_start]);

    // Step 2: 求出当前thread负责的那8个元素中的最大的
    float local_max = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < kElementsPerThread; ++i) {
        if (global_start + i < N) {
            float f = __half2float(local_pack[i]);
            local_max = fmaxf(local_max, f);
        }
        // else: skip (or you could do: local_max = fmaxf(local_max, -FLT_MAX);)
    }

    // Step 3: Warp 内规约（用 float）
    local_max = warp_reduce_max_f32(local_max);

    // Step 4: Warp leader writes to shared memory
    if (lane == 0) {
        sdata[warp_idx] = local_max;
    }
    __syncthreads();

    // Step 5: First warp reduces across warps
    if (warp_idx == 0) {
        int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
        float val = (lane < num_warps) ? sdata[lane] : -FLT_MAX;
        val = warp_reduce_max_f32(val);
        if (lane == 0) {
            atomicMax(output, val); 
        }
    }
}

// 对于不同数据精度的运算：
// float / double / int 等基本类型：可以直接用 +, -, * 等运算符，编译器会生成对应指令。
// half (FP16)、__nv_bfloat16 (BF16)、__nv_fp8_storage_t (FP8)：它们本质上是 封装的结构体或 typedef，不支持原生算术运算符重载（或仅部分支持），且 GPU 硬件对它们的计算有特殊要求。
// 要么用专用的算数指令，要么转化为fp32然后计算






int main() {
    size_t N = 1280;
    constexpr size_t BLOCK_SIZE = 128;
    const int repeat_times = 10;
    float ref_max = static_cast<float>(N); // expected max = 1280.0

    // ==============================
    // Test 0: max_kernel_naive (float)
    // ==============================
    {
        float* input = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; ++i) {
            input[i] = static_cast<float>(i + 1);
        }

        float *d_input = nullptr, *d_output = nullptr;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, sizeof(float));
        cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float result;
        float total_time = 0.0f;
        for (int rep = 0; rep < repeat_times; ++rep) {
            float init_val = -FLT_MAX;
            cudaMemcpy(d_output, &init_val, sizeof(float), cudaMemcpyHostToDevice);

            int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaEventRecord(start);
            max_kernel_naive<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;

            cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
            if (fabs(result - ref_max) >= 1e-5) {
                printf("[max_kernel_naive] ❌ FAIL at rep %d: got %f\n", rep, result);
                break;
            }
        }

        printf("[max_kernel_naive] Avg time: %.3f ms, result = %f, %s\n",
               total_time / repeat_times, result,
               (fabs(result - ref_max) < 1e-5) ? "✅ PASS" : "❌ FAIL");

        free(input);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ==============================
    // Test 1: max_kernel_shared_only (float)
    // ==============================
    {
        float* input = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; ++i) {
            input[i] = static_cast<float>(i + 1);
        }

        float *d_input = nullptr, *d_output = nullptr;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, sizeof(float));
        cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float result;
        float total_time = 0.0f;
        for (int rep = 0; rep < repeat_times; ++rep) {
            float init_val = -FLT_MAX;
            cudaMemcpy(d_output, &init_val, sizeof(float), cudaMemcpyHostToDevice);

            int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
            cudaEventRecord(start);
            max_kernel_shared_only<<<grid_size, BLOCK_SIZE, shared_mem_size>>>(d_input, d_output, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;

            cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
            if (fabs(result - ref_max) >= 1e-5) {
                printf("[max_kernel_shared_only] ❌ FAIL at rep %d: got %f\n", rep, result);
                break;
            }
        }

        printf("[max_kernel_shared_only] Avg time: %.3f ms, result = %f, %s\n",
               total_time / repeat_times, result,
               (fabs(result - ref_max) < 1e-5) ? "✅ PASS" : "❌ FAIL");

        free(input);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ==============================
    // Test 2: max_kernel_shuffle (float)
    // ==============================
    {
        float* input = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; ++i) {
            input[i] = static_cast<float>(i + 1);
        }

        float *d_input = nullptr, *d_output = nullptr;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, sizeof(float));
        cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float result;
        float total_time = 0.0f;
        for (int rep = 0; rep < repeat_times; ++rep) {
            float init_val = -FLT_MAX;
            cudaMemcpy(d_output, &init_val, sizeof(float), cudaMemcpyHostToDevice);

            int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaEventRecord(start);
            max_kernel_shuffle<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;

            cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
            if (fabs(result - ref_max) >= 1e-5) {
                printf("[max_kernel_shuffle] ❌ FAIL at rep %d: got %f\n", rep, result);
                break;
            }
        }

        printf("[max_kernel_shuffle] Avg time: %.3f ms, result = %f, %s\n",
               total_time / repeat_times, result,
               (fabs(result - ref_max) < 1e-5) ? "✅ PASS" : "❌ FAIL");

        free(input);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ==============================
    // Test 3: max_kernel_shuffle_float4 (float input)
    // ==============================
    {
        float* input = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; ++i) {
            input[i] = static_cast<float>(i + 1);
        }

        float *d_input = nullptr, *d_output = nullptr;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, sizeof(float));
        cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float result;
        float total_time = 0.0f;
        for (int rep = 0; rep < repeat_times; ++rep) {
            float init_val = -FLT_MAX;
            cudaMemcpy(d_output, &init_val, sizeof(float), cudaMemcpyHostToDevice);

            int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaEventRecord(start);
            max_kernel_shuffle_float4<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;

            cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
            if (fabs(result - ref_max) >= 1e-5) {
                printf("[max_kernel_shuffle_float4] ❌ FAIL at rep %d: got %f\n", rep, result);
                break;
            }
        }

        printf("[max_kernel_shuffle_float4] Avg time: %.3f ms, result = %f, %s\n",
               total_time / repeat_times, result,
               (fabs(result - ref_max) < 1e-5) ? "✅ PASS" : "❌ FAIL");

        free(input);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ==============================
    // Test 4: max_kernel_shuffle_pack_half (half input → float output)
    // ==============================
    {
        half* h_input = (half*)malloc(N * sizeof(half));
        for (size_t i = 0; i < N; ++i) {
            h_input[i] = __float2half(static_cast<float>(i + 1));
        }

        half *d_input = nullptr;
        float *d_output = nullptr;
        cudaMalloc(&d_input, N * sizeof(half));
        cudaMalloc(&d_output, sizeof(float));
        cudaMemcpy(d_input, h_input, N * sizeof(half), cudaMemcpyHostToDevice);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float result;
        float total_time = 0.0f;
        for (int rep = 0; rep < repeat_times; ++rep) {
            float init_val = -FLT_MAX;
            cudaMemcpy(d_output, &init_val, sizeof(float), cudaMemcpyHostToDevice);

            int num_chunks = (N + 8 - 1) / 8;
            int grid_size = (num_chunks + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaEventRecord(start);
            max_kernel_shuffle_pack_half<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float elapsed;
            cudaEventElapsedTime(&elapsed, start, stop);
            total_time += elapsed;

            cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
            if (fabs(result - ref_max) >= 1e-3) {
                printf("[max_kernel_shuffle_pack_half] ❌ FAIL at rep %d: got %f\n", rep, result);
                break;
            }
        }

        printf("[max_kernel_shuffle_pack_half] Avg time: %.3f ms, result = %f, %s\n",
               total_time / repeat_times, result,
               (fabs(result - ref_max) < 1e-3) ? "✅ PASS" : "❌ FAIL");

        free(h_input);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}