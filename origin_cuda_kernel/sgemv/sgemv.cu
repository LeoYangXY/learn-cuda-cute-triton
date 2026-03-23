#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <random>
#include <cmath>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])


// 很关键的对于四个视角的理解：
// thread 视角："我算谁"——决定每个线程负责哪些元素/哪一行哪一列。
// warp 视角："怎么协作算"——决定 warp 内如何合并访存、如何做规约。
// block 视角："资源与局部协作"——决定一个 block 覆盖多少行/多少元素，是否用 shared memory，warp 数量配置。
// grid 视角："全局覆盖范围"——决定有多少个 block、如何覆盖完整的矩阵维度。


//  Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

//如下的几个kernel都是针对特定 K 形状做的手工优化

//==k为32的倍数：
// 数据分配逻辑：
// 一个block处理4行，一行用一个warp去处理，因此就是：
// 设 warps_per_block = 4（或 8，看你想要的并行度）
// blockSize = warps_per_block * 32
// gridSize = ceil(M / warps_per_block)

template <int warps_per_block = 4>
__global__ void sgemv_32_kernel(float *a, float *x, float *y, int M,
                                        int K) {
    int row_start = blockIdx.x * warps_per_block;
    int row_offset = threadIdx.x / WARP_SIZE; // 0~(warps_per_block-1)
    int row = row_start + row_offset; // 当前 warp 负责的行
    int lane = threadIdx.x % WARP_SIZE; // 当前线程在 warp 中的位置

    int NUM_WARPS = K / WARP_SIZE; // 每行需要多少轮（K/32）

    //要做的是：每个 thread 先算自己负责的若干个（NUM_WARPS个），累加成局部和，然后 warp 内做规约得到整行点积
    //有两种方式：
    // 1.每个thread负责连续的NUM_WARPS个元素
    // 2.每个thread负责间隔的NUM_WARPS个元素，每次的step_size==32
    // 不过都是要循环NUM_WARPS次的
    
    float sum = 0.0f;
    for(int time=0;time<NUM_WARPS;++time){
        //第一轮中，thread0负责处理col0，thread1负责处理col1，...，thread31负责处理col31
        //第二轮中，thread0负责处理col32，thread1负责处理col33，...，thread31负责处理col63
        int col = time*32 + lane;
        sum += a[row*K + col] * x[col];
    }

    // 为什么我们上面的for循环里面使用的是第二种访存方式而不是第一种呢：
    // 第一种（连续访问）= 每个 thread 负责一段连续的 NUM_WARPS 个元素，比如
    // col = lane * NUM_WARPS + time
    // 问题在于：
    // 不合并访存：同一时刻（同一轮 time），warp 的 32 个线程访问的是
    // 0, NUM_WARPS, 2*NUM_WARPS, ... 这种"跨大步"的地址，不是连续地址，内存事务会被拆成很多段，带宽利用差。
    // 缓存友好性差：全局内存访问跨度大，L2/L1 预取和合并效果都不好。
    // 线程协作差：warp 内每个线程访问的地址分散，硬件很难把它们合并成少量 memory transaction。
    // 第二种：
    // 这样在同一轮 time，warp 内 32 个线程访问的是连续地址 time*32 .. time*32+31，这是最理想的 coalesced access，带宽利用最好，也更容易被缓存命中


    //现在每个thread都会拿到自己的那几个元素的sum，接下来在warp内规约得到整行的sum即可
    if (row < M) {
      float row_sum = warp_reduce_sum_f32(sum);
      if (lane == 0) {
        // 每个 warp 的 lane0 写回该行的结果
        y[row] = row_sum;
      }
    }

}

//==k为128的倍数（float4）：
// 其实本质上和上面的一样，只是每个thread在每一轮去取值的时候是以float4的形式去取4个连续的元素
template <int warps_per_block = 4>
__global__ void sgemv_128_kernel(float *a, float *x, float *y, int M,
                    int K) {
int row_start = blockIdx.x * warps_per_block;
int row_offset = threadIdx.x / WARP_SIZE; // 0~(warps_per_block-1)
int row = row_start + row_offset;
int lane = threadIdx.x % WARP_SIZE;

if (row >= M) return;

int iters = K / (WARP_SIZE * 4); // K/128
float sum = 0.0f;
for (int t = 0; t < iters; ++t) {
    // 每轮内，warp 访问连续的 128 个元素，保证合并访存
    int k = (t * WARP_SIZE + lane) * 4;
    float4 reg_x = FLOAT4(x[k]);
    float4 reg_a = FLOAT4(a[row * K + k]);
    sum += reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z +
        reg_a.w * reg_x.w;
}

float row_sum = warp_reduce_sum_f32(sum);
if (lane == 0) {
    y[row] = row_sum;
}
}


//==k为16，那么很自然的就是让一个warp负责2行
//不过至于一个block中包含多少个warp呢：这个参数 不是"固定真理"，一般是调优出来的（经验 + 性能测试）：
// warp 多一些 → 并行度更高，但寄存器/共享内存压力更大
// warp 少一些 → 资源压力小，但并行度可能不足
// 所以会按设备、K、M、算子形态做权衡。
// 此处我们还是按一个block中有4个warp，也就是128个thread去写


template <int warps_per_block = 4> //这就代表着一个block我们开128个thread
__global__ void sgemv_16_kernel(float *a, float *x, float *y, int M,
                    int K) {

    int row_start = blockIdx.x * warps_per_block *2;//一个block里面设置了4个warp，每个warp处理2行，所以一个block处理8行
    int row_offset = threadIdx.x/16;//一个block有128个thread，16个thread负责一个row，因此可以这样子算offset
    int row = row_start+row_offset;
    
    int warp_id = threadIdx.x / WARP_SIZE; // 0~3
    int lane = threadIdx.x % WARP_SIZE; // 0~31

    //0~15thread负责处理第一行，16~31thread负责处理第二行，32~47thread负责处理第三行....
    int col = lane%16;
    float val = a[row*K+col]*x[col];

    float sum = warp_reduce_sum_f32<16>(val);//注意这里引入了16这个模板参数，warp_reduce_sum_f32<16>：只在16‑lane 子组内做规约

    if(col%16==0){//每一行的一个代表thread负责把结果写入y
        y[row]=sum;
    }

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

#define CEIL(a,b) ((a+b-1)/(b))

// A: [M, K], x: [K] -> y: [M]
// K must be multiple of 32
torch::Tensor torch_sgemv_32(torch::Tensor A, torch::Tensor x) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1);
    auto y = torch::empty({M}, A.options());
    constexpr int warps_per_block = 4;
    dim3 block(warps_per_block * 32);
    dim3 grid(CEIL(M, warps_per_block));
    sgemv_32_kernel<warps_per_block><<<grid, block>>>(
        A.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), M, K);
    return y;
}

// K must be multiple of 128
torch::Tensor torch_sgemv_128(torch::Tensor A, torch::Tensor x) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1);
    auto y = torch::empty({M}, A.options());
    constexpr int warps_per_block = 4;
    dim3 block(warps_per_block * 32);
    dim3 grid(CEIL(M, warps_per_block));
    sgemv_128_kernel<warps_per_block><<<grid, block>>>(
        A.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), M, K);
    return y;
}

// K must be 16
torch::Tensor torch_sgemv_16(torch::Tensor A, torch::Tensor x) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32);
    int M = A.size(0), K = A.size(1);
    auto y = torch::empty({M}, A.options());
    constexpr int warps_per_block = 4;
    dim3 block(warps_per_block * 32);
    dim3 grid(CEIL(M, warps_per_block * 2));  // each block handles 8 rows
    sgemv_16_kernel<warps_per_block><<<grid, block>>>(
        A.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), M, K);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_sgemv_32)
    TORCH_BINDING_COMMON_EXTENSION(torch_sgemv_128)
    TORCH_BINDING_COMMON_EXTENSION(torch_sgemv_16)
}
