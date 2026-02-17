#include <algorithm>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
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



//每个thread负责计算c中的一个元素，c[i][j] = a[i][:] * b[:][j]
__global__ void sgemm_naive(float *a, float *b, float *c, int M,
                                       int N, int K){

    int row=blockIdx.y*blockDim.y+threadIdx.y; //行索引
    int col=blockIdx.x*blockDim.x+threadIdx.x; //列索引

    if(row<M && col<N){
        float sum=0.0f;
        #pragma unroll    //注意可以使用循环展开
        for(int k=0;k<K;++k){
            sum+=a[row*K+k]*b[k*N+col];
        }
        c[row*N+col]=sum;
    }
}

/*
 * 将输出矩阵 C 划分为多个tile，每个tile的大小为 BM × BN 
 * 每个 CUDA Block 负责计算一个 C-tile。
 * 每个 Thread 在该 tile 内负责计算 TM × TN 个元素。
 *
 * 为减少全局内存访问，A 和 B 的数据通过 shared memory（smem）缓存：
 * - 每次从 A 中读取 BM × BK 的分块（沿 K 维）
 * - 每次从 B 中读取 BK × BN 的分块（沿 K 维）
 * - 所有线程协同将这些分块加载到 smem 中
 * 为什么要加载BK的分块而不是每次直接从A中读取BM*K的分块，从B中读取K*BN的分块：
 * 这是因为smem有限，我们这样子在K的维度上进行分块能让smem不爆
 *
 * 然后在 smem 中完成局部矩阵乘法：(BM × BK) @ (BK × BN) → (BM × BN)
 * 循环 K / BK 次，覆盖整个 K 维度，最终累加得到完整的 C-tile。
*/

/*
 * 如何选择分块参数 BM, BN, BK, TM, TN(block大小以及每个thread要处理的元素数量)？
 *
 * 核心原则：先由 shared memory 容量确定 BM/BN/BK，再由寄存器和线程模型确定 TM/TN。
 *
 * ✅ 步骤 1: 确定 BM, BN, BK（受 shared memory 限制）
 *   - smem_usage = BM * BK * sizeof(T)   // A 的 tile
 *                + BK * BN * sizeof(T)   // B 的 tile
 *   - 以 FP32 为例（sizeof(float)=4），RTX 4090 每 block 建议 ≤ 96KB smem
 *     → 最多容纳 96KB / 4B = 24576 个 float
 *   - 若选 BK = 32，则需满足: BM*32 + 32*BN ≤ 24576
 *   - 常见高效配置: BM = 128, BN = 128
 *       → smem = 128*32 + 32*128 = 8192 floats = 32KB ✅
 *
 * ✅ 步骤 2: 确定 TM × TN（受线程数和寄存器限制）
 *   - 假设目标 block 线程数 = 256（如 16×16 网格）
 *   - 每个 thread 负责 TM×TN 个 C 元素 ⇒ (BM / TM) × (BN / TN) = 256
 *   - 代入 BM=128, BN=128:
 *         (128 / TM) × (128 / TN) = 256
 *         ⇒ TM × TN = (128 × 128) / 256 = 64
 *   - 可能的整数分解组合:
 *         TM=8,  TN=8   → 64
 *         TM=16, TN=4   → 64
 *         TM=4,  TN=16  → 64
 *
 * ✅ 步骤 3: 从可行组合中筛选最优 TM/TN
 *   - 优先 TN 是 4 的倍数 → 支持 float4 向量化加载
 *   - 若使用 MMA（Tensor Core），TM 应匹配硬件 shape（如 8 或 16）
 *   - 寄存器用量 ≈ TM×TN（累加器）+ TM + TN（临时数据）
 *       → 应 < 200 个寄存器/thread（避免溢出到 local memory）
 *   - 通常选择 TM ≈ TN（如 8×8）以平衡访存与计算
 *
 * 📌 示例结论：
 *   对于 FP32 GEMM on RTX 4090:
 *       BM=128, BN=128, BK=32, TM=8, TN=8, threads_per_block=256
 *   对于教学/轻量 kernel（如本文件）:
 *       BM=32, BN=32, BK=32, TM=4, TN=4, threads_per_block=64
 */

//1.用数学公式去写出每个thread要做的事情，在分析的时候也是使用数学公式去映射，转化，而不是硬想

//2.compute任务和load任务是独立的，分析的时候各自做合理的映射即可：
//比如我们的compute任务是在写kernel之前就确定好了，每个thread负责计算C中的TM*TN个元素

//不过这个kernel存在一些硬编码，比如有的时候应该使用TM的，我们直接使用了4；应该使用BM/TM的，我们直接使用了8，泛化性做的不够
template <const int BM = 32, const int BN = 32, const int BK = 32,const int TM = 4, const int TN = 4>
__global__ void sgemm_sliced_k_f32_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {

    //一个block负责32*32的tile，一个thread负责4*4的子tile，因此一个block中：blockIdx.x=8,blockIdx.y=8
    __shared__ float s_a[BM][BK], s_b[BK][BN];


    int block_row = blockIdx.y*BM; // 当前 block 负责的 C-tile 起始行
    int block_col = blockIdx.x*BN; // 当前 block 负责的 C-tile 起始列

    //用数学公式去写每个thread要负责计算什么：
    // C[thread_compute_row:thread_compute_row+TM][thread_compute_col:thread_compute_col+TN] = A[thread_compute_row:thread_compute_row+TM][:] @ B[:][thread_compute_col:thread_compute_col+TN]
    // 因此在每一轮的BK的循环中，要去算：C[thread_compute_row:thread_compute_row+TM][thread_compute_col:thread_compute_col+TN] += A[thread_compute_row:thread_compute_row+TM][k0:k0+BK] @ B[k0:k0+BK][thread_compute_col:thread_compute_col+TN]
    int thread_compute_row = blockIdx.y * BM + threadIdx.y * TM; // 当前 thread 负责的 C-tile 起始行
    int thread_compute_col = blockIdx.x * BN + threadIdx.x * TN; // 当前 thread 负责的 C-tile 起始列

    // 总共一个block就是8*8个thread，然后每一轮是32*32个元素要加载，
    // 所以反正一个thread要加载16个元素，至于怎么分配就是自己去决定即可
    // 比如可以一个thread做load的时候处理逻辑上1*16的子块，也就是连续的16个元素；也可以是逻辑上2*8的子块，也就是8个物理上连续的，然后跳一些地址再到下8个物理上连续的；也可以是逻辑上4*4的子块
    
    // 那至于按照什么方式去访问smem，就要去分析bank是否会conflict了：
    // bank conflict 计算：
    // - shared memory 有 32 个 bank，每个 bank 宽度是 4 字节，这是规定好的
    // - 对于shared memory中的一个元素，其对应的bank 号 = (address / 4) % 32  其中address为其首地址
    // - s_a[row][col] 的线性索引 index = row * BK + col
    // - 因为里面的元素是float形式，因此任意一个元素的地址是 address = 4 * index
    // - 所以任意一个元素所属的 bank = (row * BK + col) % 32
    // 对于此就可以进行直观的解读了：同一列 col 在不同 row 上会落到同一个 bank，所以 warp 内如果多线程同时访问“同一列不同 row”，就会发生 bank conflict
    // 因此，4×4、2×8、1×16 的bank conflict严重程度不一样（虽然最好还是加一个padding去分配shared memory）


    // 在分配的时候，我们是去思考每个thread要拿什么数据，然后再用数学公式去找到这个数据的位置与threadIdx.x,threadIdx.y,tid的统一的关系
    // 以便所有的thread都能用同样的公式去找到自己的数据
    // 而不是先盲目的写出threadIdx.x,threadIdx.y,tid（把二维的threadIdx转化为一维的，tid=threadIdx.y * blockDim.x + threadIdx.x）然后再强行分配
    // 比如我们这里想要做4*4的load的映射，那么可以是：
    // (0,0)的thread去load [0:4][0:4]的元素，(0,1)的thread去load [0:4][4:8]的元素，(1,0)的thread去load [4:8][0:4]的元素，(1,1)的thread去load [4:8][4:8]的元素，以此类推
    // 写出来这个之后，我们就可以思考这些位置和threadIdx.x,threadIdx.y,tid的关系了：
    // row = threadIdx.y*4; col = threadIdx.x*4;
    int thread_load_row = threadIdx.y * 4;//当然，这里最好不使用4，而是使用TM这样才更有泛化性
    int thread_load_col = threadIdx.x * 4;
    //那如果我们想要的是每个thread去load 2*8的元素，那么可以是：
    // 32/2 = 16 个 row-block；32/8 = 4 个 col-block；我们可以把这些block按照行主序进行标号
    // 因此(0,0)的thread去load 第(0,0)的block；thread(0,1)的thread去load 第(0,1)的block；thread(0,2)的thread去load 第(0,2)的block；thread(0,3)的thread去load 第(0,3)的block；thread(1,0)的thread去load 第(1,0)的block；以此类推
    // 因此就是 tid = threadIdx.y*8+threadIdx.x,这就是每个thread的一维索引，
    // 由上面的数学关系，我们需要去load的block的一维索引为tid，然后因为是16×4个子块，所以行号 = tid / 4（跨过每 4 个子块就进下一行），列号 = tid % 4（当前行里的第几个）
    // 因此就是要去load的元素的起始行号为 row = (tid / 4) * 2，起始列号为 col = (tid % 4) * 8


    for(int k0 = 0; k0 < K; k0 += BK) {  
        //注意：上面的每个thread要去计算什么，和这里的每个thread要去load什么可以理解为是独立的，这里该怎么load就怎么load
        //在分析shared_memory填充的时候，我们要从block的角度先入手，然后也是用数学公式去写出来：
        //在第 k0 轮,需要把A[block_row:block_row+BM][k0:k0+BK]和B[k0:k0+BK][block_col:block_col+BN]加载到shared memory中
        //因此一个比较直观简单的映射就是:
        // A[block_row+i][k0+j]放到s_a[i][j]
        // B[k0+i][block_col+j]放到s_b[i][j]
        //那么分配到每个thread，比如我们按照上面的4*4的方式去load，我们上面已经分配好了，每个thread要做load的元素的起始行号和起始列号，那么我们就可以直接套用这个公式去load了：

        for(int i=0;i<4;++i){
            for(int j=0;j<4;++j){
                // 每个thread要load的，加上循环的偏移，然后再加上block的偏移
                int a_row = thread_load_row + i+ block_row;
                int a_col = thread_load_col + j + k0;
                int s_a_row = thread_load_row + i;
                int s_a_col = thread_load_col + j;
                s_a[s_a_row][s_a_col] = (a_row < M && a_col < K) ? a[a_row*K + a_col] : 0.0f; // 注意边界检查

                int b_row = thread_load_row + i + k0;
                int b_col = thread_load_col + j + block_col;
                int s_b_row = thread_load_row + i;
                int s_b_col = thread_load_col + j;
                s_b[s_b_row][s_b_col] = (b_row < K && b_col < N) ? b[b_row*N + b_col] : 0.0f; // 注意边界检查
            }
        }

        __syncthreads(); // 确保所有线程都完成了加载

        for(int i=0;i<TM;++i){
            for(int j=0;j<TN;++j){
                //每个thread要做的是：C[thread_compute_row:thread_compute_row+TM][thread_compute_col:thread_compute_col+TN] += A[thread_compute_row:thread_compute_row+TM][k0:k0+BK] @ B[k0:k0+BK][thread_compute_col:thread_compute_col+TN]
                //因此对于每一个元素就是 C[thread_compute_row+i][thread_compute_col+j] += A[thread_compute_row+i][k0:k0+BK] @ B[k0:k0+BK][thread_compute_col+j]
                //而我们已经把A[thread_compute_row:thread_compute_row+TM][k0:k0+BK]和B[k0:k0+BK][thread_compute_col:thread_compute_col+TN]加载到了shared memory中，由之前的映射关系（A[block_row+i][k0+j]放到s_a[i][j]中（相当于一个减block_row,一个减k0），B[k0+i][block_col+j]放到s_b[i][j]）:
                //因此对于每一个元素就是 C[thread_compute_row+i][thread_compute_col+j] += s_a[thread_compute_row+i-block_row][0:BK] @ s_b[0:BK][thread_compute_col+j-block_col]
                float sum = 0.0f;
                for(int p=0;p<BK;++p){
                    sum += s_a[thread_compute_row + i - block_row][p] * s_b[p][thread_compute_col + j - block_col];
                }
                c[(thread_compute_row + i)*N + thread_compute_col + j] += sum; 
            }
        }

        __syncthreads();//如果没加，下一轮 k0 开始时，有的线程已经开始覆盖 s_a/s_b，而有的线程还在读取上一轮的数据，就会出现错结果

    }
}       


// #pragma unroll 不一定加速。
// 只有在循环小、边界固定、寄存器压力可控时才更快。
// 可能变慢的情况：寄存器溢出、指令缓存压力、编译器已经自动做了更合理的展开

//一个 SM 通常可以同时驻留多个 block，硬件资源（寄存器、shared memory、线程数、block 数上限）决定了同一时刻能放多少个 block
//只有在以下情况才会“看起来一个 block 独占一个 SM”：
// block 线程数太大（接近上限）
// shared memory 用量太高
// 每线程寄存器太多
// 这些都会把其他 block 挤掉

//用float4去load和直接用float去load的区别：
//                 ┌───────────────────────┐
//                 │      GPU Kernel       │
//                 └──────────┬────────────┘
//                            ▼
//            ┌───────────────────────────────┐
//            │ Arithmetic Intensity (AI)     │
//            │ = FLOPs / Bytes Transferred   │
//            └──────────────┬────────────────┘
//                           │
//         AI 很低 (< ~10)   │   AI 很高 (> ~50)
//         （如向量加、naive GEMM）│   （如 tiled GEMM、FFT）
//                           │
//              ┌────────────▼────────────┐
//              │                         │
//     Bandwidth-Bound (内存瓶颈)    Compute-Bound (计算瓶颈)
//              │                         │
//   优化重点：合并访存、减少冗余读写     优化重点：提高 occupancy、
//              │                    减少分支、用 tensor core
//   float vs float4 差异较小（<15%）     float4 优势更大（指令少）




//使用float4+padding去优化：
template <const int BM = 32, const int BN = 32, const int BK = 32,const int TM = 4, const int TN = 4>
__global__ void sgemm_sliced_k_f32x4_padding_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {

    // 此处的padding需要使用+4才能跑，+1会出现地址错误
    // 在 CUDA 中，当你使用 float4 类型进行读写时（无论是全局内存还是共享内存），硬件有一个铁律：
    // 起始内存地址必须是 16 字节对齐的。
    // 也就是说，地址值必须能被 16 整除 (Address % 16 == 0)。
    // 如果地址是 ...00, ...16, ...32，没问题。
    // 如果地址是 ...04, ...08, ...12，GPU 会直接报错 misaligned address
    // __shared__ float s_a[BM][BK+1], s_b[BK][BN+1];  用这个是错的

    __shared__ float s_a[BM][BK+4], s_b[BK][BN+4]; 

    int block_row = blockIdx.y*BM; 
    int block_col = blockIdx.x*BN; 
    int thread_compute_row = blockIdx.y * BM + threadIdx.y * TM; 
    int thread_compute_col = blockIdx.x * BN + threadIdx.x * TN; 
    int thread_load_row = threadIdx.y * 4;
    int thread_load_col = threadIdx.x * 4;

    for(int k0 = 0; k0 < K; k0 += BK) {  
        for(int i=0;i<4;++i){
            //其实就是把这里的for循环的float的加载变成使用float4去加载
            int a_row = thread_load_row + i+ block_row;
            int a_col = thread_load_col + k0;// j=0 起点
            int s_a_row = thread_load_row + i;
            int s_a_col = thread_load_col;//把下面的式子的j直接改为0，代表用float4去替代for(j=0;j<4;++j)的float加载循环
            FLOAT4(s_a[s_a_row][s_a_col])=FLOAT4(a[a_row*K+a_col]);

            int b_row = thread_load_row + i + k0;
            int b_col = thread_load_col + block_col;// j=0 起点
            int s_b_row = thread_load_row + i;
            int s_b_col = thread_load_col;//把下面的式子的j直接改为0，代表用float4去替代for(j=0;j<4;++j)的
            FLOAT4(s_b[s_b_row][s_b_col]) = FLOAT4(b[b_row*N + b_col]);

            // for(int j=0;j<4;++j){
            //     int a_row = thread_load_row + i+ block_row;
            //     int a_col = thread_load_col + j + k0;
            //     int s_a_row = thread_load_row + i;
            //     int s_a_col = thread_load_col + j;
            //     s_a[s_a_row][s_a_col] = (a_row < M && a_col < K) ? a[a_row*K + a_col] : 0.0f; // 注意边界检查

            //     int b_row = thread_load_row + i + k0;
            //     int b_col = thread_load_col + j + block_col;
            //     int s_b_row = thread_load_row + i;
            //     int s_b_col = thread_load_col + j;
            //     s_b[s_b_row][s_b_col] = (b_row < K && b_col < N) ? b[b_row*N + b_col] : 0.0f; // 注意边界检查
            // }

        }

        __syncthreads(); 

        for(int i=0;i<TM;++i){
            for(int j=0;j<TN;++j){
                float sum = 0.0f;
                for(int p=0;p<BK;++p){
                    sum += s_a[thread_compute_row + i - block_row][p] * s_b[p][thread_compute_col + j - block_col];
                }
                c[(thread_compute_row + i)*N + thread_compute_col + j] += sum; 
            }
        }

        __syncthreads();
    }
}       

// 使用寄存器优化：
// 因为我们是每个thread负责计算C中的TM*TN个元素，所以我们完全可以把这TM*TN个元素（频繁使用的中间变量）都放在寄存器里，来避免对全局内存的频繁访问
template <const int BM = 32, const int BN = 32, const int BK = 32,const int TM = 4, const int TN = 4>
__global__ void sgemm_sliced_k_f32x4_padding_reg_kernel(float *a, float *b, float *c, int M,
                                          int N, int K) {

    __shared__ float s_a[BM][BK+4], s_b[BK][BN+4];

    // 每个 thread 负责计算 TM×TN 个 C 元素，使用寄存器缓存
    float c_reg[TM][TN] = {0};
    int block_row = blockIdx.y*BM; 
    int block_col = blockIdx.x*BN; 
    int thread_compute_row = blockIdx.y * BM + threadIdx.y * TM; 
    int thread_compute_col = blockIdx.x * BN + threadIdx.x * TN; 
    int thread_load_row = threadIdx.y * 4;
    int thread_load_col = threadIdx.x * 4;

    for(int k0 = 0; k0 < K; k0 += BK) {  
        for(int i=0;i<4;++i){
            //其实就是把这里的for循环的float的加载变成使用float4去加载
            int a_row = thread_load_row + i+ block_row;
            int a_col = thread_load_col + k0;// j=0 起点
            int s_a_row = thread_load_row + i;
            int s_a_col = thread_load_col;//把下面的式子的j直接改为0，代表用float4去替代for(j=0;j<4;++j)的float加载循环
            FLOAT4(s_a[s_a_row][s_a_col])=FLOAT4(a[a_row*K+a_col]);

            int b_row = thread_load_row + i + k0;
            int b_col = thread_load_col + block_col;// j=0 起点
            int s_b_row = thread_load_row + i;
            int s_b_col = thread_load_col;//把下面的式子的j直接改为0，代表用float4去替代for(j=0;j<4;++j)的
            FLOAT4(s_b[s_b_row][s_b_col]) = FLOAT4(b[b_row*N + b_col]);

            // for(int j=0;j<4;++j){
            //     int a_row = thread_load_row + i+ block_row;
            //     int a_col = thread_load_col + j + k0;
            //     int s_a_row = thread_load_row + i;
            //     int s_a_col = thread_load_col + j;
            //     s_a[s_a_row][s_a_col] = (a_row < M && a_col < K) ? a[a_row*K + a_col] : 0.0f; // 注意边界检查

            //     int b_row = thread_load_row + i + k0;
            //     int b_col = thread_load_col + j + block_col;
            //     int s_b_row = thread_load_row + i;
            //     int s_b_col = thread_load_col + j;
            //     s_b[s_b_row][s_b_col] = (b_row < K && b_col < N) ? b[b_row*N + b_col] : 0.0f; // 注意边界检查
            // }

        }

        __syncthreads(); 

        for(int i=0;i<TM;++i){
            for(int j=0;j<TN;++j){
                float sum = 0.0f;
                for(int p=0;p<BK;++p){
                    sum += s_a[thread_compute_row + i - block_row][p] * s_b[p][thread_compute_col + j - block_col];
                }
                c_reg[i][j] += sum; 
            }
        }

        __syncthreads();
    }

    // 最后将寄存器中的结果写回全局内存
    for(int i=0;i<TM;++i){
        for(int j=0;j<TN;++j){
            int c_row = thread_compute_row + i;
            int c_col = thread_compute_col + j;
            if(c_row < M && c_col < N){
                c[c_row*N + c_col] = c_reg[i][j];
            }
        }
    }

}       







// ========== Host Helper Functions ==========
// ========== 新增：CUDA 错误检查宏 ==========
#define checkCudaErrors(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])


void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

bool validate(const std::vector<float>& ref, const std::vector<float>& gpu, int size, float atol = 1e-4, float rtol = 1e-5) {
    for (int i = 0; i < size; ++i) {
        float diff = fabsf(ref[i] - gpu[i]);
        if (diff > atol + rtol * fabsf(ref[i])) {
            printf("Mismatch at [%d]: ref=%f, gpu=%f, diff=%f\n", i, ref[i], gpu[i], diff);
            return false;
        }
    }
    return true;
}

void fill_random(std::vector<float>& v) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : v) x = dis(gen);
}


// ========== Main Test (已修改) ==========
int main() {
    const int M = 128;
    const int N = 256;
    const int K = 384;

    printf("Testing GEMM: %dx%d = %dx%d @ %dx%d\n", M, N, M, K, K, N);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(M * N);
    std::vector<float> h_C_naive(M * N);
    std::vector<float> h_C_opt(M * N);
    std::vector<float> h_C_reg(M * N);

    fill_random(h_A);
    fill_random(h_B);
    cpu_gemm(h_A, h_B, h_C_ref, M, N, K);

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc(&d_A, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_B, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_C, M * N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop);

    bool all_passed = true;

    // ------------------ 1. Test naive kernel ------------------
    dim3 block_naive(16, 16);
    dim3 grid_naive((N + block_naive.x - 1) / block_naive.x, (M + block_naive.y - 1) / block_naive.y);

    checkCudaErrors(cudaMemset(d_C, 0, M * N * sizeof(float)));
    checkCudaErrors(cudaEventRecord(start));
    sgemm_naive<<<grid_naive, block_naive>>>(d_A, d_B, d_C, M, N, K);
    // 【关键修改】检查 Kernel 启动错误
    checkCudaErrors(cudaGetLastError()); 
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);
    checkCudaErrors(cudaMemcpy(h_C_naive.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool naive_ok = validate(h_C_ref, h_C_naive, M * N);
    if (!naive_ok) all_passed = false;
    printf("Naive kernel: %s (%.2f ms)\n", naive_ok ? "PASS" : "FAIL", ms_naive);

    constexpr int BM = 32, BN = 32, BK = 32, TM = 4, TN = 4;
    dim3 block_opt(BN / TN, BM / TM);
    dim3 grid_opt((N + BN - 1) / BN, (M + BM - 1) / BM);

    // ------------------ 2. Test optimized kernel (scalar load) ------------------
    checkCudaErrors(cudaMemset(d_C, 0, M * N * sizeof(float)));
    checkCudaErrors(cudaEventRecord(start));
    sgemm_sliced_k_f32_kernel<BM, BN, BK, TM, TN><<<grid_opt, block_opt>>>(d_A, d_B, d_C, M, N, K);
    checkCudaErrors(cudaGetLastError()); // 【关键修改】
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    
    float ms_opt;
    cudaEventElapsedTime(&ms_opt, start, stop);
    checkCudaErrors(cudaMemcpy(h_C_opt.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool opt_ok = validate(h_C_ref, h_C_opt, M * N);
    if (!opt_ok) all_passed = false;
    printf("Optimized (scalar) kernel: %s (%.2f ms)\n", opt_ok ? "PASS" : "FAIL", ms_opt);

    // ------------------ 3. Test float4 + padding kernel ------------------
    checkCudaErrors(cudaMemset(d_C, 0, M * N * sizeof(float)));
    checkCudaErrors(cudaEventRecord(start));
    sgemm_sliced_k_f32x4_padding_kernel<BM, BN, BK, TM, TN><<<grid_opt, block_opt>>>(d_A, d_B, d_C, M, N, K);
    
    // 【关键修改】这里会捕获到 float4 非对齐访问的错误！
    // 如果使用 +1 padding，程序会在这里直接退出并打印 "CUDA Error illegal memory access..."
    // 这样就不会继续运行下面的 reg kernel，避免了误导
    checkCudaErrors(cudaGetLastError()); 
    
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    
    float ms_float4;
    cudaEventElapsedTime(&ms_float4, start, stop);
    checkCudaErrors(cudaMemcpy(h_C_opt.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool float4_ok = validate(h_C_ref, h_C_opt, M * N);
    if (!float4_ok) all_passed = false;
    printf("Float4+Padding kernel: %s (%.2f ms)\n", float4_ok ? "PASS" : "FAIL", ms_float4);

    // ------------------ 4. Test float4 + padding + register kernel ------------------
    // 如果上面出错了，程序已经退出了，不会执行到这里。
    // 如果上面没出错（比如你改成了+4），这里才能正常执行。
    checkCudaErrors(cudaMemset(d_C, 0, M * N * sizeof(float)));
    checkCudaErrors(cudaEventRecord(start));
    sgemm_sliced_k_f32x4_padding_reg_kernel<BM, BN, BK, TM, TN><<<grid_opt, block_opt>>>(d_A, d_B, d_C, M, N, K);
    checkCudaErrors(cudaGetLastError()); // 【关键修改】
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    
    float ms_reg;
    cudaEventElapsedTime(&ms_reg, start, stop);
    checkCudaErrors(cudaMemcpy(h_C_reg.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool reg_ok = validate(h_C_ref, h_C_reg, M * N);
    if (!reg_ok) all_passed = false;
    printf("Float4+Pad+Reg kernel: %s (%.2f ms)\n", reg_ok ? "PASS" : "FAIL", ms_reg);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (all_passed) {
        printf("\n✅ All tests passed!\n");
        return 0;
    } else {
        printf("\n❌ Some tests failed.\n");
        return 1;
    }
}