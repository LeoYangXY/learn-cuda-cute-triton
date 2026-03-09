#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <random>

// CUDA Half 类型支持
#include <cuda_fp16.h> 
// CUTE 核心
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm80.hpp>

using namespace cute;


// === GEMM Kernel (CUTE) ===



// 在 RTX 4070 (Ada) 上，最基础的 FP16/FP32 Tensor Core 指令形状是 m16n8k8。
// 这意味着在thread维度：一次指令需要：
// • A 矩阵块: 16 × 8 (M × K)
// • B 矩阵块: 8 × 8 (K × N)
// • C 矩阵块: 16 × 8 (M × N) -> 这是每个线程最终负责的结果大小。


//这里使用了隐式模板实例化的手段：AStride，ABlockLayout，AThreadLayout这些不是CUTE原生就有的数据类型，而是我们设置为模板参数，然后当作函数入参的类型
//在调用的时候，调用者根据传入的那个数据的类型进行隐式实例化
// 主要步骤如下：
// 类型推导：编译器通过观察你传给函数的实参（dA），反向推导出模板参数（AStride）应该是什么类型。你不需要手动写 <Stride<Int<1>, int>>。
// 实例化：一旦类型确定，编译器就会在后台“复制”一份函数代码，把所有的 AStride 替换成具体的 Stride<Int<1>, int>，然后编译这份新代码。
template <typename TA, typename TB, typename TC>
__global__ void gemm_device(
    int M, int N, int K,
    TA const* A, int ldA,
    TB const* B, int ldB,
    TC*       C, int ldC)
{

using namespace cute;
using X = Underscore;
using MMA_Atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
MMA_Atom mma_atom;

constexpr int BM = 128;
constexpr int BK = 16;
constexpr int BN = 128;

// 线程配置: (8, 16) -> 128 线程，每个block负责C中的(128,128)的元素
// 这样每个线程负责C中的: (128/8, 128/16) = (16, 8) -> 完美匹配 m16n8k8 指令
constexpr int THREADS_M = 8;
constexpr int THREADS_N = 16;
constexpr int TOTAL_THREADS = THREADS_M * THREADS_N; // 128

// 实际分配SMEM
__shared__ TA smemA[BM * BK];
__shared__ TB smemB[BK * BN];

// 创建 SMEM 的 Tensor 视图 (列优先布局)
// make_layout会默认使用列优先布局，因此不用写make_layout(make_shape(BK,BM),make_stride(1,BK)),直接写make_layout(make_shape(BM,BK))就行了
auto sA = make_tensor(make_smem_ptr(smemA), make_layout(make_shape(BM, BK)));
auto sB = make_tensor(make_smem_ptr(smemB), make_layout(make_shape(BK, BN)));



//   make_gmem_ptr(A): 拿到全局指针 A
//   make_shape(M,K): 告诉 CUTE，这块内存逻辑上是2D的 M 行 K 列
//   dA：比如传入了(1,ldA)
//   mA 现在代表 M × K 的矩阵。你可以用 mA(i, j) 访问相关元素，CUTE 会自动找出 A+ i*1+ j*ldA 这个地址对应的元素
//   这一步去设置好全局视图————设置好全局的shape和stride
auto wholeA = make_tensor(make_gmem_ptr(A), make_shape(M,K), make_stride(1,ldA));
auto wholeB = make_tensor(make_gmem_ptr(B), make_shape(K,N), make_stride(1,ldB));
auto wholeC = make_tensor(make_gmem_ptr(C), make_shape(M,N), make_stride(1,ldC));


//切出 Block 负责的整个长条,注意与开头的任务划分讲解部分进行对应
//local_tile(大矩阵, 小块形状, 小块在"块坐标系下"的坐标)，最后映射回去，实际的起始位置 = 块坐标 × 块形状
//wholeA是（M，K）的，make_shape的第 1 维大小是 _ (Underscore)，意思是 “继承大矩阵在该维度的剩余长度”。在这里，就是继承 K
auto wholeA_for_cur_block = local_tile(wholeA,make_shape(BM,_), make_coord(blockIdx.y, _));//在 M 维度上切分，取第 blockIdx.y 块；在 K 维度上不切分，全都要
auto wholeB_for_cur_block = local_tile(wholeB,make_shape(_,BN), make_coord(_, blockIdx.x));//在 N 维度上切分，取第 blockIdx.x 块；在 K 维度上不切分，全都要
auto wholeC_for_cur_block = local_tile(wholeC,make_shape(BM,BN), make_coord(blockIdx.y, blockIdx.x));



auto tid = threadIdx.y * THREADS_N + threadIdx.x; // 0 ~ 127 

// ==========================================================================
// 【阶段一：C 的分配】确定每个线程负责算哪一块 C (16x8)
// ==========================================================================
// 布局：(8, 16) -> 128 线程
// 每个线程负责：(128/8, 128/16) = (16, 8) -> 完美匹配 Tensor Core!
// 下面我们就用最直观的分配方式这样去讲解：对于 C：
//        N 维度 (128 列)
//        <------------------------------------------------------>
//        +--------+--------+--------+--------+ ... +--------+--------+
//  M     |        |        |        |        |     |        |        |
//  ^     |Block(0,0)|Block(0,1)|Block(0,2)|Block(0,3)| ... |Block(0,15)|
//  |     |(16 行 x8 列)|(16 行 x8 列)|(16 行 x8 列)|(16 行 x8 列)|     |(16 行 x8 列)|
//  |     |        |        |        |        |     |        |        |
//  |     +--------+--------+--------+--------+ ... +--------+--------+
//  |     |        |        |        |        |     |        |        |
//  |     |Block(1,0)|Block(1,1)|Block(1,2)|Block(1,3)| ... |Block(1,15)|
//  |     |(16 行 x8 列)|(16 行 x8 列)|(16 行 x8 列)|(16 行 x8 列)|     |(16 行 x8 列)|
//  |     |        |        |        |        |     |        |        |
//  |     +--------+--------+--------+--------+ ... +--------+--------+
//  |     |                  ... (共 8 行块) ...                       |
//  |     +--------+--------+--------+--------+ ... +--------+--------+
//  |     |        |        |        |        |     |        |        |
//  |     |Block(7,0)|Block(7,1)|Block(7,2)|Block(7,3)| ... |Block(7,15)|
//  |     |(16 行 x8 列)|(16 行 x8 列)|(16 行 x8 列)|(16 行 x8 列)|     |(16 行 x8 列)|
//  |     |        |        |        |        |     |        |        |
//  v     +--------+--------+--------+--------+ ... +--------+--------+
// (128 行)

auto c_thread_layout = make_layout(make_shape(THREADS_M, THREADS_N)); // (8, 16)
auto c_thread_coord  = make_coord(threadIdx.y, threadIdx.x);

// ==========================================================================
// 阶段二：数据的 load 阶段：从 gm->smem:
// ==========================================================================
// 对于 A：
// 总共是 128*8=1024 个元素。128 个线程平分。
// 一个很直观的分配方式就是：每个小块的大小是 (128/128, 8/1) = (1, 8)。即每人负责读取一行。
// M 维度 (128 行)
// ^
// |  [------------------ 8 cols (K 维度) ------------------]
// |  +-------------------------------------------------------+
// |  |  tid0 (Row 0)                                         |  
// |  |  (负责 1*8)                                           |
// |  |                                                       |
// |  |  tid1 (Row 1)                                         |  
// |  |  (负责 1*8)                                           |
// |  |                ...............                        |
// |  |                ...............                        |
// |  |  tid127 (Row 127)                                     |
// |  |  (负责 1*8)                                           |
// |  +-------------------------------------------------------+
auto a_load_layout = make_layout(make_shape(128, 1)); // 128 行，1 列 (逻辑上每人占一个位置)
auto a_load_coord  = make_coord(tid, 0);              // 第 tid 行

//对于 B：
// 总共是 8*128=1024 个元素。128 个线程平分。
// 一个很直观的分配方式就是：每个小块的大小是 (8/1, 128/128) = (8, 1)。即每人负责读取一列。
// K 维度 (8 行)
// ^
// |  [------------------ 128 cols (N 维度) ------------------]
// |  +-------------------------------------------------------+
// |  |  tid0   tid1   tid2  ...  tid127                      |  
// |  | (Col0) (Col1) (Col2)      (Col127)                    |
// |  | (8*1)  (8*1)  (8*1)       (8*1)                       |
// |  |                                                       |
// |  +-------------------------------------------------------+
auto b_load_layout = make_layout(make_shape(1, 128)); // 1 行，128 列
auto b_load_coord  = make_coord(0, tid);              // 第 tid 列


// ==========================================================================
// 阶段三：计算部分的分配：根据每个线程负责算哪一块 C (16x8)，找到对应的 smemA，smemB
// ==========================================================================
// 计算阶段：smem->reg
// 我们的每个 thread 为了去计算 C 中自己负责的这样一个 16*8 的小块，
// 那么它需要从 A 中拿到对应的 16*8 的小块，从 B 中拿到对应的 8*8 的小块。

// 我们发现：
// 要去算 C 的 Block(0,0) (16x8)，需要拿到 smemA 的前 16 行 (0-15)，以及 smemB 的前 8 列 (0-7)。
// 要去算 C 的 Block(0,1) (16x8)，需要拿到 smemA 的前 16 行 (0-15)，以及 smemB 的第 8-15 列。
// 要去算 C 的 Block(1,0) (16x8)，需要拿到 smemA 的第 16-31 行，以及 smemB 的前 8 列。
// 以此类推...

// 因此，画出来的划分图就是：
// SMEM A (128 rows x 8 cols)
// +-------------------------------------------------------+
// |  ROW BLOCK 0 (Rows 0-15)                              |
// |  [ Threads 0~15 (M_idx=0) 都要读取这一块 ]             |
// |  (每个线程都把这 16x8 加载到自己的寄存器)                |
// +-------------------------------------------------------+
// |  ROW BLOCK 1 (Rows 16-31)                             |
// |  [ Threads 16~31 (M_idx=1) 都要读取这一块 ]            |
// +-------------------------------------------------------+
// |  ...                                                  |
// +-------------------------------------------------------+
// |  ROW BLOCK 7 (Rows 112-127)                           |
// |  [ Threads 112~127 (M_idx=7) 都要读取这一块 ]          |
// +-------------------------------------------------------+

// SMEM B (8 rows x 128 cols)
// +------+------+------+------ ... ------+------+------+------+
// | CB 0 | CB 1 | CB 2 | ...            | CB 15| CB 16| ...  |
// | 8x8  | 8x8  | 8x8  |                | 8x8  | 8x8  |      |
// +------+------+------+------ ... ------+------+------+------+
//   ^      ^      ^                        ^      ^
//   |      |      |                        |      |
// T0~7   T8~15  T16~23                 T120~127 ...
// (每 8 个线程共享一块 8x8)


auto a_comp_layout = make_layout(make_shape(THREADS_M, 1)); // (8, 1)
auto a_comp_coord  = make_coord(threadIdx.y, 0); 

auto b_comp_layout = make_layout(make_shape(1, THREADS_N)); // (1, 16)
auto b_comp_coord  = make_coord(0, threadIdx.x); 

//我们在这里划分完了每个thread看到了自己所在的block的tensor局部视图之后要负责的任务，
//然后在循环中，我们只需要把这个block负责的A，B的tensor进行更新即可

//我们在实际开始计算之前只是用local_tile这样的去划分出每个block要负责的"小块"
//以及使用make_layout,make_coord去划分出每个thread要负责的"小小块"
//在真正开始计算的时候才是：根据block负责的小块，用local_partition+thread划分的layout和当前thread的coord去找到每个thread要负责的那个tensor
//做copy以及用mma指令的时候，需要用local_partition去找到每个thread负责的那个tensor片段才能处理

// 注意：此时只是分配空间（元数据），还没有数据搬运。
// 我们根据后续会使用到的tensor的形状，去先分配好对应的寄存器片段
auto register_fragment_A_for_mma_input = make_fragment_like(
    local_partition(sA, a_comp_layout, a_comp_coord)
);

auto register_fragment_B_for_mma_input = make_fragment_like(
    local_partition(sB, b_comp_layout, b_comp_coord)
);

auto register_fragment_C_accumulator_for_final_result = make_fragment_like(
    local_partition(wholeC_for_cur_block, c_thread_layout, c_thread_coord)
);
clear(register_fragment_C_accumulator_for_final_result);


int num_k_steps = K / BK;
for(int k_step = 0; k_step < num_k_steps; ++k_step){

  auto gA_tile_for_current_k_step = local_tile(
      wholeA_for_cur_block, 
      make_shape(BM, BK), 
      make_coord(_, k_step) // M 维度全取，K 维度取第 k_step 块
  );

  auto gB_tile_for_current_k_step = local_tile(
      wholeB_for_cur_block, 
      make_shape(BK, BN), 
      make_coord(k_step, _) // K 维度取第 k_step 块，N 维度全取
  );

  //把每个thread要负责的tensor写出来：
  auto gA_load_thread = local_partition(gA_tile_for_current_k_step, a_load_layout, a_load_coord);
  auto gB_load_thread = local_partition(gB_tile_for_current_k_step, b_load_layout, b_load_coord);
  auto sA_load_thread = local_partition(sA, a_load_layout, a_load_coord);
  auto sB_load_thread = local_partition(sB, b_load_layout, b_load_coord);
  
  // 实际做数据加载，从gm->smem
  copy(gA_load_thread, sA_load_thread);
  copy(gB_load_thread, sB_load_thread);

  __syncthreads();

  auto sA_compute_thread = local_partition(sA, a_comp_layout, a_comp_coord);
  auto sB_compute_thread = local_partition(sB, b_comp_layout, b_comp_coord);

  //从smem->reg
  copy(sA_compute_thread, register_fragment_A_for_mma_input);
  copy(sB_compute_thread, register_fragment_B_for_mma_input);

  // 现在数据都在寄存器里了，调用 Tensor Core 指令。
  // 公式：C_accumulator = A_input * B_input + C_accumulator
  mma_atom(
      register_fragment_A_for_mma_input,          // 输入 A (16x8)
      register_fragment_B_for_mma_input,          // 输入 B (8x8)
      register_fragment_C_accumulator_for_final_result // 输入/输出 C (16x8)
  );

  // 确保所有线程都算完了，才能进入下一轮循环去覆盖 SMEM 中的数据
  __syncthreads();
}

// 当前线程负责的 C 的全局视图 (用于最后写回)
auto gC_tile_for_storing_result = local_partition(
    wholeC_for_cur_block, 
    c_thread_layout, 
    c_thread_coord
);

copy(
    register_fragment_C_accumulator_for_final_result, // 源：寄存器 (16x8)
    gC_tile_for_storing_result                        // 目标：全局内存 (16x8)
);

}















//==================

// ============================================================================
// Host 包装函数
// ============================================================================
template <typename TA, typename TB, typename TC>
void gemm(int m, int n, int k,
          TA const* A, int ldA,
          TB const* B, int ldB,
          TC*       C, int ldC,
          cudaStream_t stream = 0)
{
  constexpr int BM = 128;
  constexpr int BN = 128;
  constexpr int THREADS_M = 8;
  constexpr int THREADS_N = 16;
  
  dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
  dim3 block(THREADS_N, THREADS_M); 

  gemm_device<TA, TB, TC><<<grid, block, 0, stream>>>(
      m, n, k,
      A, ldA,
      B, ldB,
      C, ldC
  );
}

// ============================================================================
// 测试函数 (关键修改在这里)
// ============================================================================
void run_gemm_mixed_precision(int m, int n, int k) {
    std::cout << "Running Mixed Precision GEMM (F16xF16->F32): " 
              << m << "x" << n << "x" << k << std::endl;

    // 1. 定义类型：输入是 half，输出是 float
    using InputType = cute::half_t; // 或者 __half
    using OutputType = float;

    // 2. 主机内存分配
    thrust::host_vector<InputType> h_A(m * k);
    thrust::host_vector<InputType> h_B(k * n);
    thrust::host_vector<OutputType> h_C(m * n, 0.0f);

    // 3. 初始化数据 (随机生成 float 然后转为 half)
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < m * k; ++i) {
        h_A[i] = InputType(dis(gen));
    }
    for (int i = 0; i < k * n; ++i) {
        h_B[i] = InputType(dis(gen));
    }

    // 4. 设备内存分配
    thrust::device_vector<InputType> d_A = h_A;
    thrust::device_vector<InputType> d_B = h_B;
    thrust::device_vector<OutputType> d_C = h_C;

    // 5. 启动 Kernel
    // 【关键修改 3】模板参数明确指定：<half, half, float>
    gemm<InputType, InputType, OutputType>(
        m, n, k,
        d_A.data().get(), m,   // ldA
        d_B.data().get(), k,   // ldB
        d_C.data().get(), m    // ldC
    );

    // 6. 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Launch Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Sync Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cout << "GEMM Completed Successfully!" << std::endl;
    
    // 打印结果 (结果是 float)
    thrust::host_vector<OutputType> h_result = d_C;
    std::cout << "First 5 elements of C (float): ";
    for(int i=0; i<5; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    int m = 1024, n = 1024, k = 1024;
    
    // 运行混合精度版本
    run_gemm_mixed_precision(m, n, k);
    
    return 0;
}