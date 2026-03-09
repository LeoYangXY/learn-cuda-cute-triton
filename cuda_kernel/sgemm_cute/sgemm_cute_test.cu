#include <cute/tensor.hpp>
using namespace cute;

// 这里是 A:(M, K), B:(N, K) 去算 C = A * B^T
template <class MShape, class NShape, class KShape,
          class TA, class TB, class TC,
          class Alpha, class Beta>
__global__ static
void
gemm_device(MShape M, NShape N, KShape K,
            TA const* A,
            TB const* B,
            TC      * C,
            Alpha alpha, Beta beta)
{
  using X = Underscore;

  //带有g的就是global memory部分；带有s的就是shared memory部分



  // Int<>这种就是编译期常量，int()这种就是运行时常量
  auto stride_gA = make_stride(Int<1>{}, M);
  auto stride_gB = make_stride(Int<1>{}, N);
  auto stride_gC = make_stride(Int<1>{}, M);

  constexpr auto BM = Int<128>{}; 
  constexpr auto BN = Int<128>{}; 
  constexpr auto BK = Int<  8>{}; 

  // 每个方向上的块数量
  int grid_m = ceil_div(M, BM); 
  int grid_n = ceil_div(N, BN); 
  int grid_k = ceil_div(K, BK); 

  // 每个block负责去算C中的(blk_idx_m,blk_idx_n)的这个tile
  int blk_idx_m = blockIdx.y; 
  int blk_idx_n = blockIdx.x; 

  //把GMEM用tensor包装
  auto gA_full = make_tensor(make_gmem_ptr(A), make_shape(M, K), stride_gA); 
  auto gB_full = make_tensor(make_gmem_ptr(B), make_shape(N, K), stride_gB); 
  auto gC_full = make_tensor(make_gmem_ptr(C), make_shape(M, N), stride_gC); 
  

  //实际分配SMEM+对其用tensor包装
  __shared__ TA smem_A[BM*BK];//在CUTE中我们一般写一维数组+用layout去逻辑上处理
  __shared__ TB smem_B[BN*BK];

  //对于make_layout如果不填写stride的话，是默认列优先布局的
  auto layout_sA = make_layout(make_shape(BM, BK)); 
  auto layout_sB = make_layout(make_shape(BN, BK));
  auto layout_sC = make_layout(make_shape(BM, BN));
  //block维度的编程：将原始的shared memory指针包装成带有定义好 Layout 的 CUTE Tensor
  auto sA = make_tensor(make_smem_ptr(smem_A), layout_sA); 
  auto sB = make_tensor(make_smem_ptr(smem_B), layout_sB); 

  //具体任务的执行分配到thread维度：从gA->sA,gB->sB的load操作
  //得到每个thread要负责的tensor
  auto thread_layout_sA_load = make_layout(make_shape(Int<32>{},Int<8>{})); 
  auto thread_layout_sB_load = make_layout(make_shape(Int<32>{}, Int<8>{}));
  auto sA_load_cur_thread = local_partition(sA,thread_layout_sA_load,threadIdx.x);
  auto sB_load_cur_thread = local_partition(sB,thread_layout_sB_load,threadIdx.x);


  //在block维度上确定任务分配：去思考每个block要负责GMEM中的哪一块&每个block要负责SMEM中的哪一块
  //local_tile可以用2D坐标，local_partition只能使用1D坐标，然后用layout去反解
  auto gC_blk_shape = make_shape(BM, BN);
  auto gC_blk_coord = make_coord(blk_idx_m, blk_idx_n);
  auto gC_cur_block = local_tile(gC_full, gC_blk_shape, gC_blk_coord);//用"块坐标"去理解local_tile
  
  //然后是要把block的任务划分到每个thread去做,注意要根据gC_cur_block这个tensor以及我们的总thread数去进行分析,比如我们这里让256个thread按照列优先的顺序铺到gC_cur_block这个tensor上
  auto thread_layout_gC = make_layout(make_shape(Int<16>{},Int<16>{}));//这里必须要使用Int<16>，不能直接写16：因为后续使用了make_fragment_like(gC_cur_thread); 这意味着这个tensor需要在编译期已知所有信息
  auto gC_cur_thread = local_partition(gC_cur_block,thread_layout_gC,threadIdx.x);//这得到的是一个tensor，只是一个视图，还没有实际数据的搬运和计算发生


  for(int k=0;k<grid_k;k++){
    auto gA_cur_block = local_tile(gA_full,make_shape(BM,BK),make_coord(blk_idx_m,k));
    auto gB_cur_block = local_tile(gB_full,make_shape(BN,BK),make_coord(k,blk_idx_n));

    //因为现在我们找到的gA_cur_block和sA_cur_block是一样的shape和stride，一种简单的分配方式就是：在局部视图上：每个thread从gA的(i,j)位置读取，写入到sA的(i,j)位置
    auto thread_layout_gA_read = thread_layout_sA_load;
    auto thread_layout_gB_read = thread_layout_sB_load;

    auto gA_read_cur_thread = local_partition(gA_cur_block,thread_layout_gA_read,threadIdx.x);
    auto gB_read_cur_thread = local_partition(gB_cur_block,thread_layout_gB_read,threadIdx.x);
    
    copy(gA_read_cur_thread,sA_load_cur_thread);
    copy(gB_read_cur_thread,sB_load_cur_thread);

    __syncthreads(); // 确保所有线程完成 SMEM 写入后再继续

    




  }






  // 7. 准备用于计算的SMEM视图 (SM -> Reg)
  // --- 用于计算 (SM -> Reg) ---
  // 使用 thread_layout_gC (MMA 布局) 确定哪些 SMEM 元素馈送给 MMA 单元。
  // Step<_1, X> 表示我们在 MMA 期间在内部遍历 SMEM 的 K 维度 (对于A)。
  // Step<X, _1> 表示我们在 MMA 期间在内部遍历 SMEM 的 K 维度 (对于B)。
  auto tCsA_compute = local_partition(sA, thread_layout_gC, threadIdx.x, Step<_1, X>{}); // [REG] 用于 MMA (A) 的 SMEM 视图
  auto tCsB_compute = local_partition(sB, thread_layout_gC, threadIdx.x, Step< X,_1>{}); // [REG] 用于 MMA (B) 的 SMEM 视图

  // ==========================================================================
  // 8. 初始化累加器 (寄存器)
  // ==========================================================================
  // 创建一个与输出 Tile 划分 (gC_cur_thread) 布局匹配的寄存器片段
  // 因为是用于分配寄存器的，因此gC_cur_thread这个tensor必须是编译期已知的
  auto tCrC_acc = make_fragment_like(gC_cur_thread); 
  clear(tCrC_acc); // 清零累加器

  // ==========================================================================
  // 9. 主 K-循环 (加载 -> 同步 -> 计算)
  // ==========================================================================
  // 策略变更：不在外部创建 3D tensor (BM, BK, GridK)，
  // 而是为每一次 k 迭代动态提取一个 2D 切片 (BM, BK)。
  // 这使得维度匹配变得显式：2D 全局切片 <-> 2D 线程布局。

  // 预先定义好 A 和 B 的切片形状和固定坐标，避免循环内重复构造 shape 对象
  auto a_blk_shape = make_shape(BM, BK);
  int blk_coord_m_A = blk_idx_m; 
  
  auto b_blk_shape = make_shape(BN, BK);
  int blk_coord_n_B = blk_idx_n;
  
  for (int k_iter = 0; k_iter < grid_k; ++k_iter)
  {
    // ----------------------------------------------------------------------
    // 步骤 A: 动态全局内存切片 (显式的 2D 视图)
    // ----------------------------------------------------------------------
    
    // 提取 A 的第 k 个 Tile: 坐标 (blk_idx_m, k_iter)
    // 结果 gA_slice 严格是 2D：形状 (128, 8)。没有隐藏维度！
    auto gA_slice = local_tile(gA_full, a_blk_shape, make_coord(blk_coord_m_A, k_iter));
    
    // 提取 B 的第 k 个 Tile: 坐标 (blk_idx_n, k_iter)
    // 结果 gB_slice 严格是 2D：形状 (128, 8)。
    auto gB_slice = local_tile(gB_full, b_blk_shape, make_coord(blk_coord_n_B, k_iter));

    // ----------------------------------------------------------------------
    // 步骤 B: 线程划分 (全局内存 -> 寄存器)
    // ----------------------------------------------------------------------
    
    // 使用预定义的线程布局，将 2D 全局切片映射到此线程的寄存器。
    // 因为输入纯粹是 2D 的，所以不存在隐式的维度保留逻辑。
    // tAgA_reg: [REG] 包含此线程从 GM 加载的 4 个元素的指针/值。
    auto tAgA_reg = local_partition(gA_slice, thread_layout_sA_load, threadIdx.x);
    auto tBgB_reg = local_partition(gB_slice, thread_layout_sB_load, threadIdx.x);

    // ----------------------------------------------------------------------
    // 步骤 C: 拷贝 全局内存 到 共享内存
    // ----------------------------------------------------------------------
    
    // 从全局寄存器 (tAgA_reg) 拷贝到共享内存寄存器 (sA_load_cur_thread)。
    // 两个 Tensor 具有兼容的形状 (源自 thread_layout_sA_load)。
    copy(tAgA_reg, sA_load_cur_thread);
    copy(tBgB_reg, sB_load_cur_thread);

    // 确保所有线程在完成写入 SMEM 后再开始读取以进行计算。
    __syncthreads();

    // ----------------------------------------------------------------------
    // 步骤 D: 在共享内存上执行矩阵乘累加 (MMA)
    // ----------------------------------------------------------------------
    
    // 在从 SMEM 加载到寄存器的数据上执行核心 GEMM 操作。
    // tCsA_compute/tCsB_compute 描述了如何遍历 SMEM 的 K 维度。
    // tCrC_acc 累加结果。
    gemm(tCsA_compute, tCsB_compute, tCrC_acc);

    // 确保所有线程在完成读取 SMEM 后，下一次迭代再覆盖它。
    __syncthreads();
  }

  // ==========================================================================
  // 10. 尾声：将结果写回全局内存
  // ==========================================================================
  // C = alpha * 累加器 + beta * 原始_C
  // gC_cur_thread 是当前线程在全局内存 C 中的视图
  axpby(alpha, tCrC_acc, beta, gC_cur_thread);
}

// =================================================================================================
// 2. Host 启动器
// =================================================================================================

template <typename TA, typename TB, typename TC,
          typename Alpha, typename Beta>
void
gemm(int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta,
     TC      * C, int ldC,
     cudaStream_t stream = 0)
{
  // 验证：此简化版 Kernel 假设紧密的列优先布局。
  if (ldA != m || ldB != n || ldC != m) {
      printf("警告：此简化版 Kernel 仅支持紧密列优先矩阵 (ld==dim)。\n");
      printf("检测到 ldA=%d (期望 %d), ldB=%d (期望 %d), ldC=%d (期望 %d)。\n", 
             ldA, m, ldB, n, ldC, m);
      // 在生产环境中，此处应返回错误或回退到通用 Kernel。
  }

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);

  constexpr auto BM = Int<128>{};
  constexpr auto BN = Int<128>{};
  
  // 线程块大小源自 C-计算布局 (16x16 = 256 线程)
  auto thread_layout_gC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(thread_layout_gC)); 
  
  // Grid 维度：(N_blocks, M_blocks) -> 注意：CUDA grid 是 (x, y)，我们将 x 映射到 N，y 映射到 M
  dim3 dimGrid(ceil_div(n, BN), ceil_div(m, BM));

  // 启动 Kernel
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
      M, N, K,
      A, B, C,
      alpha, beta);
  
  // 检查启动错误
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel 启动失败：%s\n", cudaGetErrorString(err));
      exit(1);
  }
}

// =================================================================================================
// 3. Main 函数与验证
// =================================================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// 参考 CPU GEMM: C = alpha * A * B^T + beta * C
// 假设列优先存储
void reference_gemm(int m, int n, int k, float alpha, const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int l = 0; l < k; ++l) {
                // A(i, l) * B(j, l) 因为数学上是 B 转置 (B^T)
                // 在列优先中：A[i + l*m], B[j + l*n]
                acc += A[i + l * m] * B[j + l * n];
            }
            C[i + j * m] = alpha * acc + beta * C[i + j * m];
        }
    }
}

// 验证工具函数
bool verify_result(int m, int n, const float* gpu, const float* cpu, float tol = 1e-4f) {
    int max_err_idx = -1;
    float max_err = 0.0f;
    for (int i = 0; i < m * n; ++i) {
        float diff = std::abs(gpu[i] - cpu[i]);
        if (diff > max_err) {
            max_err = diff;
            max_err_idx = i;
        }
    }
    if (max_err > tol) {
        int row = max_err_idx % m;
        int col = max_err_idx / m;
        printf("失败：最大误差 %f 位于 (%d, %d)。GPU: %f, CPU: %f\n", 
               max_err, row, col, gpu[max_err_idx], cpu[max_err_idx]);
        return false;
    }
    printf("通过：最大误差 %f 在容忍度 %f 范围内\n", max_err, tol);
    return true;
}

int main(int argc, char** argv) {
    int m = 512, n = 1024, k = 256;
    if (argc >= 2) m = atoi(argv[1]);
    if (argc >= 3) n = atoi(argv[2]);
    if (argc >= 4) k = atoi(argv[3]);

    printf("运行 GEMM 验证：M=%d, N=%d, K=%d\n", m, n, k);

    size_t size_A = m * k;
    size_t size_B = n * k;
    size_t size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C_gpu(size_C, 0.0f), h_C_cpu(size_C, 0.0f);

    // 用随机值初始化输入
    for (size_t i = 0; i < size_A; ++i) h_A[i] = (float)(rand() % 10) / 10.0f;
    for (size_t i = 0; i < size_B; ++i) h_B[i] = (float)(rand() % 10) / 10.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;

    // 启动 GPU Kernel
    // 注意：传入 ldA=m, ldB=n, ldC=m 以匹配 Kernel 的假设
    gemm(m, n, k, alpha, d_A, m, d_B, n, beta, d_C, m);
    cudaDeviceSynchronize();

    // 运行 CPU 参考实现
    reference_gemm(m, n, k, alpha, h_A.data(), h_B.data(), beta, h_C_cpu.data());

    // 拷贝结果回主机
    cudaMemcpy(h_C_gpu.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证
    bool passed = verify_result(m, n, h_C_gpu.data(), h_C_cpu.data());

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return passed ? 0 : 1;
}