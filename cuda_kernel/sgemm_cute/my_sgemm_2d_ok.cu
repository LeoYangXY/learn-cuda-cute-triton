/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * GEMM Kernel: C = alpha * A * B^T + beta * C
 * 内存布局假设：列优先 (Column-Major)，即 ldA=M, ldB=N, ldC=M
 * 
 * 数据流向策略：
 * 1. 全局内存 (GMEM) -> 共享内存 (SMEM): 在 K 循环中分块加载
 * 2. 共享内存 (SMEM) -> 寄存器 (Reg): 为计算进行线程划分
 * 3. 计算：在寄存器上执行矩阵乘法
 **************************************************************************************************/
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

  // ==========================================================================
  // 1. 定义步长 (全局内存布局)
  // ==========================================================================
  // 假设列优先存储 (Column-Major): stride(1, LeadingDim)
  // 注意：这里硬编码了 ldA=M, ldB=N, ldC=M。如果 Host 传入非紧密矩阵，结果将错误。
  auto stride_gA = make_stride(Int<1>{}, M);
  auto stride_gB = make_stride(Int<1>{}, N);
  auto stride_gC = make_stride(Int<1>{}, M);

  // ==========================================================================
  // 2. 定义块大小 (Tile 维度)
  // ==========================================================================
  constexpr auto blk_m = Int<128>{}; // Block M 维度
  constexpr auto blk_n = Int<128>{}; // Block N 维度
  constexpr auto blk_k = Int<  8>{}; // Block K 维度 (内部 Tile)

  // Grid 维度 (每个方向上的块数量)
  int grid_m = ceil_div(M, blk_m); 
  int grid_n = ceil_div(N, blk_n); 
  int grid_k = ceil_div(K, blk_k); 

  // 当前 Block 的索引 (在整个 Kernel 运行期间固定不变)
  int blk_idx_m = blockIdx.y; // 此 Block 负责计算第几个 M-Tile
  int blk_idx_n = blockIdx.x; // 此 Block 负责计算第几个 N-Tile

  // ==========================================================================
  // 3. 定义线程布局 (线程如何映射到 Tile)
  // ==========================================================================
  // 共享内存缓冲区的 Layout (Tile 的物理形状)
  auto layout_sA = make_layout(make_shape(blk_m, blk_k)); 
  auto layout_sB = make_layout(make_shape(blk_n, blk_k));
  auto layout_sC = make_layout(make_shape(blk_m, blk_n));

  // 线程分布 Layout (256 个线程到 Tile 的逻辑映射)
  // tA/tB: 32x8 布局，stride(1, 32)。将 256 个线程映射到 128x8 的 Tile。
  //        每个线程负责 4 个元素。
  auto thread_layout_A = make_layout(make_shape(Int<32>{}, Int< 8>{}), make_stride(Int<1>{}, Int<32>{}));
  auto thread_layout_B = make_layout(make_shape(Int<32>{}, Int< 8>{}), make_stride(Int<1>{}, Int<32>{}));
  
  // tC: 16x16 布局。将 256 个线程映射到 128x128 的输出 Tile。
  //     每个线程负责 4 个元素用于累加。
  auto thread_layout_C = make_layout(make_shape(Int<16>{}, Int<16>{}), make_stride(Int<1>{}, Int<16>{}));

  // ==========================================================================
  // 4. 分配共享内存缓冲区
  // ==========================================================================
  __shared__ TA smem_A[cosize_v<decltype(layout_sA)>];
  __shared__ TB smem_B[cosize_v<decltype(layout_sB)>];
  
  // 将原始指针包装成带有定义好 Layout 的 CUTE Tensor
  auto sA = make_tensor(make_smem_ptr(smem_A), layout_sA); // [SMEM] 形状：(128, 8)
  auto sB = make_tensor(make_smem_ptr(smem_B), layout_sB); // [SMEM] 形状：(128, 8)

  // ==========================================================================
  // 5. 表示完整的全局内存 Tensor
  // ==========================================================================
  auto gA_full = make_tensor(make_gmem_ptr(A), make_shape(M, K), stride_gA); // [GMEM] 形状：(M, K)
  auto gB_full = make_tensor(make_gmem_ptr(B), make_shape(N, K), stride_gB); // [GMEM] 形状：(N, K)
  auto gC_full = make_tensor(make_gmem_ptr(C), make_shape(M, N), stride_gC); // [GMEM] 形状：(M, N)

  // ==========================================================================
  // 6. 提取输出 Tile (C 的全局内存视图)
  // ==========================================================================
  // 从全局内存中提取分配给此 Block 的特定 (blk_m x blk_n) Tile
  auto c_blk_shape = make_shape(blk_m, blk_n);
  auto c_blk_coord = make_coord(blk_idx_m, blk_idx_n);
  
  // gC: [GMEM] 输出 Tile 的视图。形状：(128, 128)
  auto gC = local_tile(gC_full, c_blk_shape, c_blk_coord); 
  
  // 为此线程划分 gC：
  // 使用 thread_layout_C 将 2D 全局 Tile (128x128) 映射到此线程的寄存器。
  // 结果 tCgC 是一个寄存器 tensor，持有此线程负责的约 4 个元素。
  auto tCgC = local_partition(gC, thread_layout_C, threadIdx.x);

  // ==========================================================================
  // 7. 准备输入 Tile (A 和 B) 的静态坐标
  // ==========================================================================
  // 我们将在循环中动态切片这些 Tensor，但 M/N 坐标是固定的。
  
  // A: 固定的 M 坐标，变化的 K 坐标 (在循环中)
  auto a_blk_shape = make_shape(blk_m, blk_k);
  int blk_coord_m_A = blk_idx_m; 

  // B: 固定的 N 坐标，变化的 K 坐标 (在循环中)
  auto b_blk_shape = make_shape(blk_n, blk_k);
  int blk_coord_n_B = blk_idx_n;

  // ==========================================================================
  // 8. 划分共享内存视图 (静态部分)
  // ==========================================================================
  // 由于共享内存布局 (sA, sB) 是恒定的，我们可以预先计算
  // 每个线程如何访问 SMEM 以进行“加载”和“计算”。
  
  // --- 用于加载 (GM -> SM) ---
  // 使用 thread_layout_A/B 确定此线程写入哪些 SMEM 地址。
  auto tAsA_load = local_partition(sA, thread_layout_A, threadIdx.x); // [REG] 用于加载 A 的 SMEM 视图
  auto tBsB_load = local_partition(sB, thread_layout_B, threadIdx.x); // [REG] 用于加载 B 的 SMEM 视图

  // --- 用于计算 (SM -> Reg) ---
  // 使用 thread_layout_C (MMA 布局) 确定哪些 SMEM 元素馈送给 MMA 单元。
  // Step<_1, X> 表示我们在 MMA 期间在内部遍历 SMEM 的 K 维度。
  auto tCsA_compute = local_partition(sA, thread_layout_C, threadIdx.x, Step<_1, X>{}); // [REG] 用于 MMA (A) 的 SMEM 视图
  auto tCsB_compute = local_partition(sB, thread_layout_C, threadIdx.x, Step< X,_1>{}); // [REG] 用于 MMA (B) 的 SMEM 视图

  // ==========================================================================
  // 9. 初始化累加器 (寄存器)
  // ==========================================================================
  // 创建一个与输出 Tile 划分 (tCgC) 布局匹配的寄存器片段
  auto tCrC_acc = make_fragment_like(tCgC); 
  clear(tCrC_acc); // 清零累加器

  // ==========================================================================
  // 10. 主 K-循环 (加载 -> 同步 -> 计算)
  // ==========================================================================
  // 策略变更：不在外部创建 3D tensor (BM, BK, GridK)，
  // 而是为每一次 k 迭代动态提取一个 2D 切片 (BM, BK)。
  // 这使得维度匹配变得显式：2D 全局切片 <-> 2D 线程布局。
  
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
    auto tAgA_reg = local_partition(gA_slice, thread_layout_A, threadIdx.x);
    auto tBgB_reg = local_partition(gB_slice, thread_layout_B, threadIdx.x);

    // ----------------------------------------------------------------------
    // 步骤 C: 拷贝 全局内存 到 共享内存
    // ----------------------------------------------------------------------
    
    // 从全局寄存器 (tAgA_reg) 拷贝到共享内存寄存器 (tAsA_load)。
    // 两个 Tensor 具有兼容的形状 (源自 thread_layout_A)。
    copy(tAgA_reg, tAsA_load);
    copy(tBgB_reg, tBsB_load);

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
  // 11. 尾声：将结果写回全局内存
  // ==========================================================================
  // C = alpha * 累加器 + beta * 原始_C
  axpby(alpha, tCrC_acc, beta, tCgC);
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

  constexpr auto blk_m = Int<128>{};
  constexpr auto blk_n = Int<128>{};
  
  // 线程块大小源自 C-计算布局 (16x16 = 256 线程)
  auto thread_layout_C = make_layout(make_shape(Int<16>{}, Int<16>{}));
  dim3 dimBlock(size(thread_layout_C)); 
  
  // Grid 维度：(N_blocks, M_blocks) -> 注意：CUDA grid 是 (x, y)，我们将 x 映射到 N，y 映射到 M
  dim3 dimGrid(ceil_div(n, blk_n), ceil_div(m, blk_m));

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