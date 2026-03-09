/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#include <cute/tensor.hpp>

using namespace cute;

//这里是A：（M，K） B：（N，K） 去算C=A*B^T
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

  // 1. Define strides (使用传入的 M, N, K 动态计算，注意：这里假设 ldA=M, ldB=N, ldC=M)
  // 如果 Host 端传入的是非紧密矩阵 (ldA != M)，这里计算出的结果将是错误的！
  auto dA = make_stride(Int<1>{}, M);
  auto dB = make_stride(Int<1>{}, N);
  auto dC = make_stride(Int<1>{}, M);

  // 2. Define block sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int<  8>{};

int grid_m = ceil_div(M, bM); // M 方向的块数
int grid_n = ceil_div(N, bN); // N 方向的块数
int grid_k = ceil_div(K, bK); // K 方向的块数 (用于循环次数)

// 当前 Block 负责的计算位置 (固定不变)
int m_idx = blockIdx.y; // 第几个 M 块
int n_idx = blockIdx.x; // 第几个 N 块



  // 3. Define Layouts (这是关键：先定义 Layout 对象)
  auto layout_sA = make_layout(make_shape(bM, bK)); 
  auto layout_sB = make_layout(make_shape(bN, bK));
  auto layout_sC = make_layout(make_shape(bM, bN));

  auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}),make_stride(Int<1>{},Int<32>{}));
  auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}),make_stride(Int<1>{},Int<32>{}));
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}),make_stride(Int<1>{},Int<16>{}));

  // 4. Shared memory buffers
  __shared__ TA smemA[cosize_v<decltype(layout_sA)>];
  __shared__ TB smemB[cosize_v<decltype(layout_sB)>];
  
  auto sA = make_tensor(make_smem_ptr(smemA), layout_sA); 
  auto sB = make_tensor(make_smem_ptr(smemB), layout_sB);

  // 5. Represent the full tensors
  auto mA = make_tensor(make_gmem_ptr(A), make_shape(M,K), dA);
  auto mB = make_tensor(make_gmem_ptr(B), make_shape(N,K), dB);
  auto mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);


auto c_blk_shape = make_shape(bM, bN);
auto c_blk_coord = make_coord(m_idx, n_idx);
// gC: 全局内存中属于当前 Block 的那一块 (BM x BN)
auto gC = local_tile(mC, c_blk_shape, c_blk_coord); 
//只接受scaler作为坐标，然后local_partition会使用tC这个layout去进行划分的：tC是shape(16,16),stride(1,16)的，
//逻辑上是：第一列对应tid为0~17，第2列对应tid为18~35.那么比如threadIdx.x=18的话，就会到第1列的第2行
auto tCgC = local_partition(gC, tC, threadIdx.x);

//   // 6. Get the appropriate blocks
auto a_blk_shape = make_shape(bM, bK);
// Coord: (blockIdx.y, _) 
//   - blockIdx.y: 固定 M 方向的块索引
//   - _: 表示 K 方向要切分出所有可能的块，形成一个新的维度!!
auto a_blk_coord = make_coord(blockIdx.y, _);//这里的_代表我们拿走切出来的所有块，最后其实就是拿走A的多个行
auto gA = local_tile(mA, a_blk_shape, a_blk_coord);
//按照a_blk_shape去分片，也就是划分为BM*BK的小片。不过我们使用了"_"这个去取出所有的小片
//那么也就是相当于gA=[小片0，小片1，小片2...]，也就是(BM,BK,grid_k)这样的shape
// 结果 gA 的形状: (bM, bK, num_k_blocks)

// --- B 的切分 ---
// Shape: (bN, bK)
auto b_blk_shape = make_shape(bN, bK);
// Coord: (blockIdx.x, _)
//   - blockIdx.x: 固定 N 方向的块索引
//   - _: 遍历 K
auto b_blk_coord = make_coord(blockIdx.x, _);

auto gB = local_tile(mB, b_blk_shape, b_blk_coord);


  // 7. Partition copying

// 调用 local_partition(gA, tA, tid) 时的维度匹配逻辑：
// 1. 输入定义：
//    - tA: 2D Layout (形状 S0 x S1)，描述线程在二维平面上的分布。
//    - gA: 3D Tensor (形状 D0 x D1 x D2)。
//
// 2. 匹配过程（从前往后/按逻辑顺序）：
//    - CUTE 尝试用 tA 覆盖 gA 的前两个维度 (D0, D1)。
//    - tA 定义的线程布局将“消耗”掉这两个维度，将其转换为每个线程本地的寄存器视图。
//
// 3. 剩余维度处理：
//    - gA 的第 3 维 (D2，通常对应 K 轴的块索引) 未被 tA 覆盖。
//    - 该维度将被原封不动地保留在结果 Tensor (tAgA) 中，作为最高维或剩余维存在。
  auto tAgA = local_partition(gA, tA, threadIdx.x);
  auto tAsA = local_partition(sA, tA, threadIdx.x);

  auto tBgB = local_partition(gB, tB, threadIdx.x);
  auto tBsB = local_partition(sB, tB, threadIdx.x);

  // 8. Partition for compute
  auto tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
  auto tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});


  // 9. Accumulators
  auto tCrC = make_fragment_like(tCgC);
  clear(tCrC);

  // 10. Compute Loop
  auto k_max = size<2>(tAgA);

  for (int k = 0; k < k_max; ++k)
  {
    // Copy gmem to smem
    copy(tAgA(_,_,k), tAsA);
    copy(tBgB(_,_,k), tBsB);

    // 注意：普通的 copy 不需要 cp_async_fence。
    // 如果要用 async copy，需要配置 copy_atom。这里为了简单和正确性，只用 syncthreads。
    __syncthreads();

    // Compute gemm on smem
    gemm(tCsA, tCsB, tCrC);

    __syncthreads();
  }

  // Epilogue
  axpby(alpha, tCrC, beta, tCgC);
}

// =================================================================================================
// 2. Host Launcher
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
  // 警告：这个简化版 Kernel 强制假设 ldA==m, ldB==n, ldC==m。
  // 如果传入的 ldA != m，计算结果将错误。
  if (ldA != m || ldB != n || ldC != m) {
      printf("Warning: This simplified kernel only supports column-major tight matrices (ld==dim).\n");
      printf("Detected ldA=%d (expected %d), ldB=%d (expected %d), ldC=%d (expected %d).\n", 
             ldA, m, ldB, n, ldC, m);
      // 在实际应用中应在此处报错或回退到通用版本
  }

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);

  auto bM = Int<128>{};
  auto bN = Int<128>{};
  
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

  dim3 dimBlock(size(tC)); // 256 threads
  dim3 dimGrid(ceil_div(n, bN), ceil_div(m, bM));

  // 调用时只传指针，不传 layout 对象
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
      M, N, K,
      A, B, C,
      alpha, beta);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
      printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
      exit(1);
  }
}

// =================================================================================================
// 3. Main & Verification (保持不变)
// =================================================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

void reference_gemm(int m, int n, int k, float alpha, const float* A, const float* B, float beta, float* C) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int l = 0; l < k; ++l) {
                acc += A[i + l * m] * B[j + l * n];
            }
            C[i + j * m] = alpha * acc + beta * C[i + j * m];
        }
    }
}

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
        printf("FAIL: Max error %f at (%d, %d). GPU: %f, CPU: %f\n", 
               max_err, row, col, gpu[max_err_idx], cpu[max_err_idx]);
        return false;
    }
    printf("PASS: Max error %f within tolerance %f\n", max_err, tol);
    return true;
}

int main(int argc, char** argv) {
    int m = 512, n = 1024, k = 256;
    if (argc >= 2) m = atoi(argv[1]);
    if (argc >= 3) n = atoi(argv[2]);
    if (argc >= 4) k = atoi(argv[3]);

    printf("Running GEMM Verification: M=%d, N=%d, K=%d\n", m, n, k);

    size_t size_A = m * k;
    size_t size_B = n * k;
    size_t size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C_gpu(size_C, 0.0f), h_C_cpu(size_C, 0.0f);

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

    // 注意：这里传入的 ldA=m, ldB=n, ldC=m 符合简化版 Kernel 的硬编码假设
    gemm(m, n, k, alpha, d_A, m, d_B, n, beta, d_C, m);
    cudaDeviceSynchronize();

    reference_gemm(m, n, k, alpha, h_A.data(), h_B.data(), beta, h_C_cpu.data());

    cudaMemcpy(h_C_gpu.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    bool passed = verify_result(m, n, h_C_gpu.data(), h_C_cpu.data());

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return passed ? 0 : 1;
}