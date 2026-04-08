/**
 * =============================================================================
 * 原生 CUDA HGEMM (fp16) — 对应 cutedsl_ref/sgemm.py 的各版本
 * =============================================================================
 *
 * 语义: C[M,N] = A[M,K] × B[N,K]^T  (B 转置存储, 和 CuTeDSL 版本一致)
 *
 * 版本对应关系:
 *   V1: Naive fp16 (标量 FMA, 无 Tensor Core)  — 对应 cuda_core_scalar_gemm
 *   V2: WMMA Tensor Core + SMEM tiling         — 对应 GemmWmmaVectorizedCopy
 *   V3: WMMA + cp.async multi-stage pipeline   — 对应 GemmWmmaCpAsyncMultistage
 *
 * 编译: 需要 -arch=sm_80+ (WMMA fp16 需要 SM80+)
 * =============================================================================
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_pipeline.h>
#include <stdio.h>

using namespace nvcuda;

// =============================================================================
// V1: Naive fp16 GEMM (标量 FMA, 无 Tensor Core)
// =============================================================================
// C[M,N] = A[M,K] × B[N,K]^T
// 每个线程算 C 中 1 个元素: C[row][col] = dot(A[row,:], B[col,:])
__global__ void hgemm_naive_kernel(
    const half *A, const half *B, half *C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += __half2float(A[row * K + k]) * __half2float(B[col * K + k]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

// =============================================================================
// V2: WMMA Tensor Core + SMEM Tiling
// =============================================================================
// 对应 CuTeDSL 的 GemmWmmaVectorizedCopy
//
// 分层任务划分:
//   Block 级: 每个 block 计算 C 的一个 (BM×BN) tile
//   Warp 级:  每个 warp 使用 wmma::mma_sync 计算 16×16×16 的子块
//   Thread 级: WMMA 内部 32 线程协作完成矩阵乘
//
// 数据流: GMEM → SMEM (协作加载) → WMMA fragment (硬件搬) → 累加器 → GMEM

// Tile 参数
#define BM_V2 128
#define BN_V2 128
#define BK_V2 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void hgemm_wmma_kernel(
    const half *A, const half *B, half *C,
    int M, int N, int K)
{
    // Block 负责 C[bm:bm+BM][bn:bn+BN]
    const int bm = blockIdx.y * BM_V2;
    const int bn = blockIdx.x * BN_V2;

    // SMEM: 存放一个 K-tile 的 A 和 B 数据
    // A tile: (BM, BK), B tile: (BN, BK) — B 是转置存储
    __shared__ half sA[BM_V2][BK_V2];
    __shared__ half sB[BN_V2][BK_V2];

    // Warp 在 tile 内的位置
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;  // 128 threads = 4 warps

    // 每个 warp 负责的 (WMMA_M × WMMA_N) 子块位置
    // 4 warps 覆盖 128×128: 每个 warp 需要计算多个 16×16 子块
    // warp 布局: 2×2 warp grid, 每个 warp 负责 (64×64) 区域中的 4×4=16 个 16×16 块
    const int warp_row = (warp_id / 2) * 64;  // 0 or 64
    const int warp_col = (warp_id % 2) * 64;  // 0 or 64

    // 声明 WMMA fragment 累加器 (4×4 = 16 个 16×16 子块)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // K 方向主循环
    for (int k0 = 0; k0 < K; k0 += BK_V2) {
        // ====== 协作加载 GMEM → SMEM ======
        // 128 线程协作加载 BM×BK = 128×32 = 4096 个 half (A)
        // 以及 BN×BK = 128×32 = 4096 个 half (B)
        // 每个线程加载 4096/128 = 32 个元素 (A) + 32 个元素 (B)
        for (int i = threadIdx.x; i < BM_V2 * BK_V2 / 8; i += blockDim.x) {
            int row = (i * 8) / BK_V2;
            int col = (i * 8) % BK_V2;
            int g_row = bm + row;
            int g_col = k0 + col;
            if (g_row < M && g_col + 7 < K) {
                *reinterpret_cast<int4*>(&sA[row][col]) =
                    *reinterpret_cast<const int4*>(&A[g_row * K + g_col]);
            } else {
                for (int x = 0; x < 8; x++) {
                    int gr = g_row, gc = g_col + x;
                    sA[row][col + x] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
                }
            }
        }
        for (int i = threadIdx.x; i < BN_V2 * BK_V2 / 8; i += blockDim.x) {
            int row = (i * 8) / BK_V2;
            int col = (i * 8) % BK_V2;
            int g_row = bn + row;
            int g_col = k0 + col;
            if (g_row < N && g_col + 7 < K) {
                *reinterpret_cast<int4*>(&sB[row][col]) =
                    *reinterpret_cast<const int4*>(&B[g_row * K + g_col]);
            } else {
                for (int x = 0; x < 8; x++) {
                    int gr = g_row, gc = g_col + x;
                    sB[row][col + x] = (gr < N && gc < K) ? B[gr * K + gc] : __float2half(0.0f);
                }
            }
        }
        __syncthreads();

        // ====== WMMA 计算 ======
        // 每个 warp 在其 64×64 区域内迭代 4×4 个 16×16 子块
        for (int ik = 0; ik < BK_V2; ik += WMMA_K) {
            for (int wi = 0; wi < 4; wi++) {
                for (int wj = 0; wj < 4; wj++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;

                    // A fragment: sA[warp_row + wi*16 : +16][ik : +16]
                    wmma::load_matrix_sync(frag_a, &sA[warp_row + wi * WMMA_M][ik],
                                           BK_V2);  // ldm = BK (row-major stride)

                    // B fragment: sB[warp_col + wj*16 : +16][ik : +16]
                    // B 在 SMEM 中是 (BN, BK) row-major, 但我们要的是 B^T
                    // 所以 B 按行存储, load 为 col_major 等价于转置
                    wmma::load_matrix_sync(frag_b, &sB[warp_col + wj * WMMA_N][ik],
                                           BK_V2);

                    wmma::mma_sync(acc[wi][wj], frag_a, frag_b, acc[wi][wj]);
                }
            }
        }
        __syncthreads();
    }

    // ====== 写回 GMEM ======
    for (int wi = 0; wi < 4; wi++) {
        for (int wj = 0; wj < 4; wj++) {
            int c_row = bm + warp_row + wi * WMMA_M;
            int c_col = bn + warp_col + wj * WMMA_N;
            if (c_row < M && c_col < N) {
                // 先存到 fp16 fragment 再写
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_half;
                for (int t = 0; t < acc[wi][wj].num_elements; t++)
                    acc_half.x[t] = __float2half(acc[wi][wj].x[t]);
                wmma::store_matrix_sync(&C[c_row * N + c_col], acc_half, N, wmma::mem_row_major);
            }
        }
    }
}

// =============================================================================
// V3: WMMA + cp.async Multi-Stage Pipeline
// =============================================================================
// 对应 CuTeDSL 的 GemmWmmaCpAsyncMultistage
//
// 在 V2 基础上加入:
//   1. cp.async: GMEM→SMEM 异步拷贝 (绕过寄存器)
//   2. 3-stage SMEM buffer 轮转: Prologue-MainLoop-Epilogue 三段式
//   3. commit_group / wait_group 控制流水线深度
//
// 软件流水线结构:
//   Prologue: 发射 stage-1 个 cp.async, 灌满管线
//   MainLoop: 发射新拷贝 + wait + WMMA 计算 (三层重叠)
//   Epilogue: 排空管线, 完成剩余计算

#define BM_V3 128
#define BN_V3 128
#define BK_V3 32
#define NUM_STAGES 3

__global__ void hgemm_wmma_cpasync_kernel(
    const half *A, const half *B, half *C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * BM_V3;
    const int bn = blockIdx.x * BN_V3;

    // 3-stage SMEM buffer
    __shared__ half sA[NUM_STAGES][BM_V3][BK_V3];
    __shared__ half sB[NUM_STAGES][BN_V3][BK_V3];

    const int warp_id = threadIdx.x / 32;
    const int warp_row = (warp_id / 2) * 64;
    const int warp_col = (warp_id % 2) * 64;

    // 累加器
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int num_k_tiles = K / BK_V3;

    // 辅助: cp.async 加载一个 stage
    auto load_stage = [&](int stage, int k_tile) {
        int k0 = k_tile * BK_V3;
        // 加载 A: (BM, BK) = 128×32 = 4096 half, 每线程搬 128-bit = 8 half
        for (int i = threadIdx.x; i < BM_V3 * BK_V3 / 8; i += blockDim.x) {
            int row = (i * 8) / BK_V3;
            int col = (i * 8) % BK_V3;
            int g_row = bm + row;
            int g_col = k0 + col;
            __pipeline_memcpy_async(
                &sA[stage][row][col],
                &A[g_row * K + g_col],
                16);  // 16 bytes = 8 half
        }
        // 加载 B: (BN, BK)
        for (int i = threadIdx.x; i < BN_V3 * BK_V3 / 8; i += blockDim.x) {
            int row = (i * 8) / BK_V3;
            int col = (i * 8) % BK_V3;
            int g_row = bn + row;
            int g_col = k0 + col;
            __pipeline_memcpy_async(
                &sB[stage][row][col],
                &B[g_row * K + g_col],
                16);
        }
        __pipeline_commit();
    };

    // 辅助: 对一个 stage 做 WMMA 计算
    auto compute_stage = [&](int stage) {
        for (int ik = 0; ik < BK_V3; ik += WMMA_K) {
            for (int wi = 0; wi < 4; wi++) {
                for (int wj = 0; wj < 4; wj++) {
                    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;
                    wmma::load_matrix_sync(frag_a, &sA[stage][warp_row + wi * WMMA_M][ik], BK_V3);
                    wmma::load_matrix_sync(frag_b, &sB[stage][warp_col + wj * WMMA_N][ik], BK_V3);
                    wmma::mma_sync(acc[wi][wj], frag_a, frag_b, acc[wi][wj]);
                }
            }
        }
    };

    // ====== Prologue: 发射 stage-1 = 2 个异步拷贝 ======
    for (int s = 0; s < NUM_STAGES - 1 && s < num_k_tiles; s++) {
        load_stage(s, s);
    }

    // ====== MainLoop ======
    for (int kidx = 0; kidx < num_k_tiles; kidx++) {
        int stage = kidx % NUM_STAGES;

        // ① 发射下一个 Tile 的异步拷贝 (和当前计算重叠)
        int next_load = kidx + NUM_STAGES - 1;
        if (next_load < num_k_tiles) {
            load_stage(next_load % NUM_STAGES, next_load);
        }

        // ② wait: 确保当前 stage 数据就绪
        __pipeline_wait_prior(NUM_STAGES - 2);
        __syncthreads();

        // ③ WMMA 计算
        compute_stage(stage);

        __syncthreads();
    }

    // ====== 写回 GMEM ======
    for (int wi = 0; wi < 4; wi++) {
        for (int wj = 0; wj < 4; wj++) {
            int c_row = bm + warp_row + wi * WMMA_M;
            int c_col = bn + warp_col + wj * WMMA_N;
            if (c_row < M && c_col < N) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_half;
                for (int t = 0; t < acc[wi][wj].num_elements; t++)
                    acc_half.x[t] = __float2half(acc[wi][wj].x[t]);
                wmma::store_matrix_sync(&C[c_row * N + c_col], acc_half, N, wmma::mem_row_major);
            }
        }
    }
}


// =============================================================================
// V4: WMMA + cp.async + Warp Specialization + Double Buffer (SM80+)
// =============================================================================
// 对应 CuTeDSL 的 GemmWmmaTmaWarpSpecializedPipeline
//
// 核心思想: Warp Specialization (异构分工)
//   Warp 0~3 (128线程): MMA consumer — 全职计算
//   Warp 4   (32 线程):  producer — 发起 cp.async 搬数据
//   所有线程通过 __syncthreads 协调
//
// 架构限制说明:
//   在原生 CUDA 中, 真正的 warp specialization (producer/consumer 完全异步并行)
//   需要 mbarrier (SM90+) 或 TMA 硬件。这里用 __syncthreads 简化了同步,
//   导致 producer 和 consumer 实际不能完全并行, 性能不如 V3。
//
//   真正高性能的 warp specialization 需要:
//     - SM90+ TMA 硬件: 搬运不占线程, producer warp 几乎无开销
//     - mbarrier: 细粒度 per-stage 同步, 不需要全局 barrier
//     - 这正是 CuTeDSL V11 (PipelineTmaAsync) 做的事
//
//   本版本的教学意义: 展示 warp 角色分离的代码结构,
//   理解了结构后, 用 CuTeDSL/CUTLASS 3.x 可以轻松获得真正的性能。

#define BM_V4 128
#define BN_V4 128
#define BK_V4 32
#define NUM_STAGES_V4 2
#define NUM_MMA_WARPS 4
#define TMA_WARP_ID 4
#define THREADS_V4 160  // 5 warps = 160 threads

__global__ void hgemm_wmma_warp_specialized_kernel(
    const half *A, const half *B, half *C,
    int M, int N, int K)
{
    const int bm = blockIdx.y * BM_V4;
    const int bn = blockIdx.x * BN_V4;
    const int warp_id = threadIdx.x / 32;
    const int tid_in_block = threadIdx.x;

    const bool is_producer = (warp_id == TMA_WARP_ID);
    const bool is_consumer = (warp_id < NUM_MMA_WARPS);

    // 2-stage SMEM buffer
    __shared__ half sA[NUM_STAGES_V4][BM_V4][BK_V4];
    __shared__ half sB[NUM_STAGES_V4][BN_V4][BK_V4];

    const int num_k_tiles = K / BK_V4;

    // Consumer: WMMA 累加器
    int warp_row = 0, warp_col = 0;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[4][4];
    if (is_consumer) {
        warp_row = (warp_id / 2) * 64;
        warp_col = (warp_id % 2) * 64;
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                wmma::fill_fragment(acc[i][j], 0.0f);
    }

    // ====== Prologue: producer 加载第一个 stage ======
    {
        int stage = 0, k0 = 0;
        // 所有 160 线程协作加载 (利用所有带宽)
        for (int i = tid_in_block; i < BM_V4 * BK_V4 / 8; i += THREADS_V4) {
            int row = (i * 8) / BK_V4;
            int col = (i * 8) % BK_V4;
            __pipeline_memcpy_async(&sA[stage][row][col], &A[(bm + row) * K + k0 + col], 16);
        }
        for (int i = tid_in_block; i < BN_V4 * BK_V4 / 8; i += THREADS_V4) {
            int row = (i * 8) / BK_V4;
            int col = (i * 8) % BK_V4;
            __pipeline_memcpy_async(&sB[stage][row][col], &B[(bn + row) * K + k0 + col], 16);
        }
        __pipeline_commit();
    }

    // ====== MainLoop: producer 预取下一个 stage, consumer 计算当前 stage ======
    for (int kidx = 0; kidx < num_k_tiles; kidx++) {
        int cur_stage = kidx % NUM_STAGES_V4;
        int next_stage = (kidx + 1) % NUM_STAGES_V4;
        int next_k0 = (kidx + 1) * BK_V4;

        // 等待当前 stage 的 cp.async 完成
        __pipeline_wait_prior(0);
        __syncthreads();

        // Producer: 预取下一个 Tile (和 consumer 计算重叠)
        if (kidx + 1 < num_k_tiles) {
            // 只用 producer warp + 部分 consumer 线程协作加载
            // (所有线程参与可以最大化带宽)
            for (int i = tid_in_block; i < BM_V4 * BK_V4 / 8; i += THREADS_V4) {
                int row = (i * 8) / BK_V4;
                int col = (i * 8) % BK_V4;
                __pipeline_memcpy_async(&sA[next_stage][row][col], &A[(bm + row) * K + next_k0 + col], 16);
            }
            for (int i = tid_in_block; i < BN_V4 * BK_V4 / 8; i += THREADS_V4) {
                int row = (i * 8) / BK_V4;
                int col = (i * 8) % BK_V4;
                __pipeline_memcpy_async(&sB[next_stage][row][col], &B[(bn + row) * K + next_k0 + col], 16);
            }
            __pipeline_commit();
        }

        // Consumer: WMMA 计算当前 stage
        if (is_consumer) {
            for (int ik = 0; ik < BK_V4; ik += WMMA_K) {
                for (int wi = 0; wi < 4; wi++) {
                    for (int wj = 0; wj < 4; wj++) {
                        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_a;
                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_b;
                        wmma::load_matrix_sync(frag_a, &sA[cur_stage][warp_row + wi * WMMA_M][ik], BK_V4);
                        wmma::load_matrix_sync(frag_b, &sB[cur_stage][warp_col + wj * WMMA_N][ik], BK_V4);
                        wmma::mma_sync(acc[wi][wj], frag_a, frag_b, acc[wi][wj]);
                    }
                }
            }
        }

        __syncthreads();
    }

    // ====== 写回 GMEM (只有 consumer warps) ======
    if (is_consumer) {
        for (int wi = 0; wi < 4; wi++) {
            for (int wj = 0; wj < 4; wj++) {
                int c_row = bm + warp_row + wi * WMMA_M;
                int c_col = bn + warp_col + wj * WMMA_N;
                if (c_row < M && c_col < N) {
                    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_half;
                    for (int t = 0; t < acc[wi][wj].num_elements; t++)
                        acc_half.x[t] = __float2half(acc[wi][wj].x[t]);
                    wmma::store_matrix_sync(&C[c_row * N + c_col], acc_half, N, wmma::mem_row_major);
                }
            }
        }
    }
}


// =============================================================================
// Torch bindings
// =============================================================================
#include <torch/types.h>
#include <torch/extension.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CEIL(a,b) ((a+b-1)/(b))

// V1: Naive fp16 GEMM
// A: [M, K], B: [N, K] (B transposed) -> C: [M, N]
torch::Tensor torch_hgemm_naive(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(0);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid(CEIL(N, 16), CEIL(M, 16));
    hgemm_naive_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K);
    return C;
}

// V2: WMMA Tensor Core
torch::Tensor torch_hgemm_wmma(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(0);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(128);  // 4 warps
    dim3 grid(CEIL(N, BN_V2), CEIL(M, BM_V2));
    hgemm_wmma_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K);
    return C;
}

// V3: WMMA + cp.async pipeline
torch::Tensor torch_hgemm_wmma_cpasync(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(0);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(128);  // 4 warps
    dim3 grid(CEIL(N, BN_V3), CEIL(M, BM_V3));
    hgemm_wmma_cpasync_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K);
    return C;
}

// V4: WMMA + Warp Specialization + mbarrier pipeline (SM90+)
torch::Tensor torch_hgemm_wmma_warp_specialized(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(0);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(THREADS_V4);  // 5 warps = 160 threads
    dim3 grid(CEIL(N, BN_V4), CEIL(M, BM_V4));
    hgemm_wmma_warp_specialized_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_hgemm_naive)
    TORCH_BINDING_COMMON_EXTENSION(torch_hgemm_wmma)
    TORCH_BINDING_COMMON_EXTENSION(torch_hgemm_wmma_cpasync)
    TORCH_BINDING_COMMON_EXTENSION(torch_hgemm_wmma_warp_specialized)
}
