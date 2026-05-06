#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <torch/types.h>
#include <torch/extension.h>

#define CEIL(a, b) ((a + b - 1) / (b))

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
  if (((T).options().dtype() != (th_type))) {                \
    std::cout << "Tensor Info:" << (T).options() << std::endl; \
    throw std::runtime_error("values must be " #th_type);    \
  }

// ==================== Graph Algorithm Kernels ====================
// All-Pairs Shortest Path (Floyd-Warshall)
// Single Source Shortest Path (Bellman-Ford style)
//
// 【任务划分逻辑】
//
// ▸ Floyd-Warshall (All-Pairs Shortest Path):
//   算法: for k=0..N-1: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
//   天然的外循环依赖 (k 依赖 k-1)，但内层 (i,j) 完全并行
//
//   - 外循环 k: 在 host 端控制，每轮启动一次 kernel
//   - 内层: grid = (ceil(N/16), ceil(N/16)), block = (16, 16)
//     每个 thread 负责更新 dist[i][j] 一个元素
//   - 总复杂度 O(N³)，但内层 O(N²) 完全并行
//
//   Tiled 优化 (Phase 1):
//   - 将距离矩阵按 FW_TILE×FW_TILE 分块
//   - 对角块的自更新: 加载 tile 到 smem → 在 smem 中做小 Floyd-Warshall
//   - 减少全局内存访问 (tile 内的 K 步在 smem 中完成)
//
// ▸ Bellman-Ford (Single Source Shortest Path):
//   算法: 重复 N-1 次松弛所有边 → 收敛到最短路径
//   每次松弛完全独立 → 一次 kernel 松弛所有边
//
//   - grid = ceil(num_edges / blockDim.x), 每个 thread 负责 1 条边
//   - 松弛: if dist[u] + w < dist[v]: atomicMin(&dist[v], new_dist)
//   - atomicMin 保证并发安全
//   - changed flag: 如果某轮没有任何更新 → 提前终止
//   - Host 端循环最多 N-1 轮

// ---- Floyd-Warshall: All-Pairs Shortest Path ----
// dist: [N, N] adjacency matrix with distances (FLT_MAX for no edge)
// One iteration of the algorithm: for each (i,j), check if going through k is shorter
__global__ void floyd_warshall_step(float* dist, int N, int k) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        float d_ik = dist[i * N + k];
        float d_kj = dist[k * N + j];
        if (d_ik < FLT_MAX && d_kj < FLT_MAX) {
            float new_dist = d_ik + d_kj;
            if (new_dist < dist[i * N + j]) {
                dist[i * N + j] = new_dist;
            }
        }
    }
}

// Tiled Floyd-Warshall for better memory access patterns
#define FW_TILE 16

__global__ void floyd_warshall_tiled_phase1(float* dist, int N, int k_block) {
    __shared__ float smem[FW_TILE][FW_TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int base = k_block * FW_TILE;
    int i = base + ty, j = base + tx;

    smem[ty][tx] = (i < N && j < N) ? dist[i * N + j] : FLT_MAX;
    __syncthreads();

    for (int k = 0; k < FW_TILE; ++k) {
        __syncthreads();
        if (smem[ty][k] < FLT_MAX && smem[k][tx] < FLT_MAX) {
            float new_val = smem[ty][k] + smem[k][tx];
            if (new_val < smem[ty][tx]) {
                smem[ty][tx] = new_val;
            }
        }
    }
    __syncthreads();

    if (i < N && j < N) {
        dist[i * N + j] = smem[ty][tx];
    }
}

// ---- Bellman-Ford: Single Source Shortest Path ----
// Relaxation step: for each edge (u, v, w), if dist[u] + w < dist[v], update dist[v]
__global__ void bellman_ford_relax(const int* src_nodes, const int* dst_nodes,
                                    const float* weights, float* dist,
                                    int num_edges, int* changed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        int u = src_nodes[idx];
        int v = dst_nodes[idx];
        float w = weights[idx];
        if (dist[u] < FLT_MAX) {
            float new_dist = dist[u] + w;
            if (new_dist < dist[v]) {
                atomicMin(reinterpret_cast<int*>(&dist[v]), __float_as_int(new_dist));
                *changed = 1;
            }
        }
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_floyd_warshall(torch::Tensor dist_matrix) {
    CHECK_TORCH_TENSOR_DTYPE(dist_matrix, torch::kFloat32);
    int N = dist_matrix.size(0);
    auto result = dist_matrix.clone();

    dim3 block(16, 16);
    dim3 grid(CEIL(N, 16), CEIL(N, 16));

    for (int k = 0; k < N; ++k) {
        floyd_warshall_step<<<grid, block>>>(result.data_ptr<float>(), N, k);
    }
    return result;
}

torch::Tensor torch_bellman_ford(torch::Tensor src_nodes, torch::Tensor dst_nodes,
                                  torch::Tensor weights, int num_nodes, int source) {
    int num_edges = src_nodes.numel();

    // Initialize distances
    auto dist = torch::full({num_nodes}, FLT_MAX,
                            torch::dtype(torch::kFloat32).device(src_nodes.device()));
    dist[source] = 0.0f;

    auto changed = torch::zeros({1}, torch::dtype(torch::kInt32).device(src_nodes.device()));

    int block = 256;
    int grid = CEIL(num_edges, block);

    // Iterate at most num_nodes-1 times
    for (int iter = 0; iter < num_nodes - 1; ++iter) {
        changed.zero_();
        bellman_ford_relax<<<grid, block>>>(
            src_nodes.data_ptr<int>(), dst_nodes.data_ptr<int>(),
            weights.data_ptr<float>(), dist.data_ptr<float>(),
            num_edges, changed.data_ptr<int>());
        cudaDeviceSynchronize();
        if (changed.item<int>() == 0) break;
    }
    return dist;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_floyd_warshall)
    TORCH_BINDING_COMMON_EXTENSION(torch_bellman_ford)
}
