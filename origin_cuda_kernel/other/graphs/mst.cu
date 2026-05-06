#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
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

// ==================== Minimum Spanning Tree (Borůvka's Algorithm) ====================
//
// 输入: 邻接矩阵 [N, N] (FLT_MAX 表示无边)
// 输出: MST 边列表或总权重
//
// 【任务划分逻辑】
//
// Borůvka 算法天然适合 GPU 并行:
// - 每个节点独立寻找其所在分量的最小出边 → 完全并行
// - 每轮至少合并一半的分量 → O(log N) 轮
//
// ▸ Step 1: Find minimum outgoing edge for each component
//   - 每个 thread 负责一个节点
//   - 遍历该节点的所有边，找到连接不同分量的最小边
//   - atomicMin 更新分量的最小出边
//
// ▸ Step 2: Merge components (Union-Find with path compression)
//   - 每个 thread 处理一条最小出边
//   - 将两端分量合并
//
// ▸ Step 3: Path compression
//   - 每个 thread 对自己的 parent 做路径压缩
//
// 整体复杂度: O(E * log V) 在 GPU 上并行化
// Host 控制外层循环 (最多 log N 轮)

// Find root of component (device function, no recursion for GPU)
__device__ int find_root(int* parent, int x) {
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];  // path compression (halving)
        x = parent[x];
    }
    return x;
}

// Step 1: For each node, find the minimum weight edge to a different component
__global__ void find_min_edge_kernel(const float* adj, const int* parent,
                                      float* min_weight, int* min_edge_dst,
                                      int N) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    int my_root = find_root(const_cast<int*>(parent), node);
    float best_w = FLT_MAX;
    int best_dst = -1;

    for (int j = 0; j < N; ++j) {
        if (j == node) continue;
        float w = adj[node * N + j];
        if (w < FLT_MAX) {
            int other_root = find_root(const_cast<int*>(parent), j);
            if (other_root != my_root && w < best_w) {
                best_w = w;
                best_dst = j;
            }
        }
    }

    // Atomic update: find minimum for this component
    if (best_dst >= 0) {
        // Use atomicMin on integer representation of float for component
        int old = atomicMin(reinterpret_cast<int*>(&min_weight[my_root]),
                           __float_as_int(best_w));
        if (__float_as_int(best_w) <= old) {
            min_edge_dst[my_root] = best_dst;
        }
    }
}

// Step 2: Merge components based on minimum edges found
__global__ void merge_components_kernel(int* parent, const float* min_weight,
                                         const int* min_edge_dst, float* mst_weight,
                                         int* changed, int N) {
    int comp = blockIdx.x * blockDim.x + threadIdx.x;
    if (comp >= N) return;
    if (parent[comp] != comp) return;  // not a root
    if (min_edge_dst[comp] < 0) return;  // no outgoing edge

    int dst = min_edge_dst[comp];
    int dst_root = find_root(parent, dst);

    if (dst_root == comp) return;  // already same component

    // Merge: smaller root becomes parent (to avoid cycles)
    if (comp < dst_root) {
        parent[dst_root] = comp;
        atomicAdd(mst_weight, min_weight[comp]);
        *changed = 1;
    }
}

// Step 3: Path compression pass
__global__ void path_compress_kernel(int* parent, int N) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;

    int root = node;
    while (parent[root] != root) {
        root = parent[root];
    }
    // Compress path
    int current = node;
    while (parent[current] != root) {
        int next = parent[current];
        parent[current] = root;
        current = next;
    }
}

// ==================== Torch Binding ====================

torch::Tensor torch_mst(torch::Tensor adj_matrix) {
    CHECK_TORCH_TENSOR_DTYPE(adj_matrix, torch::kFloat32);
    int N = adj_matrix.size(0);

    // Initialize Union-Find: parent[i] = i
    auto parent = torch::arange(N, torch::dtype(torch::kInt32).device(adj_matrix.device()));
    auto min_weight = torch::full({N}, FLT_MAX, adj_matrix.options());
    auto min_edge_dst = torch::full({N}, -1, torch::dtype(torch::kInt32).device(adj_matrix.device()));
    auto mst_weight = torch::zeros({1}, adj_matrix.options());
    auto changed = torch::zeros({1}, torch::dtype(torch::kInt32).device(adj_matrix.device()));

    int block = 256;
    int grid = CEIL(N, block);

    // Borůvka's algorithm: iterate until no more merges
    for (int iter = 0; iter < N; ++iter) {
        // Reset
        min_weight.fill_(FLT_MAX);
        min_edge_dst.fill_(-1);
        changed.zero_();

        // Find minimum outgoing edges
        find_min_edge_kernel<<<grid, block>>>(adj_matrix.data_ptr<float>(),
            parent.data_ptr<int>(), min_weight.data_ptr<float>(),
            min_edge_dst.data_ptr<int>(), N);
        cudaDeviceSynchronize();

        // Merge components
        merge_components_kernel<<<grid, block>>>(parent.data_ptr<int>(),
            min_weight.data_ptr<float>(), min_edge_dst.data_ptr<int>(),
            mst_weight.data_ptr<float>(), changed.data_ptr<int>(), N);
        cudaDeviceSynchronize();

        // Path compression
        path_compress_kernel<<<grid, block>>>(parent.data_ptr<int>(), N);
        cudaDeviceSynchronize();

        // Check convergence
        if (changed.item<int>() == 0) break;
    }

    return mst_weight;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_mst)
}
