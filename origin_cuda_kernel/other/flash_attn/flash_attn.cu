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

// ==================== Flash Attention (Simplified) ====================
// Q: [N, d], K: [N, d], V: [N, d] → O: [N, d]
// 核心思想: tiled computation + online softmax (避免 O(N²) 显存)
//
// 【任务划分逻辑】
//
// ▸ 外层: 按 Q 的 block 划分 (Br 行为一组)
//   - grid.x = ceil(N / Br)
//   - 每个 block 负责输出 O[br*Br : (br+1)*Br, :] 这 Br 行
//
// ▸ 内层: 遍历 K/V 的所有 block (Bc 行为一组)
//   - for each kv_block: 加载 K_tile, V_tile 到 smem
//   - 计算 S_tile = Q_tile @ K_tile^T  (Br × Bc 的注意力分数)
//   - Online Softmax 更新:
//     ① 找新 tile 的 row_max → 更新全局 row_max
//     ② 用旧/新 max 的差修正之前的累积: O *= exp(old_max - new_max)
//     ③ 计算新 tile 的 exp(S - new_max) → P_tile
//     ④ 累加: O += P_tile @ V_tile
//     ⑤ 更新行 sum
//   - 最终: O /= row_sum → 归一化输出
//
// ▸ 关键优化:
//   - Q tile 常驻 smem/register，K/V tile 流式加载
//   - 全程不需要存 N×N 的注意力矩阵
//   - 内存复杂度: O(N) 而非 O(N²)

#define BR 32   // Q block rows
#define BC 32   // KV block cols
#define D_HEAD 64  // head dimension (固定)

// Flash Attention forward kernel (single head, no mask)
__global__ void flash_attn_kernel(const float* Q, const float* K, const float* V,
                                   float* O, int N, int d) {
    // Block 负责 Q 的 [br_start, br_start + BR) 行
    int br_start = blockIdx.x * BR;
    int tid = threadIdx.x;  // tid ∈ [0, BR)，每个 thread 负责 output 的一行

    if (br_start + tid >= N) return;

    // 每个 thread 维护自己那一行的: O_acc[d], row_max, row_sum
    float o_acc[D_HEAD] = {0.0f};
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    // Q row for this thread (从 global memory 读一次)
    float q_row[D_HEAD];
    for (int j = 0; j < d; ++j) {
        q_row[j] = Q[(br_start + tid) * d + j];
    }

    // 遍历所有 KV blocks
    int num_kv_blocks = CEIL(N, BC);
    for (int kv_blk = 0; kv_blk < num_kv_blocks; ++kv_blk) {
        int kv_start = kv_blk * BC;

        // 计算 S[tid][0..BC-1] = Q[br_start+tid] @ K[kv_start:kv_start+BC]^T
        float s_row[BC];
        for (int j = 0; j < BC; ++j) {
            int kv_idx = kv_start + j;
            if (kv_idx >= N) {
                s_row[j] = -FLT_MAX;
                continue;
            }
            float dot = 0.0f;
            for (int k = 0; k < d; ++k) {
                dot += q_row[k] * K[kv_idx * d + k];
            }
            s_row[j] = dot / sqrtf((float)d);  // scaled
        }

        // Online softmax update
        // ① 找 new tile 的 max
        float tile_max = -FLT_MAX;
        for (int j = 0; j < BC; ++j) {
            tile_max = fmaxf(tile_max, s_row[j]);
        }

        // ② 更新全局 max
        float old_max = row_max;
        float new_max = fmaxf(old_max, tile_max);

        // ③ 修正之前的累积
        float correction = expf(old_max - new_max);
        row_sum *= correction;
        for (int j = 0; j < d; ++j) {
            o_acc[j] *= correction;
        }

        // ④ 计算 exp(s - new_max) 并累加
        for (int j = 0; j < BC; ++j) {
            int kv_idx = kv_start + j;
            if (kv_idx >= N) continue;
            float p = expf(s_row[j] - new_max);
            row_sum += p;
            // O += p * V[kv_idx]
            for (int k = 0; k < d; ++k) {
                o_acc[k] += p * V[kv_idx * d + k];
            }
        }

        row_max = new_max;
    }

    // 归一化
    float inv_sum = 1.0f / row_sum;
    for (int j = 0; j < d; ++j) {
        O[(br_start + tid) * d + j] = o_acc[j] * inv_sum;
    }
}

// ==================== Torch Bindings ====================

torch::Tensor torch_flash_attn(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_TORCH_TENSOR_DTYPE(Q, torch::kFloat32);
    int N = Q.size(0), d = Q.size(1);
    auto O = torch::empty_like(Q);

    int num_q_blocks = CEIL(N, BR);
    flash_attn_kernel<<<num_q_blocks, BR>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), N, d);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(torch_flash_attn)
}
