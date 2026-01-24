练习cuda-kernel，参考leet-cuda学习


每个kernel都使用单文件组织，避免复杂的目录结构，工具函数直接贴在当前文件开头


我们主要写fp32数据类型，不做过多数据类型的  
ok代表已完成，no代表我们不去做这个的


### 📌 Elementwise Ops

ok！

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | elementwise_f32 | f32 | / | ⭐️ |
|  | elementwise_f32x4 | f32 | / | ⭐️ |
|  | elementwise_f16 | f16 | / | ⭐️ |
|  | elementwise_f16x2 | f16 | / | ⭐️ |
|  | elementwise_f16x8 | f16 | / | ⭐️ |
|  | elementwise_f16x8_pack | f16 | / | ⭐️⭐️ |

float4去处理fp32，一次性读取4个
half2去处理fp16，一次性读取2个
每个thread申请4个half2，负责8个元素
直接reinterpret_cast去读出128bits，用这个pack版本处理half

<!-- 
- 
- 
- 
- 
-->





### 📌 Activation Functions

和elementwise里面的kernel几乎一致，不再写了

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | sigmoid_f32 | f32 | / | ⭐️ |
|  | sigmoid_f32x4 | f32 | / | ⭐️ |
|  | sigmoid_f16 | f16 | / | ⭐️ |
|  | sigmoid_f16x2 | f16 | / | ⭐️ |
|  | sigmoid_f16x8 | f16 | / | ⭐️ |
|  | sigmoid_f16x8_pack | f16 | / | ⭐️⭐️ |
|  | relu_f32 | f32 | / | ⭐️ |
|  | relu_f32x4 | f32 | / | ⭐️ |
|  | relu_f16 | f16 | / | ⭐️ |
|  | relu_f16x2 | f16 | / | ⭐️ |
|  | relu_f16x8 | f16 | / | ⭐️ |
|  | relu_f16x8_pack | f16 | / | ⭐️⭐️ |
|  | elu_f32 | f32 | / | ⭐️ |
|  | elu_f32x4 | f32 | / | ⭐️ |
|  | elu_f16 | f16 | / | ⭐️ |
|  | elu_f16x2 | f16 | / | ⭐️ |
|  | elu_f16x8 | f16 | / | ⭐️ |
|  | elu_f16x8_pack | f16 | / | ⭐️⭐️ |
|  | gelu_f32 | f32 | / | ⭐️ |
|  | gelu_f32x4 | f32 | / | ⭐️ |
|  | gelu_f16 | f16 | / | ⭐️ |
|  | gelu_f16x2 | f16 | / | ⭐️ |
|  | gelu_f16x8 | f16 | / | ⭐️ |
|  | gelu_f16x8_pack | f16 | / | ⭐️⭐️ |
|  | swish_f32 | f32 | / | ⭐️ |
|  | swish_f32x4 | f32 | / | ⭐️ |
|  | swish_f16 | f16 | / | ⭐️ |
|  | swish_f16x2 | f16 | / | ⭐️ |
|  | swish_f16x8 | f16 | / | ⭐️ |
|  | swish_f16x8_pack | f16 | / | ⭐️⭐️ |
|  | hardswish_f32 | f32 | / | ⭐️ |
|  | hardswish_f32x4 | f32 | / | ⭐️ |
|  | hardswish_f16 | f16 | / | ⭐️ |
|  | hardswish_f16x2 | f16 | / | ⭐️ |
|  | hardswish_f16x8 | f16 | / | ⭐️ |
|  | hardswish_f16x8_pack | f16 | / | ⭐️⭐️ |
|  | hardshrink_f32 | f32 | / | ⭐️ |
|  | hardshrink_f32x4 | f32 | / | ⭐️ |
|  | hardshrink_f16 | f16 | / | ⭐️ |
|  | hardshrink_f16x2 | f16 | / | ⭐️ |
|  | hardshrink_f16x8 | f16 | / | ⭐️ |
|  | hardshrink_f16x8_pack | f16 | / | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 Embedding Lookup

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | embedding_f32 | f32 | / | ⭐️ |
|  | embedding_f32x4 | f32 | / | ⭐️ |
|  | embedding_f32x4_pack | f32 | / | ⭐️ |
|  | embedding_f16 | f16 | / | ⭐️ |
|  | embedding_f16x2 | f16 | / | ⭐️ |
|  | embedding_f16x8 | f16 | / | ⭐️ |
|  | embedding_f16x8_pack | f16 | / | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 Histogram (Integer Only)

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | histogram_i32 | i32 | / | ⭐️ |
|  | histogram_i32x4 | i32 | / | ⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 Matrix Transpose

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | mat_trans_f32_col2row{2d} | f32 | / | ⭐️ |
|  | mat_trans_f32_row2col{2d} | f32 | / | ⭐️ |
|  | mat_trans_f32_diagonal2d | f32 | / | ⭐️⭐️ |
|  | mat_trans_f32x4_col2row{2d} | f32 | / | ⭐️⭐️ |
|  | mat_trans_f32x4_row2col{2d} | f32 | / | ⭐️⭐️ |
|  | mat_trans_cute | f32 | / | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 Warp & Block Reduction
ok

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | warp_reduce_{all} | all | all | ⭐️⭐️ |
|  | block_all_reduce_f32_f32 | f32 | f32 | ⭐️⭐️ |
|  | block_all_reduce_f32x4_f32 | f32 | f32 | ⭐️⭐️ |
|  | block_all_reduce_f16_f16 | f16 | f16 | ⭐️⭐️ |
|  | block_all_reduce_f16_f32 | f16 | f32 | ⭐️⭐️ |
|  | block_all_reduce_f16x2_f16 | f16 | f16 | ⭐️⭐️ |
|  | block_all_reduce_f16x2_f32 | f16 | f32 | ⭐️⭐️ |
|  | block_all_reduce_f16x8_pack_f16 | f16 | f16 | ⭐️⭐️ |
|  | block_all_reduce_f16x8_pack_f32 | f16 | f32 | ⭐️⭐️ |
|  | block_all_reduce_bf16_bf16 | bf16 | bf16 | ⭐️⭐️ |
|  | block_all_reduce_bf16_f32 | bf16 | f32 | ⭐️⭐️ |
|  | block_all_reduce_bf16x2_bf16 | bf16 | bf16 | ⭐️⭐️ |
|  | block_all_reduce_bf16x2_f32 | bf16 | f32 | ⭐️⭐️ |
|  | block_all_reduce_bf16x8_pack_bf16 | bf16 | bf16 | ⭐️⭐️ |
|  | block_all_reduce_bf16x8_pack_f32 | bf16 | f32 | ⭐️⭐️ |
|  | block_all_reduce_fp8_e4m3_f16 | fp8_e4m3 | f16 | ⭐️⭐️⭐️ |
|  | block_all_reduce_fp8_e5m2_f16 | fp8_e5m2 | f16 | ⭐️⭐️⭐️ |
|  | block_all_reduce_fp8_e4m3x16_pack_f16 | fp8_e4m3 | f16 | ⭐️⭐️⭐️ |
|  | block_all_reduce_fp8_e5m2x16_pack_f16 | fp8_e5m2 | f16 | ⭐️⭐️⭐️ |
|  | block_all_reduce_i8_i32 | i8 | i32 | ⭐️⭐️ |
|  | block_all_reduce_i8x16_pack_i32 | i8 | i32 | ⭐️⭐️ |

只用shared_memory
shared_memory+warp_shuffle,每个thread负责一个元素
每个thread负责4个元素，用float4去读取
用pack去处理128bits的元素，那个LDST128BITS的语法糖

<!-- 
- 
- 
- 
- 
-->


### 📌 Dot Product

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | dot_product_f32 | f32 | f32 | ⭐️⭐️ |
|  | dot_product_f32x4 | f32 | f32 | ⭐️⭐️ |
|  | dot_product_f16_f32 | f16 | f32 | ⭐️⭐️ |
|  | dot_product_f16x2_f32 | f16 | f32 | ⭐️⭐️ |
|  | dot_product_f16x8_pack_f32 | f16 | f32 | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 Softmax Variants

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | softmax_f32_per_tok | f32 | f32 | ⭐️⭐️ |
|  | softmax_f32x4_per_tok | f32 | f32 | ⭐️⭐️ |
|  | safe_softmax_f32_per_tok | f32 | f32 | ⭐️⭐️ |
|  | safe_softmax_f32x4_per_tok | f32 | f32 | ⭐️⭐️ |
|  | safe_softmax_f16_f32_per_tok | f16 | f32 | ⭐️⭐️ |
|  | safe_softmax_f16x2_f32_per_tok | f16 | f32 | ⭐️⭐️ |
|  | safe_softmax_f16x8_pack_f32_per_tok | f16 | f32 | ⭐️⭐️ |
|  | online_safe_softmax_f32_per_token | f32 | f32 | ⭐️⭐️ |
|  | online_safe_softmax_f32x4_pack_per_tok | f32 | f32 | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 RoPE (Rotary Position Embedding)

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | rope_f32 | f32 | f32 | ⭐️⭐️ |
|  | rope_f32x4_pack | f32 | f32 | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 Layer Normalization

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | layer_norm_f32 | f32 | f32 | ⭐️⭐️ |
|  | layer_norm_f32x4 | f32 | f32 | ⭐️⭐️ |
|  | layer_norm_f16_f16 | f16 | f16 | ⭐️⭐️ |
|  | layer_norm_f16x2_f16 | f16 | f16 | ⭐️⭐️ |
|  | layer_norm_f16x8_f16 | f16 | f16 | ⭐️⭐️ |
|  | layer_norm_f16x8_pack_f16 | f16 | f16 | ⭐️⭐️ |
|  | layer_norm_f16x8_pack_f32 | f16 | f32 | ⭐️⭐️ |
|  | layer_norm_f16_f32 | f16 | f32 | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 RMS Normalization

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | rms_norm_f32 | f32 | f32 | ⭐️⭐️ |
|  | rms_norm_f32x4 | f32 | f32 | ⭐️⭐️ |
|  | rms_norm_f16_f16 | f16 | f16 | ⭐️⭐️ |
|  | rms_norm_f16x2_f16 | f16 | f16 | ⭐️⭐️ |
|  | rms_norm_f16x8_f16 | f16 | f16 | ⭐️⭐️ |
|  | rms_norm_f16x8_f32 | f16 | f32 | ⭐️⭐️ |
|  | rms_norm_f16x8_pack_f16 | f16 | f16 | ⭐️⭐️ |
|  | rms_norm_f16x8_pack_f32 | f16 | f32 | ⭐️⭐️ |
|  | rms_norm_f16_f32 | f16 | f32 | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 Other Ops

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | nms_f32 | f32 | / | ⭐️⭐️ |
|  | merge_attn_states | f16/bf16/f32 | f32 | ⭐️⭐️ |
|  | notes v1(deprecated) | f32 | f32 | ⭐️⭐️ |
|  | How to use nsys/ncu(timeline/ptx/sass) | / | / | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 GEMV (Matrix-Vector Multiply)

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | sgemv_k32_f32 | f32 | f32 | ⭐️⭐️⭐️ |
|  | sgemv_k128_f32x4 | f32 | f32 | ⭐️⭐️⭐️ |
|  | sgemv_k16_f32 | f32 | f32 | ⭐️⭐️⭐️ |
|  | hgemv_k32_f16 | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemv_k128_f16x4 | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemv_k16_f16 | f16 | f16 | ⭐️⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 GEMM (Matrix-Matrix Multiply)

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | sgemm_naive_f32 | f32 | f32 | ⭐️⭐️ |
|  | sgemm_sliced_k_f32 | f32 | f32 | ⭐️⭐️⭐️ |
|  | sgemm_t_8x8_sliced_k_f32x4 | f32 | f32 | ⭐️⭐️⭐️ |
|  | sgemm_t_8x8_sliced_k...bcf | f32 | f32 | ⭐️⭐️⭐️ |
|  | sgemm_t_8x8_sliced_k...dbuf | f32 | f32 | ⭐️⭐️⭐️ |
|  | sgemm_t_8x8_sliced_k16...dbuf | f32 | f32 | ⭐️⭐️⭐️ |
|  | sgemm_t_8x8_sliced_k16...async | f32 | f32 | ⭐️⭐️⭐️ |
|  | sgemm_wmma_m16n16k8...stages* | tf32 | f32 | ⭐️⭐️⭐️ |
|  | sgemm_wmma_m16n16k8...swizzle* | tf32 | f32 | ⭐️⭐️⭐️ |
|  | hgemm_naive_f16 | f16 | f16 | ⭐️⭐️ |
|  | hgemm_sliced_k_f16 | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_t_8x8_sliced_k_f16x4 | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_t_8x8_sliced_k_f16x4_pack | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_t_8x8_sliced_k_f16x8_pack | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_t_8x8_sliced_k...dbuf | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_t_8/16x8...k16/32...dbuf | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_t_8/16x8...k16/32...async | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_wmma_m16n16k16...naive* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_wmma_m16n16k16...mma4x2* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_wmma_m16n16k16...mma4x4* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_wmma_m16n16k16...dbuf* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_wmma_m32n8k16....dbuf* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_wmma_m16n16k16...stages* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_wmma_m16n16k16...swizzle* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_mma_m16n8k16...naive* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_mma_m16n8k16...mma2x4* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_mma_m16n8k16...stages* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_mma_m16n8k16...swizzle* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_mma_m16n8k16...swizzle{smem}* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_mma_m16n8k16...swizzle{tn}{smem}* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_mma_stages_swizzle{smem}...cute* | f16 | f16 | ⭐️⭐️⭐️ |
|  | hgemm_mma_cublas* | f16 | f16 | ⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->


### 📌 FlashAttention Variants

| 完成情况 | CUDA 内核 | 元素数据类型 | 累加数据类型 | 难度 |
|:---|:---|:---|:---|:---|
|  | flash_attn_cute(naive) | f16 | f32 | ⭐️⭐️⭐️ |
|  | How to implement MMA smem swizzle* | f16 | f16 | ⭐️⭐️⭐️ |
|  | flash_attn_mma_stages_split_kv* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages_split_q* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages...shared_kv* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages...shared_qkv* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages...tiling_qk* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages...tiling_qkv* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages...shared_kv{f32}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages...shared_qkv{f32}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages...tiling_qk{f32}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma_stages...tiling_qkv{f32}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...shared_kv{f32}{rr}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...shared_qkv{f32}{rr}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...shared_kv_swizzle{q}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...shared_kv_swizzle{qk}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...shared_kv_swizzle{qkv}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...shared_qkv_swizzle{q}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...shared_qkv_swizzle{qk}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...shared_qkv_swizzle{qkv}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...tiling_qk_swizzle{q}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...tiling_qk_swizzle{qk}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...tiling_qk_swizzle{qkv}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...tiling_qkv_swizzle{q}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...tiling_qkv_swizzle{qk}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn_mma...tiling_qkv_swizzle{qkv}* | f16 | f16 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn...tiling_qkv_swizzle{q}{f32}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn...tiling_qkv_swizzle{qk}{f32}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |
|  | flash_attn...tiling_qkv_swizzle{qkv}{f32}* | f16 | f32 | ⭐️⭐️⭐️⭐️ |

<!-- 
- 
- 
- 
- 
-->