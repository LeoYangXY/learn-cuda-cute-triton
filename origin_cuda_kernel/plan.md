## 已完成的 Tensara 算子实现

| 目录 | 覆盖的 Tensara 题目 | 状态 |
|------|-------------------|------|
| `activations/` | ReLU, LeakyReLU, GELU, Sigmoid, Tanh, ELU, SELU, Softplus, Swish, HardSigmoid | ✅ 全部通过 |
| `conv1d/` | 1D Convolution (naive, shared, const memory, vec4) | ✅ 全部通过 |
| `conv2d/` | 2D Convolution (naive, tiled shared memory) | ✅ 全部通过 |
| `conv3d/` | 3D Convolution, 3D Max Pooling, 3D Average Pooling | ✅ 全部通过 |
| `pooling/` | 1D Avg Pooling, 1D Max Pooling, 2D Avg Pooling, 2D Max Pooling | ✅ 全部通过 |
| `reductions/` | Sum, Max, Min, Argmax, Argmin, Mean, Product over dim | ✅ 全部通过 |
| `norms/` | RMS Norm, L1 Norm, L2 Norm, Frobenius Norm | ✅ 全部通过 |
| `softmax/` | Softmax, Online Softmax, Log Softmax | ✅ 全部通过 |
| `loss/` | Huber Loss, MSE Loss, Hinge Loss, Cosine Similarity | ✅ 全部通过 |
| `loss_advanced/` | KL Divergence, Triplet Margin Loss | ✅ 全部通过 |
| `matops/` | Matrix Scalar Mul, Diagonal MatMul, Layer Norm, Batch Norm | ✅ 全部通过 |
| `scan/` | Cumulative Sum, Cumulative Product, Running Sum | ✅ 全部通过 |
| `graphics/` | Grayscale, Histogram, Thresholding, Box Blur, Edge Detection | ✅ 全部通过 |
| `sorting/` | Bitonic Sort | ✅ 全部通过 |
| `graphs/` | Floyd-Warshall (All-Pairs SP), Bellman-Ford (SSSP) | ✅ 全部通过 |
| `fused_gemm/` | GEMM+Bias+ReLU, GEMM+Swish, GEMM+Sigmoid+Sum, GEMM+Mul+LeakyReLU | ✅ 全部通过 |
| `attention/` | Scaled Dot-Product Attention | ✅ 全部通过 |
| `tensor_matmul/` | 3D/4D Tensor-Matrix Mul, Square MatMul, Upper/Lower Triangular MatMul | ✅ 全部通过 |

加上你已有的 `add/`, `reduce_max/`, `sgemm/`, `sgemv/`, `embedding/`, `layer_norm/`, `transpose/`，Tensara 上 84 道题中的绝大多数核心算子都已覆盖（除了密码学相关的 ECC/Finite Field 和 MXFP/NVFP 量化题目，这些过于特殊且与常规 GPU 编程学习关联不大）。