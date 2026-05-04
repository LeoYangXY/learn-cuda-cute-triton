import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling fused_gemm kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "fused_gemm.cu"),
    funcs=[
        "torch_gemm_bias_relu",
        "torch_gemm_swish",
        "torch_gemm_sigmoid_sum",
        "torch_gemm_mul_leaky_relu",
    ],
)

M, K, N = 1024, 1024, 1024
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

# ===== GEMM + Bias + ReLU =====
print(f"\n===== GEMM + Bias + ReLU ({M}x{K} @ {K}x{N}) =====")
bias = torch.randn(N, device="cuda", dtype=torch.float32)

def pytorch_gemm_bias_relu(a, b, bi):
    return F.relu(a @ b + bi)

benchmark_kernels(
    {"gemm_bias_relu": lambda a, b, bi: lib["torch_gemm_bias_relu"](a, b, bi)},
    pytorch_gemm_bias_relu, A, B, bias, atol=1e-2, rtol=1e-2
)

# ===== GEMM + Swish =====
print(f"\n===== GEMM + Swish ({M}x{K} @ {K}x{N}) =====")

def pytorch_gemm_swish(a, b):
    c = a @ b
    return c * torch.sigmoid(c)

benchmark_kernels(
    {"gemm_swish": lambda a, b: lib["torch_gemm_swish"](a, b)},
    pytorch_gemm_swish, A, B, atol=1e-2, rtol=1e-2
)

# ===== GEMM + Sigmoid + Sum =====
print(f"\n===== GEMM + Sigmoid + Sum ({M}x{K} @ {K}x{N}) =====")

def pytorch_gemm_sigmoid_sum(a, b):
    return torch.sigmoid(a @ b).sum().unsqueeze(0)

benchmark_kernels(
    {"gemm_sigmoid_sum": lambda a, b: lib["torch_gemm_sigmoid_sum"](a, b)},
    pytorch_gemm_sigmoid_sum, A, B, atol=1.0, rtol=1e-2
)

# ===== GEMM + Mul + LeakyReLU =====
print(f"\n===== GEMM + Mul + LeakyReLU ({M}x{K} @ {K}x{N}) =====")
D = torch.randn(M, N, device="cuda", dtype=torch.float32)
alpha = 0.01

def pytorch_gemm_mul_leaky_relu(a, b, d):
    val = (a @ b) * d
    return F.leaky_relu(val, alpha)

benchmark_kernels(
    {"gemm_mul_leaky_relu": lambda a, b, d: lib["torch_gemm_mul_leaky_relu"](a, b, d, alpha)},
    pytorch_gemm_mul_leaky_relu, A, B, D, atol=1e-2, rtol=1e-2
)
