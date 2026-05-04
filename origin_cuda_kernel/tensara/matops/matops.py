import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling matops kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "matops.cu"),
    funcs=[
        "torch_mat_scalar_mul",
        "torch_diag_matmul",
        "torch_layer_norm",
        "torch_batch_norm",
    ],
)

M, N = 4096, 4096

# ===== Matrix Scalar Multiplication =====
print(f"\n===== Matrix Scalar Multiplication ({M}x{N}) =====")
A = torch.randn(M, N, device="cuda", dtype=torch.float32)
alpha = 2.5

benchmark_kernels(
    {"mat_scalar_mul": lambda a: lib["torch_mat_scalar_mul"](a, alpha)},
    lambda a: a * alpha, A, atol=1e-5, rtol=1e-5
)

# ===== Diagonal Matrix Multiplication =====
print(f"\n===== Diagonal Matrix Multiplication ({M}x{N}) =====")
d = torch.randn(M, device="cuda", dtype=torch.float32)

def pytorch_diag_matmul(d_vec, mat):
    return d_vec.unsqueeze(1) * mat

benchmark_kernels(
    {"diag_matmul": lambda d_vec, mat: lib["torch_diag_matmul"](d_vec, mat)},
    pytorch_diag_matmul, d, A, atol=1e-5, rtol=1e-5
)

# ===== Layer Normalization =====
print(f"\n===== Layer Normalization ({M}x{N}) =====")
gamma = torch.ones(N, device="cuda", dtype=torch.float32)
beta = torch.zeros(N, device="cuda", dtype=torch.float32)
eps = 1e-5

def pytorch_layer_norm(inp):
    return F.layer_norm(inp, [N], gamma, beta, eps)

benchmark_kernels(
    {"layer_norm": lambda inp: lib["torch_layer_norm"](inp, gamma, beta, eps)},
    pytorch_layer_norm, A, atol=1e-4, rtol=1e-4
)

# ===== Batch Normalization =====
M_bn, C_bn = 8192, 512
print(f"\n===== Batch Normalization ({M_bn}x{C_bn}) =====")
x_bn = torch.randn(M_bn, C_bn, device="cuda", dtype=torch.float32)
running_mean = torch.randn(C_bn, device="cuda", dtype=torch.float32)
running_var = torch.rand(C_bn, device="cuda", dtype=torch.float32) + 0.1
weight_bn = torch.ones(C_bn, device="cuda", dtype=torch.float32)
bias_bn = torch.zeros(C_bn, device="cuda", dtype=torch.float32)

def pytorch_batch_norm(inp):
    return F.batch_norm(inp, running_mean, running_var, weight_bn, bias_bn, training=False, eps=eps)

benchmark_kernels(
    {"batch_norm": lambda inp: lib["torch_batch_norm"](inp, running_mean, running_var, weight_bn, bias_bn, eps)},
    pytorch_batch_norm, x_bn, atol=1e-4, rtol=1e-4
)
