import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling tensor_matmul kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "tensor_matmul.cu"),
    funcs=[
        "torch_tensor3d_matmul",
        "torch_tensor4d_matmul",
        "torch_square_matmul",
        "torch_upper_tri_matmul",
        "torch_lower_tri_matmul",
    ],
)

# ===== 3D Tensor-Matrix Multiplication =====
B, M, K, N = 16, 512, 512, 512
print(f"\n===== 3D Tensor-Matrix Mul (B={B}, M={M}, K={K}, N={N}) =====")
T3d = torch.randn(B, M, K, device="cuda", dtype=torch.float32)
W = torch.randn(K, N, device="cuda", dtype=torch.float32)

def pytorch_tensor3d_matmul(t, w):
    return torch.bmm(t, w.unsqueeze(0).expand(B, -1, -1))

benchmark_kernels(
    {"tensor3d_matmul": lambda t, w: lib["torch_tensor3d_matmul"](t, w)},
    pytorch_tensor3d_matmul, T3d, W, atol=1e-2, rtol=1e-2
)

# ===== 4D Tensor-Matrix Multiplication =====
B4, C4, M4, K4, N4 = 4, 8, 256, 256, 256
print(f"\n===== 4D Tensor-Matrix Mul (B={B4}, C={C4}, M={M4}, K={K4}, N={N4}) =====")
T4d = torch.randn(B4, C4, M4, K4, device="cuda", dtype=torch.float32)
W4 = torch.randn(K4, N4, device="cuda", dtype=torch.float32)

def pytorch_tensor4d_matmul(t, w):
    return torch.matmul(t, w)

benchmark_kernels(
    {"tensor4d_matmul": lambda t, w: lib["torch_tensor4d_matmul"](t, w)},
    pytorch_tensor4d_matmul, T4d, W4, atol=1e-2, rtol=1e-2
)

# ===== Square Matrix Multiplication =====
N_sq = 2048
print(f"\n===== Square Matrix Multiplication (N={N_sq}) =====")
A_sq = torch.randn(N_sq, N_sq, device="cuda", dtype=torch.float32)
B_sq = torch.randn(N_sq, N_sq, device="cuda", dtype=torch.float32)

def pytorch_matmul(a, b):
    return a @ b

benchmark_kernels(
    {"square_matmul": lambda a, b: lib["torch_square_matmul"](a, b)},
    pytorch_matmul, A_sq, B_sq, atol=1e-1, rtol=1e-2
)

# ===== Upper Triangular Matrix Multiplication =====
N_tri = 1024
print(f"\n===== Upper Triangular MatMul (N={N_tri}) =====")
A_tri = torch.randn(N_tri, N_tri, device="cuda", dtype=torch.float32)
B_tri = torch.randn(N_tri, N_tri, device="cuda", dtype=torch.float32)

def pytorch_upper_tri_matmul(a, b):
    c = a @ b
    return torch.triu(c)

benchmark_kernels(
    {"upper_tri_matmul": lambda a, b: lib["torch_upper_tri_matmul"](a, b)},
    pytorch_upper_tri_matmul, A_tri, B_tri, atol=1e-2, rtol=1e-2
)

# ===== Lower Triangular Matrix Multiplication =====
print(f"\n===== Lower Triangular MatMul (N={N_tri}) =====")

def pytorch_lower_tri_matmul(a, b):
    c = a @ b
    return torch.tril(c)

benchmark_kernels(
    {"lower_tri_matmul": lambda a, b: lib["torch_lower_tri_matmul"](a, b)},
    pytorch_lower_tri_matmul, A_tri, B_tri, atol=1e-2, rtol=1e-2
)
