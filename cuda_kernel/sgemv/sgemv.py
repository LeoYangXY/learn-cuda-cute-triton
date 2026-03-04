import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling sgemv kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "sgemv.cu"),
    funcs=["torch_sgemv_32", "torch_sgemv_128", "torch_sgemv_16"],
)

def pytorch_mv(A, x):
    return A @ x

M = 128

# ===== Test 1: K=96 (multiple of 32) =====
K1 = 96
print(f"\n===== SGEMV K%32==0 (M={M}, K={K1}) =====")
A1 = torch.randn(M, K1, device="cuda", dtype=torch.float32)
x1 = torch.randn(K1, device="cuda", dtype=torch.float32)

kernels_32 = {"sgemv_32": lambda A, x: lib["torch_sgemv_32"](A, x)}
benchmark_kernels(kernels_32, pytorch_mv, A1, x1, atol=1e-3, rtol=1e-3)

# ===== Test 2: K=256 (multiple of 128) =====
K2 = 256
print(f"\n===== SGEMV K%128==0 (M={M}, K={K2}) =====")
A2 = torch.randn(M, K2, device="cuda", dtype=torch.float32)
x2 = torch.randn(K2, device="cuda", dtype=torch.float32)

kernels_128 = {"sgemv_128": lambda A, x: lib["torch_sgemv_128"](A, x)}
benchmark_kernels(kernels_128, pytorch_mv, A2, x2, atol=1e-3, rtol=1e-3)

# ===== Test 3: K=16 =====
K3 = 16
M3 = 128  # must be multiple of 8 for this kernel
print(f"\n===== SGEMV K=16 (M={M3}, K={K3}) =====")
A3 = torch.randn(M3, K3, device="cuda", dtype=torch.float32)
x3 = torch.randn(K3, device="cuda", dtype=torch.float32)

kernels_16 = {"sgemv_16": lambda A, x: lib["torch_sgemv_16"](A, x)}
benchmark_kernels(kernels_16, pytorch_mv, A3, x3, atol=1e-3, rtol=1e-3)
