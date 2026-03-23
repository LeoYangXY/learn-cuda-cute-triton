import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, check, timed, benchmark_kernels

# cutlass include path
cutlass_include = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cutlass", "include")

print("Compiling sgemm_cute kernel (with cutlass headers)...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "sgemm_cute.cu"),
    funcs=["torch_sgemm_cute"],
    extra_cuda_cflags=["-std=c++17"],
    extra_include_paths=[cutlass_include],
)

# M, N, K must be multiples of 32
M, N, K = 128, 256, 384
print(f"\n===== SGEMM CUTE (M={M}, N={N}, K={K}) =====")

A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

def pytorch_matmul(A, B):
    return A @ B

kernels = {
    "sgemm_cute": lambda A, B: lib["torch_sgemm_cute"](A, B),
}

benchmark_kernels(kernels, pytorch_matmul, A, B, atol=1e-3, rtol=1e-3)
