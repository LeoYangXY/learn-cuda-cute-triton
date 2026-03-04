import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling sgemm kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "sgemm.cu"),
    funcs=[
        "torch_sgemm_naive",
        "torch_sgemm_sliced_k",
        "torch_sgemm_f32x4_padding",
        "torch_sgemm_f32x4_padding_reg",
    ],
)

# M, N, K must be multiples of 32 for tiled kernels
M, N, K = 128, 256, 384
print(f"\n===== SGEMM (M={M}, N={N}, K={K}) =====")

A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

def pytorch_matmul(A, B):
    return A @ B

kernels = {
    "sgemm_naive":           lambda A, B: lib["torch_sgemm_naive"](A, B),
    "sgemm_sliced_k":        lambda A, B: lib["torch_sgemm_sliced_k"](A, B),
    "sgemm_f32x4_padding":   lambda A, B: lib["torch_sgemm_f32x4_padding"](A, B),
    "sgemm_f32x4_padding_reg": lambda A, B: lib["torch_sgemm_f32x4_padding_reg"](A, B),
}

benchmark_kernels(kernels, pytorch_matmul, A, B, atol=1e-3, rtol=1e-3)
