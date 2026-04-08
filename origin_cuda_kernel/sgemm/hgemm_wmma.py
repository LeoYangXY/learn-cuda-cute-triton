"""
原生 CUDA HGEMM (fp16) 性能测试
对应 cutedsl_ref/sgemm.py 的各版本

语义: C[M,N] = A[M,K] × B[N,K]^T  (B 转置存储)
"""
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling HGEMM WMMA kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "hgemm_wmma.cu"),
    funcs=[
        "torch_hgemm_naive",
        "torch_hgemm_wmma",
        "torch_hgemm_wmma_cpasync",
        "torch_hgemm_wmma_warp_specialized",
    ],
    extra_cuda_cflags=["-arch=sm_90"],
)

# M, N, K must be multiples of 128 for WMMA tiled kernels
M, N, K = 4096, 4096, 4096
print(f"\n{'='*70}")
print(f"HGEMM fp16 性能对比 (M={M}, N={N}, K={K})")
print(f"语义: C[M,N] = A[M,K] × B[N,K]^T")
print(f"{'='*70}")

A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(N, K, device="cuda", dtype=torch.float16)  # B 转置存储: (N, K)

def pytorch_matmul(A, B):
    return torch.matmul(A, B.T)

kernels = {
    "V1 hgemm_naive (fp16 scalar FMA)":    lambda A, B: lib["torch_hgemm_naive"](A, B),
    "V2 hgemm_wmma (Tensor Core)":         lambda A, B: lib["torch_hgemm_wmma"](A, B),
    "V3 hgemm_wmma_cpasync (3-stage)":     lambda A, B: lib["torch_hgemm_wmma_cpasync"](A, B),
    "V4 hgemm_warp_specialized (mbarrier)": lambda A, B: lib["torch_hgemm_wmma_warp_specialized"](A, B),
}

benchmark_kernels(kernels, pytorch_matmul, A, B, atol=5.0, rtol=0.1)
