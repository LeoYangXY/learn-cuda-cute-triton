import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling reduction kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "reductions.cu"),
    funcs=[
        "torch_reduce_sum_dim1",
        "torch_reduce_max_dim1",
        "torch_reduce_min_dim1",
        "torch_reduce_argmax_dim1",
        "torch_reduce_argmin_dim1",
        "torch_reduce_mean_dim1",
        "torch_reduce_prod_dim1",
    ],
)

M, N = 4096, 4096
print(f"\n===== Reductions over dim=1 ({M}x{N}) =====")
x = torch.randn(M, N, device="cuda", dtype=torch.float32)

# Sum
print("\n--- Sum over dim=1 ---")
benchmark_kernels(
    {"sum_dim1": lib["torch_reduce_sum_dim1"]},
    lambda inp: inp.sum(dim=1), x, atol=1e-2, rtol=1e-2
)

# Max
print("\n--- Max over dim=1 ---")
benchmark_kernels(
    {"max_dim1": lib["torch_reduce_max_dim1"]},
    lambda inp: inp.max(dim=1).values, x, atol=1e-5, rtol=1e-5
)

# Min
print("\n--- Min over dim=1 ---")
benchmark_kernels(
    {"min_dim1": lib["torch_reduce_min_dim1"]},
    lambda inp: inp.min(dim=1).values, x, atol=1e-5, rtol=1e-5
)

# Argmax
print("\n--- Argmax over dim=1 ---")
benchmark_kernels(
    {"argmax_dim1": lib["torch_reduce_argmax_dim1"]},
    lambda inp: inp.argmax(dim=1), x, atol=0, rtol=0
)

# Argmin
print("\n--- Argmin over dim=1 ---")
benchmark_kernels(
    {"argmin_dim1": lib["torch_reduce_argmin_dim1"]},
    lambda inp: inp.argmin(dim=1), x, atol=0, rtol=0
)

# Mean
print("\n--- Mean over dim=1 ---")
benchmark_kernels(
    {"mean_dim1": lib["torch_reduce_mean_dim1"]},
    lambda inp: inp.mean(dim=1), x, atol=1e-3, rtol=1e-3
)

# Product (use smaller N to avoid overflow)
M2, N2 = 1024, 64
x_small = torch.randn(M2, N2, device="cuda", dtype=torch.float32) * 0.5 + 1.0
print(f"\n--- Product over dim=1 ({M2}x{N2}) ---")
benchmark_kernels(
    {"prod_dim1": lib["torch_reduce_prod_dim1"]},
    lambda inp: inp.prod(dim=1), x_small, atol=1e-1, rtol=1e-1
)
