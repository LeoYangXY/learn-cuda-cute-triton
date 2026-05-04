import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling scan kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "scan.cu"),
    funcs=[
        "torch_cumsum_naive",
        "torch_cumsum_parallel",
        "torch_cumprod",
        "torch_running_sum",
        "torch_running_sum_prefix",
    ],
)

# ===== Cumulative Sum =====
M, N = 1024, 1024
print(f"\n===== Cumulative Sum ({M}x{N}) =====")
x = torch.randn(M, N, device="cuda", dtype=torch.float32)

def pytorch_cumsum(inp):
    return inp.cumsum(dim=1)

benchmark_kernels(
    {"cumsum_naive": lib["torch_cumsum_naive"],
     "cumsum_parallel": lib["torch_cumsum_parallel"]},
    pytorch_cumsum, x, atol=1e-2, rtol=1e-2
)

# ===== Cumulative Product =====
M2, N2 = 1024, 256
x_prod = torch.randn(M2, N2, device="cuda", dtype=torch.float32) * 0.5 + 1.0
print(f"\n===== Cumulative Product ({M2}x{N2}) =====")

def pytorch_cumprod(inp):
    return inp.cumprod(dim=1)

benchmark_kernels(
    {"cumprod": lib["torch_cumprod"]},
    pytorch_cumprod, x_prod, atol=1e-1, rtol=1e-1
)

# ===== 1D Running Sum =====
N_run = 4 * 1024 * 1024
K = 64
print(f"\n===== 1D Running Sum (N={N_run}, K={K}) =====")
x_run = torch.randn(N_run, device="cuda", dtype=torch.float32)

def pytorch_running_sum(inp):
    # sliding window sum of size K
    # use unfold
    pad_inp = torch.nn.functional.pad(inp, (K - 1, 0))
    return pad_inp.unfold(0, K, 1).sum(dim=1)

benchmark_kernels(
    {"running_sum": lambda inp: lib["torch_running_sum"](inp, K),
     "running_sum_prefix": lambda inp: lib["torch_running_sum_prefix"](inp, K)},
    pytorch_running_sum, x_run, atol=1e-2, rtol=1e-2
)
