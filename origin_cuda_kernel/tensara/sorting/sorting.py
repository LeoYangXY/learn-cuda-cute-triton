import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling sorting kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "sorting.cu"),
    funcs=[
        "torch_bitonic_sort",
    ],
)

# ===== Small array =====
N = 1024
print(f"\n===== Bitonic Sort (N={N}) =====")
x = torch.randn(N, device="cuda", dtype=torch.float32)

def pytorch_sort(inp):
    return inp.sort().values

benchmark_kernels(
    {"bitonic_sort": lib["torch_bitonic_sort"]},
    pytorch_sort, x, atol=1e-5, rtol=1e-5
)

# ===== Medium array =====
N2 = 64 * 1024
print(f"\n===== Bitonic Sort (N={N2}) =====")
x2 = torch.randn(N2, device="cuda", dtype=torch.float32)

benchmark_kernels(
    {"bitonic_sort": lib["torch_bitonic_sort"]},
    pytorch_sort, x2, atol=1e-5, rtol=1e-5
)
