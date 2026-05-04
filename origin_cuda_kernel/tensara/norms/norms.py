import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling normalization kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "norms.cu"),
    funcs=[
        "torch_rms_norm",
        "torch_l1_norm",
        "torch_l2_norm",
        "torch_frobenius_norm",
    ],
)

M, N = 4096, 4096
x = torch.randn(M, N, device="cuda", dtype=torch.float32)

# ===== RMS Norm =====
print(f"\n===== RMS Normalization ({M}x{N}) =====")
weight = torch.ones(N, device="cuda", dtype=torch.float32)
eps = 1e-5

def pytorch_rms_norm(inp):
    rms = torch.sqrt(inp.pow(2).mean(dim=-1, keepdim=True) + eps)
    return inp / rms * weight

benchmark_kernels(
    {"rms_norm": lambda inp: lib["torch_rms_norm"](inp, weight, eps)},
    pytorch_rms_norm, x, atol=1e-4, rtol=1e-4
)

# ===== L1 Norm =====
print(f"\n===== L1 Normalization ({M}x{N}) =====")

def pytorch_l1_norm(inp):
    norm = inp.abs().sum(dim=-1, keepdim=True)
    return inp / norm.clamp(min=1e-12)

benchmark_kernels(
    {"l1_norm": lib["torch_l1_norm"]},
    pytorch_l1_norm, x, atol=1e-4, rtol=1e-4
)

# ===== L2 Norm =====
print(f"\n===== L2 Normalization ({M}x{N}) =====")

def pytorch_l2_norm(inp):
    return F.normalize(inp, p=2, dim=-1)

benchmark_kernels(
    {"l2_norm": lib["torch_l2_norm"]},
    pytorch_l2_norm, x, atol=1e-4, rtol=1e-4
)

# ===== Frobenius Norm =====
print(f"\n===== Frobenius Normalization ({M}x{N}) =====")

def pytorch_frobenius_norm(inp):
    norm = torch.norm(inp, p='fro')
    return inp / norm

benchmark_kernels(
    {"frobenius_norm": lib["torch_frobenius_norm"]},
    pytorch_frobenius_norm, x, atol=1e-4, rtol=1e-4
)
