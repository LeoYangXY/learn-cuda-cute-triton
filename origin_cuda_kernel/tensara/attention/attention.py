import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling attention kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "attention.cu"),
    funcs=[
        "torch_sdpa",
    ],
)

# ===== Scaled Dot-Product Attention =====
B, H, S, D = 2, 8, 256, 64
BH = B * H
print(f"\n===== Scaled Dot-Product Attention (B={B}, H={H}, S={S}, D={D}) =====")

Q = torch.randn(BH, S, D, device="cuda", dtype=torch.float32)
K = torch.randn(BH, S, D, device="cuda", dtype=torch.float32)
V = torch.randn(BH, S, D, device="cuda", dtype=torch.float32)

def pytorch_sdpa(q, k, v):
    scale = 1.0 / (D ** 0.5)
    scores = torch.bmm(q, k.transpose(1, 2)) * scale
    attn = F.softmax(scores, dim=-1)
    return torch.bmm(attn, v)

benchmark_kernels(
    {"sdpa": lambda q, k, v: lib["torch_sdpa"](q, k, v)},
    pytorch_sdpa, Q, K, V, atol=1e-3, rtol=1e-3
)

# Larger sequence length
S2 = 512
print(f"\n===== SDPA (B={B}, H={H}, S={S2}, D={D}) =====")
Q2 = torch.randn(BH, S2, D, device="cuda", dtype=torch.float32)
K2 = torch.randn(BH, S2, D, device="cuda", dtype=torch.float32)
V2 = torch.randn(BH, S2, D, device="cuda", dtype=torch.float32)

benchmark_kernels(
    {"sdpa": lambda q, k, v: lib["torch_sdpa"](q, k, v)},
    pytorch_sdpa, Q2, K2, V2, atol=1e-3, rtol=1e-3
)
