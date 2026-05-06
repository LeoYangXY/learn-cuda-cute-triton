import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling loss kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "loss.cu"),
    funcs=[
        "torch_huber_loss",
        "torch_mse_loss",
        "torch_hinge_loss",
        "torch_cosine_similarity",
    ],
)

N = 4 * 1024 * 1024
pred = torch.randn(N, device="cuda", dtype=torch.float32)
target = torch.randn(N, device="cuda", dtype=torch.float32)

# ===== Huber Loss =====
print(f"\n===== Huber Loss (N={N}) =====")
delta = 1.0

def pytorch_huber(p, t):
    return F.huber_loss(p, t, delta=delta, reduction='mean').unsqueeze(0)

benchmark_kernels(
    {"huber_loss": lambda p, t: lib["torch_huber_loss"](p, t, delta)},
    pytorch_huber, pred, target, atol=1e-2, rtol=1e-2
)

# ===== MSE Loss =====
print(f"\n===== MSE Loss (N={N}) =====")

def pytorch_mse(p, t):
    return F.mse_loss(p, t, reduction='mean').unsqueeze(0)

benchmark_kernels(
    {"mse_loss": lambda p, t: lib["torch_mse_loss"](p, t)},
    pytorch_mse, pred, target, atol=1e-2, rtol=1e-2
)

# ===== Hinge Loss =====
print(f"\n===== Hinge Loss (N={N}) =====")
target_hinge = torch.sign(torch.randn(N, device="cuda", dtype=torch.float32))  # +1 or -1

def pytorch_hinge(p, t):
    return (F.relu(1.0 - t * p)).mean().unsqueeze(0)

benchmark_kernels(
    {"hinge_loss": lambda p, t: lib["torch_hinge_loss"](p, t)},
    pytorch_hinge, pred, target_hinge, atol=1e-2, rtol=1e-2
)

# ===== Cosine Similarity =====
M, D = 4096, 4096
print(f"\n===== Cosine Similarity ({M}x{D}) =====")
a = torch.randn(M, D, device="cuda", dtype=torch.float32)
b = torch.randn(M, D, device="cuda", dtype=torch.float32)

def pytorch_cosine(a, b):
    return F.cosine_similarity(a, b, dim=1)

benchmark_kernels(
    {"cosine_sim": lambda a, b: lib["torch_cosine_similarity"](a, b)},
    pytorch_cosine, a, b, atol=1e-4, rtol=1e-4
)
