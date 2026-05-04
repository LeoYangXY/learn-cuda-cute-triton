import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling advanced loss kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "loss_advanced.cu"),
    funcs=[
        "torch_kl_div",
        "torch_triplet_loss",
    ],
)

# ===== KL Divergence =====
M, N = 4096, 1024
print(f"\n===== KL Divergence ({M}x{N}) =====")
# Create valid probability distributions
logits = torch.randn(M, N, device="cuda", dtype=torch.float32)
log_probs = F.log_softmax(logits, dim=-1)
target = F.softmax(torch.randn(M, N, device="cuda", dtype=torch.float32), dim=-1)

def pytorch_kl_div(lp, t):
    # sum(target * (log(target) - log_probs))
    return (t * (t.log() - lp)).sum().unsqueeze(0)

benchmark_kernels(
    {"kl_div": lambda lp, t: lib["torch_kl_div"](lp, t)},
    pytorch_kl_div, log_probs, target, atol=1.0, rtol=1e-2
)

# ===== Triplet Margin Loss =====
M_t, D_t = 4096, 512
margin = 1.0
print(f"\n===== Triplet Margin Loss ({M_t}x{D_t}, margin={margin}) =====")
anchor = torch.randn(M_t, D_t, device="cuda", dtype=torch.float32)
positive = torch.randn(M_t, D_t, device="cuda", dtype=torch.float32)
negative = torch.randn(M_t, D_t, device="cuda", dtype=torch.float32)

def pytorch_triplet_loss(a, p, n):
    dp = (a - p).norm(dim=1)
    dn = (a - n).norm(dim=1)
    return F.relu(dp - dn + margin)

benchmark_kernels(
    {"triplet_loss": lambda a, p, n: lib["torch_triplet_loss"](a, p, n, margin)},
    pytorch_triplet_loss, anchor, positive, negative, atol=1e-3, rtol=1e-3
)
