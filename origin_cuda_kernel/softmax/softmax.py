import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling softmax kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "softmax.cu"),
    funcs=[
        "torch_softmax",
        "torch_online_softmax",
        "torch_log_softmax",
    ],
)

M, N = 4096, 4096
print(f"\n===== Softmax ({M}x{N}) =====")
x = torch.randn(M, N, device="cuda", dtype=torch.float32)

benchmark_kernels(
    {"softmax": lib["torch_softmax"],
     "online_softmax": lib["torch_online_softmax"]},
    lambda inp: F.softmax(inp, dim=-1), x, atol=1e-4, rtol=1e-4
)

# ===== Log Softmax =====
print(f"\n===== Log Softmax ({M}x{N}) =====")
benchmark_kernels(
    {"log_softmax": lib["torch_log_softmax"]},
    lambda inp: F.log_softmax(inp, dim=-1), x, atol=1e-4, rtol=1e-4
)
