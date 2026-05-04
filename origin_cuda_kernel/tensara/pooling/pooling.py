import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling pooling kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "pooling.cu"),
    funcs=[
        "torch_avg_pool1d",
        "torch_avg_pool1d_vec4",
        "torch_max_pool1d",
        "torch_max_pool1d_vec4",
        "torch_avg_pool2d",
        "torch_max_pool2d",
    ],
)

# ===== 1D Average Pooling =====
N = 4 * 1024 * 1024
K, S = 7, 1
print(f"\n===== 1D Average Pooling (N={N}, K={K}, S={S}) =====")
x1d = torch.randn(N, device="cuda", dtype=torch.float32)

def pytorch_avg_pool1d(inp):
    return F.avg_pool1d(inp.view(1, 1, -1), K, S).view(-1)

benchmark_kernels(
    {"avg_pool1d": lambda inp: lib["torch_avg_pool1d"](inp, K, S),
     "avg_pool1d_vec4": lambda inp: lib["torch_avg_pool1d_vec4"](inp, K, S)},
    pytorch_avg_pool1d, x1d, atol=1e-4, rtol=1e-4
)

# ===== 1D Max Pooling =====
print(f"\n===== 1D Max Pooling (N={N}, K={K}, S={S}) =====")

def pytorch_max_pool1d(inp):
    return F.max_pool1d(inp.view(1, 1, -1), K, S).view(-1)

benchmark_kernels(
    {"max_pool1d": lambda inp: lib["torch_max_pool1d"](inp, K, S),
     "max_pool1d_vec4": lambda inp: lib["torch_max_pool1d_vec4"](inp, K, S)},
    pytorch_max_pool1d, x1d, atol=1e-5, rtol=1e-5
)

# ===== 2D Average Pooling =====
H, W = 4096, 4096
KH, KW, SH, SW = 3, 3, 1, 1
print(f"\n===== 2D Average Pooling ({H}x{W}, K={KH}x{KW}, S={SH}x{SW}) =====")
x2d = torch.randn(H, W, device="cuda", dtype=torch.float32)

def pytorch_avg_pool2d(inp):
    return F.avg_pool2d(inp.view(1, 1, H, W), (KH, KW), (SH, SW), padding=0).view(
        (H - KH) // SH + 1, (W - KW) // SW + 1
    )

benchmark_kernels(
    {"avg_pool2d": lambda inp: lib["torch_avg_pool2d"](inp, KH, KW, SH, SW)},
    pytorch_avg_pool2d, x2d, atol=1e-4, rtol=1e-4
)

# ===== 2D Max Pooling =====
print(f"\n===== 2D Max Pooling ({H}x{W}, K={KH}x{KW}, S={SH}x{SW}) =====")

def pytorch_max_pool2d(inp):
    return F.max_pool2d(inp.view(1, 1, H, W), (KH, KW), (SH, SW), padding=0).view(
        (H - KH) // SH + 1, (W - KW) // SW + 1
    )

benchmark_kernels(
    {"max_pool2d": lambda inp: lib["torch_max_pool2d"](inp, KH, KW, SH, SW)},
    pytorch_max_pool2d, x2d, atol=1e-5, rtol=1e-5
)
