import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling conv2d kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "conv2d.cu"),
    funcs=[
        "torch_conv2d_naive",
        "torch_conv2d_shared",
    ],
)

H, W = 2048, 2048
KH, KW = 3, 3
print(f"\n===== 2D Convolution ({H}x{W}, kernel {KH}x{KW}) =====")
x = torch.randn(H, W, device="cuda", dtype=torch.float32)
w = torch.randn(KH, KW, device="cuda", dtype=torch.float32)

def pytorch_conv2d(inp, ker):
    # PyTorch conv2d expects (N, C, H, W) input and (out_C, in_C, KH, KW) weight
    # For correlation (no flip), use conv2d with flipped kernel
    return F.conv2d(
        inp.view(1, 1, H, W),
        ker.view(1, 1, KH, KW)
    ).view(H - KH + 1, W - KW + 1)

benchmark_kernels(
    {"conv2d_naive": lambda inp, ker: lib["torch_conv2d_naive"](inp, ker),
     "conv2d_shared": lambda inp, ker: lib["torch_conv2d_shared"](inp, ker)},
    pytorch_conv2d, x, w, atol=1e-3, rtol=1e-3
)

# Larger kernel
KH2, KW2 = 5, 5
print(f"\n===== 2D Convolution ({H}x{W}, kernel {KH2}x{KW2}) =====")
w2 = torch.randn(KH2, KW2, device="cuda", dtype=torch.float32)

def pytorch_conv2d_5x5(inp, ker):
    return F.conv2d(
        inp.view(1, 1, H, W),
        ker.view(1, 1, KH2, KW2)
    ).view(H - KH2 + 1, W - KW2 + 1)

benchmark_kernels(
    {"conv2d_naive": lambda inp, ker: lib["torch_conv2d_naive"](inp, ker),
     "conv2d_shared": lambda inp, ker: lib["torch_conv2d_shared"](inp, ker)},
    pytorch_conv2d_5x5, x, w2, atol=1e-3, rtol=1e-3
)
