import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling conv3d kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "conv3d.cu"),
    funcs=[
        "torch_conv3d",
        "torch_max_pool3d",
        "torch_avg_pool3d",
    ],
)

# ===== 3D Convolution =====
D, H, W = 64, 64, 64
KD, KH, KW = 3, 3, 3
print(f"\n===== 3D Convolution ({D}x{H}x{W}, kernel {KD}x{KH}x{KW}) =====")
x3d = torch.randn(D, H, W, device="cuda", dtype=torch.float32)
w3d = torch.randn(KD, KH, KW, device="cuda", dtype=torch.float32)

def pytorch_conv3d(inp, ker):
    return F.conv3d(
        inp.view(1, 1, D, H, W),
        ker.view(1, 1, KD, KH, KW)
    ).view(D - KD + 1, H - KH + 1, W - KW + 1)

benchmark_kernels(
    {"conv3d": lambda inp, ker: lib["torch_conv3d"](inp, ker)},
    pytorch_conv3d, x3d, w3d, atol=1e-3, rtol=1e-3
)

# ===== 3D Max Pooling =====
SD, SH, SW = 2, 2, 2
KD_p, KH_p, KW_p = 2, 2, 2
print(f"\n===== 3D Max Pooling ({D}x{H}x{W}, K={KD_p}x{KH_p}x{KW_p}, S={SD}x{SH}x{SW}) =====")

def pytorch_max_pool3d(inp):
    return F.max_pool3d(
        inp.view(1, 1, D, H, W), (KD_p, KH_p, KW_p), (SD, SH, SW)
    ).view((D-KD_p)//SD+1, (H-KH_p)//SH+1, (W-KW_p)//SW+1)

benchmark_kernels(
    {"max_pool3d": lambda inp: lib["torch_max_pool3d"](inp, KD_p, KH_p, KW_p, SD, SH, SW)},
    pytorch_max_pool3d, x3d, atol=1e-5, rtol=1e-5
)

# ===== 3D Average Pooling =====
print(f"\n===== 3D Average Pooling ({D}x{H}x{W}, K={KD_p}x{KH_p}x{KW_p}, S={SD}x{SH}x{SW}) =====")

def pytorch_avg_pool3d(inp):
    return F.avg_pool3d(
        inp.view(1, 1, D, H, W), (KD_p, KH_p, KW_p), (SD, SH, SW)
    ).view((D-KD_p)//SD+1, (H-KH_p)//SH+1, (W-KW_p)//SW+1)

benchmark_kernels(
    {"avg_pool3d": lambda inp: lib["torch_avg_pool3d"](inp, KD_p, KH_p, KW_p, SD, SH, SW)},
    pytorch_avg_pool3d, x3d, atol=1e-4, rtol=1e-4
)
