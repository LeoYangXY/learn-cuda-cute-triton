import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling reduce_max kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "reduce_max.cu"),
    funcs=[
        "torch_reduce_max_naive",
        "torch_reduce_max_shared",
        "torch_reduce_max_shuffle",
        "torch_reduce_max_shuffle_float4",
        "torch_reduce_max_shuffle_pack_half",
    ],
)

def pytorch_max(x):
    return x.max().unsqueeze(0)

# ===== FP32 test =====
# N must be divisible by 4 for float4 kernel
N = 1024 * 1024
print(f"\n===== FP32 Reduce Max (N={N}) =====")
x_f32 = torch.randn(N, device="cuda", dtype=torch.float32)

f32_kernels = {
    "naive":         lib["torch_reduce_max_naive"],
    "shared_only":   lib["torch_reduce_max_shared"],
    "shuffle":       lib["torch_reduce_max_shuffle"],
    "shuffle_float4": lib["torch_reduce_max_shuffle_float4"],
}
benchmark_kernels(f32_kernels, pytorch_max, x_f32, atol=1e-5, rtol=1e-5)

# ===== FP16 test (pack_half kernel) =====
# N must be divisible by 8 for pack kernel
N_f16 = 1024 * 1024
print(f"\n===== FP16 Reduce Max (N={N_f16}) =====")
x_f16 = torch.randn(N_f16, device="cuda", dtype=torch.float16)

def pytorch_max_f16_to_f32(x):
    return x.float().max().unsqueeze(0)

f16_kernels = {
    "shuffle_pack_half": lib["torch_reduce_max_shuffle_pack_half"],
}
benchmark_kernels(f16_kernels, pytorch_max_f16_to_f32, x_f16, atol=1e-2, rtol=1e-2)
