import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, check, timed, benchmark_kernels

# Load CUDA kernels
print("Compiling add kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "add.cu"),
    funcs=[
        "torch_eltwise_add_f32_scalar",
        "torch_eltwise_add_f32_vector",
        "torch_eltwise_add_f16_scalar",
        "torch_eltwise_add_f16_vec2",
        "torch_eltwise_add_f16_vec2_4",
        "torch_eltwise_add_f16_vec2_4_unroll",
        "torch_eltwise_add_f16_pack",
    ],
)

def run_f32_kernel(fn, a, b):
    c = torch.empty_like(a)
    fn(a, b, c)
    return c

def run_f16_kernel(fn, a, b):
    c = torch.empty_like(a)
    fn(a, b, c)
    return c

def pytorch_add(a, b):
    return a + b


# ===== FP32 test =====
N = 64 * 1024 * 1024
print(f"\n===== FP32 Elementwise Add (N={N}) =====")
a_f32 = torch.randn(N, device="cuda", dtype=torch.float32)
b_f32 = torch.randn(N, device="cuda", dtype=torch.float32)

f32_kernels = {
    "f32_scalar": lambda a, b: run_f32_kernel(lib["torch_eltwise_add_f32_scalar"], a, b),
    "f32_vector": lambda a, b: run_f32_kernel(lib["torch_eltwise_add_f32_vector"], a, b),
}
benchmark_kernels(f32_kernels, pytorch_add, a_f32, b_f32, atol=1e-5, rtol=1e-5)

# ===== FP16 test =====
print(f"\n===== FP16 Elementwise Add (N={N}) =====")
# N must be divisible by 8 for pack kernels
N_f16 = 64 * 1024 * 1024
a_f16 = torch.randn(N_f16, device="cuda", dtype=torch.float16)
b_f16 = torch.randn(N_f16, device="cuda", dtype=torch.float16)

f16_kernels = {
    "f16_scalar": lambda a, b: run_f16_kernel(lib["torch_eltwise_add_f16_scalar"], a, b),
    "f16_vec2": lambda a, b: run_f16_kernel(lib["torch_eltwise_add_f16_vec2"], a, b),
    "f16_vec2_4": lambda a, b: run_f16_kernel(lib["torch_eltwise_add_f16_vec2_4"], a, b),
    "f16_vec2_4_unroll": lambda a, b: run_f16_kernel(lib["torch_eltwise_add_f16_vec2_4_unroll"], a, b),
    "f16_pack": lambda a, b: run_f16_kernel(lib["torch_eltwise_add_f16_pack"], a, b),
}
benchmark_kernels(f16_kernels, pytorch_add, a_f16, b_f16, atol=1e-2, rtol=1e-2)
