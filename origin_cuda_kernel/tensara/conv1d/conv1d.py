import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling conv1d kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "conv1d.cu"),
    funcs=[
        "torch_conv1d_naive",
        "torch_conv1d_shared",
        "torch_conv1d_const",
        "torch_conv1d_vec4",
    ],
)

def pytorch_conv1d(input_t, kernel_t):
    # Our CUDA kernel computes correlation (no kernel flip)
    # F.conv1d also does correlation by default
    return torch.nn.functional.conv1d(
        input_t.view(1, 1, -1), kernel_t.view(1, 1, -1)
    ).view(-1)


# ===== Test 1: Small kernel =====
N = 4 * 1024 * 1024
K = 7
print(f"\n===== 1D Convolution (N={N}, K={K}) =====")
x = torch.randn(N, device="cuda", dtype=torch.float32)
w = torch.randn(K, device="cuda", dtype=torch.float32)

kernels = {
    "naive": lambda inp, ker: lib["torch_conv1d_naive"](inp, ker),
    "shared": lambda inp, ker: lib["torch_conv1d_shared"](inp, ker),
    "const": lambda inp, ker: lib["torch_conv1d_const"](inp, ker),
    "vec4": lambda inp, ker: lib["torch_conv1d_vec4"](inp, ker),
}
benchmark_kernels(kernels, pytorch_conv1d, x, w, atol=1e-3, rtol=1e-3)

# ===== Test 2: Larger kernel =====
K2 = 64
print(f"\n===== 1D Convolution (N={N}, K={K2}) =====")
w2 = torch.randn(K2, device="cuda", dtype=torch.float32)

kernels2 = {
    "naive": lambda inp, ker: lib["torch_conv1d_naive"](inp, ker),
    "shared": lambda inp, ker: lib["torch_conv1d_shared"](inp, ker),
    "const": lambda inp, ker: lib["torch_conv1d_const"](inp, ker),
    "vec4": lambda inp, ker: lib["torch_conv1d_vec4"](inp, ker),
}
benchmark_kernels(kernels2, pytorch_conv1d, x, w2, atol=1e-3, rtol=1e-3)
