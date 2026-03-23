import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling layer_norm kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "layer_norm.cu"),
    funcs=["torch_layer_norm_f32", "torch_layer_norm_float4"],
)

N = 128
K = 256  # must be divisible by 4, and <= 1024
g = 1.2
b = 0.5

print(f"\n===== Layer Norm (N={N}, K={K}, g={g}, b={b}) =====")

x = torch.randn(N, K, device="cuda", dtype=torch.float32)

def pytorch_layer_norm(x, g, b):
    """Simple scalar-affine layer norm matching the kernel's behavior."""
    eps = 1e-5
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * g + b

kernels = {
    "layer_norm_f32":    lambda x, g, b: lib["torch_layer_norm_f32"](x, g, b),
    "layer_norm_float4": lambda x, g, b: lib["torch_layer_norm_float4"](x, g, b),
}

benchmark_kernels(kernels, pytorch_layer_norm, x, g, b, atol=1e-4, rtol=1e-4)
