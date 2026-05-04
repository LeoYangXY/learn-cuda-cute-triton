import sys
import os
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling activation kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "activations.cu"),
    funcs=[
        "torch_relu_f32", "torch_relu_f32x4",
        "torch_leaky_relu_f32", "torch_leaky_relu_f32x4",
        "torch_gelu_f32", "torch_gelu_f32x4",
        "torch_sigmoid_f32", "torch_sigmoid_f32x4",
        "torch_tanh_f32", "torch_tanh_f32x4",
        "torch_elu_f32", "torch_elu_f32x4",
        "torch_selu_f32", "torch_selu_f32x4",
        "torch_softplus_f32", "torch_softplus_f32x4",
        "torch_swish_f32", "torch_swish_f32x4",
        "torch_hard_sigmoid_f32", "torch_hard_sigmoid_f32x4",
    ],
)

M, N_dim = 8192, 8192
total = M * N_dim
x = torch.randn(total, device="cuda", dtype=torch.float32)

# ===== ReLU =====
print(f"\n===== ReLU ({M}x{N_dim}) =====")
benchmark_kernels(
    {"relu_f32": lib["torch_relu_f32"], "relu_f32x4": lib["torch_relu_f32x4"]},
    F.relu, x, atol=1e-5, rtol=1e-5
)

# ===== Leaky ReLU =====
print(f"\n===== Leaky ReLU ({M}x{N_dim}, alpha=0.01) =====")
alpha = 0.01
benchmark_kernels(
    {"leaky_relu_f32": lambda inp: lib["torch_leaky_relu_f32"](inp, alpha),
     "leaky_relu_f32x4": lambda inp: lib["torch_leaky_relu_f32x4"](inp, alpha)},
    lambda inp: F.leaky_relu(inp, alpha), x, atol=1e-5, rtol=1e-5
)

# ===== GELU =====
print(f"\n===== GELU ({M}x{N_dim}) =====")
benchmark_kernels(
    {"gelu_f32": lib["torch_gelu_f32"], "gelu_f32x4": lib["torch_gelu_f32x4"]},
    lambda inp: F.gelu(inp, approximate='tanh'), x, atol=1e-4, rtol=1e-4
)

# ===== Sigmoid =====
print(f"\n===== Sigmoid ({M}x{N_dim}) =====")
benchmark_kernels(
    {"sigmoid_f32": lib["torch_sigmoid_f32"], "sigmoid_f32x4": lib["torch_sigmoid_f32x4"]},
    torch.sigmoid, x, atol=1e-5, rtol=1e-5
)

# ===== Tanh =====
print(f"\n===== Tanh ({M}x{N_dim}) =====")
benchmark_kernels(
    {"tanh_f32": lib["torch_tanh_f32"], "tanh_f32x4": lib["torch_tanh_f32x4"]},
    torch.tanh, x, atol=1e-5, rtol=1e-5
)

# ===== ELU =====
print(f"\n===== ELU ({M}x{N_dim}, alpha=1.0) =====")
benchmark_kernels(
    {"elu_f32": lambda inp: lib["torch_elu_f32"](inp, 1.0),
     "elu_f32x4": lambda inp: lib["torch_elu_f32x4"](inp, 1.0)},
    lambda inp: F.elu(inp, alpha=1.0), x, atol=1e-5, rtol=1e-5
)

# ===== SELU =====
print(f"\n===== SELU ({M}x{N_dim}) =====")
benchmark_kernels(
    {"selu_f32": lib["torch_selu_f32"], "selu_f32x4": lib["torch_selu_f32x4"]},
    F.selu, x, atol=1e-4, rtol=1e-4
)

# ===== Softplus =====
print(f"\n===== Softplus ({M}x{N_dim}) =====")
benchmark_kernels(
    {"softplus_f32": lib["torch_softplus_f32"], "softplus_f32x4": lib["torch_softplus_f32x4"]},
    F.softplus, x, atol=1e-4, rtol=1e-4
)

# ===== Swish =====
print(f"\n===== Swish ({M}x{N_dim}) =====")
benchmark_kernels(
    {"swish_f32": lib["torch_swish_f32"], "swish_f32x4": lib["torch_swish_f32x4"]},
    F.silu, x, atol=1e-5, rtol=1e-5
)

# ===== Hard Sigmoid =====
print(f"\n===== Hard Sigmoid ({M}x{N_dim}) =====")
benchmark_kernels(
    {"hard_sigmoid_f32": lib["torch_hard_sigmoid_f32"],
     "hard_sigmoid_f32x4": lib["torch_hard_sigmoid_f32x4"]},
    F.hardsigmoid, x, atol=1e-5, rtol=1e-5
)
