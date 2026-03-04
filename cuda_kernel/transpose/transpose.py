import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, benchmark_kernels

lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "transpose.cu"),
    funcs=[
        "torch_transpose_naive",
        "torch_transpose_4float",
        "torch_transpose_shared",
        "torch_transpose_shared_padding",
    ],
    verbose=True,
)

# ---------- test ----------
M, N = 1024, 2048
src = torch.randn(M, N, device="cuda", dtype=torch.float32)
ref_fn = lambda x: x.t().contiguous()

kernels = {
    "naive":          lambda x: lib["torch_transpose_naive"](x),
    "4float":         lambda x: lib["torch_transpose_4float"](x),
    "shared":         lambda x: lib["torch_transpose_shared"](x),
    "shared_padding": lambda x: lib["torch_transpose_shared_padding"](x),
}

print("\n===== Transpose (M={}, N={}) =====".format(M, N))
benchmark_kernels(kernels, ref_fn, src)
