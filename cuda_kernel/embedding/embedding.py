import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling embedding kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "embedding.cu"),
    funcs=["torch_embedding_float4", "torch_embedding_pack"],
)

n = 1024
vocab_size = 32768
emb_size = 1024  # must be divisible by 4

print(f"\n===== Embedding Lookup (n={n}, vocab={vocab_size}, emb={emb_size}) =====")

weight = torch.randn(vocab_size, emb_size, device="cuda", dtype=torch.float32)
idx = torch.randint(0, vocab_size, (n,), device="cuda", dtype=torch.int32)

def pytorch_embedding(idx, weight):
    return weight[idx.long()]

kernels = {
    "embedding_float4": lambda idx, weight: lib["torch_embedding_float4"](idx, weight),
    "embedding_pack":   lambda idx, weight: lib["torch_embedding_pack"](idx, weight),
}

benchmark_kernels(kernels, pytorch_embedding, idx, weight, atol=1e-5, rtol=1e-5)
