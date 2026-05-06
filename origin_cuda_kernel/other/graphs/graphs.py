import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import load_cuda, check, timed, benchmark_kernels

print("Compiling graph kernels...")
lib = load_cuda(
    cuda_src=os.path.join(os.path.dirname(__file__), "graphs.cu"),
    funcs=[
        "torch_floyd_warshall",
        "torch_bellman_ford",
    ],
)

# ===== Floyd-Warshall =====
N = 512
print(f"\n===== Floyd-Warshall All-Pairs Shortest Path (N={N}) =====")

# Create random graph with some edges
INF = float('inf')
dist = torch.full((N, N), INF, device="cuda", dtype=torch.float32)
# Set diagonal to 0
for i in range(N):
    dist[i, i] = 0.0
# Add random edges
num_edges = N * 10
for _ in range(num_edges):
    i = torch.randint(0, N, (1,)).item()
    j = torch.randint(0, N, (1,)).item()
    if i != j:
        w = torch.rand(1).item() * 10 + 0.1
        dist[i, j] = min(dist[i, j].item(), w)

# PyTorch reference (CPU Floyd-Warshall for correctness)
def pytorch_floyd_warshall(d):
    d_np = d.cpu().numpy().copy()
    n = d_np.shape[0]
    for k in range(n):
        d_np = np.minimum(d_np, d_np[:, k:k+1] + d_np[k:k+1, :])
    return torch.from_numpy(d_np).cuda()

# Only test correctness on small graph
N_small = 128
dist_small = torch.full((N_small, N_small), 1e10, device="cuda", dtype=torch.float32)
for i in range(N_small):
    dist_small[i, i] = 0.0
for _ in range(N_small * 5):
    i = torch.randint(0, N_small, (1,)).item()
    j = torch.randint(0, N_small, (1,)).item()
    if i != j:
        w = torch.rand(1).item() * 10 + 0.1
        dist_small[i, j] = min(dist_small[i, j].item(), w)

print("Correctness check on small graph (N=128):")
result_cuda = lib["torch_floyd_warshall"](dist_small)
result_ref = pytorch_floyd_warshall(dist_small)
max_diff = (result_cuda - result_ref).abs().max().item()
print(f"  max_diff = {max_diff:.6e} {'✅ PASS' if max_diff < 1e-3 else '❌ FAIL'}")

# Benchmark on larger graph
print(f"\nBenchmark Floyd-Warshall (N={N}):")
_, ms = timed(lib["torch_floyd_warshall"], dist, warmup=2, rep=5)
print(f"  CUDA: {ms:.4f} ms")

# ===== Bellman-Ford =====
print(f"\n===== Bellman-Ford SSSP =====")
N_bf = 1024
E_bf = N_bf * 20
src_list = torch.randint(0, N_bf, (E_bf,), device="cuda", dtype=torch.int32)
dst_list = torch.randint(0, N_bf, (E_bf,), device="cuda", dtype=torch.int32)
w_list = torch.rand(E_bf, device="cuda", dtype=torch.float32) * 10 + 0.1
source = 0

result_bf = lib["torch_bellman_ford"](src_list, dst_list, w_list, N_bf, source)
print(f"  Bellman-Ford computed (N={N_bf}, E={E_bf})")
print(f"  Reachable nodes: {(result_bf < 1e10).sum().item()}/{N_bf}")

_, ms = timed(lib["torch_bellman_ford"], src_list, dst_list, w_list, N_bf, source, warmup=2, rep=10)
print(f"  Time: {ms:.4f} ms")
