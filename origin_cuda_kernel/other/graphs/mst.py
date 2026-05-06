import torch
from torch.utils.cpp_extension import load
import os

# Build the CUDA extension
current_dir = os.path.dirname(os.path.abspath(__file__))
mst_module = load(
    name='mst_module',
    sources=[os.path.join(current_dir, 'mst.cu')],
    extra_cuda_cflags=['-O3', '--use_fast_math'],
    build_directory=os.path.join(current_dir, 'build_mst'),
    verbose=False
)

def test_mst():
    print("=" * 50)
    print("Testing Minimum Spanning Tree (Borůvka)")
    print("=" * 50)

    # Create a small graph as adjacency matrix
    N = 6
    INF = float('inf')
    adj = torch.full((N, N), INF, device='cuda')

    # Add edges (undirected)
    edges = [
        (0, 1, 4), (0, 2, 3), (1, 2, 1),
        (1, 3, 2), (2, 3, 4), (3, 4, 2),
        (4, 5, 6), (2, 5, 5),
    ]
    for u, v, w in edges:
        adj[u, v] = w
        adj[v, u] = w

    mst_weight = mst_module.torch_mst(adj)
    print(f"MST total weight: {mst_weight.item():.2f}")
    # Expected: 1 + 2 + 2 + 3 + 5 = 13 (or similar depending on MST)
    # Edges: (1,2,1), (1,3,2), (3,4,2), (0,2,3), (2,5,5) = 13
    print(f"Expected MST weight: 13.0")

if __name__ == '__main__':
    test_mst()
    print("\nMST test done!")
