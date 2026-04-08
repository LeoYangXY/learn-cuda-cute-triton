"""
=============================================================================
验证 tv_layout = (8, (2, 16)):(32, (16, 1))
用 1D 的 8 个线程对 2D 的 16×16 矩阵做 elementwise add
=============================================================================
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

NUM_THREADS = 8


@cute.kernel
def add_2d_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
                  tv_layout: cute.Layout, tiler: cute.Shape):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blkA = cute.local_tile(gA, tiler, (bidx, 0))
    blkB = cute.local_tile(gB, tiler, (bidx, 0))
    blkC = cute.local_tile(gC, tiler, (bidx, 0))

    copy_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gA.element_type,
        num_bits_per_copy=32,  # 每次搬 1 个 float
    )

    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler)
    thr_copy = tiled_copy.get_slice(tidx)

    tAsA = thr_copy.partition_S(blkA)
    tBsB = thr_copy.partition_S(blkB)
    tCsC = thr_copy.partition_D(blkC)

    # GMEM → 寄存器
    rA = cute.make_fragment_like(tAsA)
    rB = cute.make_fragment_like(tBsB)
    cute.copy(tiled_copy, tAsA, rA)
    cute.copy(tiled_copy, tBsB, rB)

    # 逐元素加法
    rC = cute.make_fragment_like(rA)
    for i in range(cute.size(rC)):
        rC[i] = rA[i] + rB[i]

    # 寄存器 → GMEM
    cute.copy(tiled_copy, rC, tCsC)


@cute.jit
def add_2d_launch(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    # tv_layout = (8, (2, 16)) : (32, (16, 1))
    tv_layout = cute.make_layout(
        shape=(NUM_THREADS, (2, 16)),
        stride=(32, (16, 1))
    )
    tiler = (16, 16)  # 2D tiler: 16 行 × 16 列

    add_2d_kernel(mA, mB, mC, tv_layout, tiler).launch(
        grid=(1, 1, 1), block=(NUM_THREADS, 1, 1)
    )


# ============================================================================
# 主函数
# ============================================================================
cutlass.cuda.initialize_cuda_context()

# 真正的 2D tensor: 16×16
# A[i][j] = i * 16 + j (行优先)
A = torch.arange(256, dtype=torch.float32, device="cuda").reshape(16, 16)
B = torch.ones(16, 16, dtype=torch.float32, device="cuda")
C = torch.zeros(16, 16, dtype=torch.float32, device="cuda")

print("A (16×16):")
print(A.int())
print()
print("B (16×16, 全1):")
print(B.int())
print()

gA = from_dlpack(A)
gB = from_dlpack(B)
gC = from_dlpack(C)

add_2d_launch(gA, gB, gC)
torch.cuda.synchronize()

expected = A + B
match = torch.allclose(C, expected)

print("tv_layout = (8, (2, 16)):(32, (16, 1))")
print()
print("C (结果, 16×16):")
print(C.int())
print()
print(f"expected (期望, 16×16):")
print(expected.int())
print()
if match:
    print("✅ 结果正确! C == A + B")
else:
    diff = (C - expected).abs()
    print(f"❌ 结果不对! 最大误差: {diff.max().item()}")
    print("差异位置:")
    print(torch.nonzero(diff > 0))
