import cutlass.cute as cute

@cute.jit
def vec_add(A, B, C, num: int):
    # ❗ 不需要 make_shape
    gA = cute.make_tensor(A)
    gB = cute.make_tensor(B)
    gC = cute.make_tensor(C)

    # block = 128 threads
    block_layout = cute.make_layout(128)

    # tile
    bA = cute.local_tile(gA, 128, cute.block_idx.x)
    bB = cute.local_tile(gB, 128, cute.block_idx.x)
    bC = cute.local_tile(gC, 128, cute.block_idx.x)

    # partition
    tA = cute.local_partition(bA, block_layout, cute.thread_idx.x)
    tB = cute.local_partition(bB, block_layout, cute.thread_idx.x)
    tC = cute.local_partition(bC, block_layout, cute.thread_idx.x)

    # register
    rA = cute.make_fragment_like(tA)
    rB = cute.make_fragment_like(tB)
    rC = cute.make_fragment_like(tC)

    cute.copy(tA, rA)
    cute.copy(tB, rB)

    for i in range(cute.size(rA)):
        rC[i] = rA[i] + rB[i]

    cute.copy(rC, tC)

if __name__ == "__main__":
    import torch

    num = 1024
    A = torch.arange(num, device="cuda", dtype=torch.float32)
    B = 2 * torch.arange(num, device="cuda", dtype=torch.float32)
    C = torch.empty_like(A)

    vec_add(A, B, C, num)