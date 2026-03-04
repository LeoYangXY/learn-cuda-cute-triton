from utils import auto_tune_and_benchmark
import torch
import triton
import triton.language as tl




@triton.jit
def reduce_max(x_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE:tl.constexpr,
):
    pid=tl.program_id(0)
    block_start = pid*BLOCK_SIZE

    block_tensor = block_start+tl.arrange(0,BLOCK_SIZE)

