# add.py
from utils import auto_tune_and_benchmark
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,#为什么 BLOCK_SIZE 必须是 constexpr？ 这是因为tl.arange(start, end) 要求 end 是编译期常量，因为 Triton 需要知道这个 tensor 的 shape（长度） 来分配寄存器。如果 BLOCK_SIZE 是 runtime 变量（比如从 host 传进来的一个普通 int），Triton 无法在编译时确定 offsets 有多长 → 编译失败。
):
    #去划分每个block需要处理的数据start位置
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    #用tensor视角表达出这个block要处理的数据范围
    offsets = block_start + tl.arange(0, BLOCK_SIZE)#一个指针配合上一个tensor的语义:相当于这个tensor里面所有的元素都加上了block_start的偏移
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def triton_add(x, y, BLOCK_SIZE=1024):
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output



if __name__ == "__main__":
    sizes_to_test = [2**i for i in range(12, 26)]  # 4K to 32M
    auto_tune_and_benchmark(
        triton_func=triton_add,
        torch_func=lambda x, y: x + y,
        sizes=sizes_to_test,
        block_sizes=[64, 128, 256, 512, 1024],
    )