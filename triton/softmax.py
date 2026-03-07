from utils import auto_tune_and_benchmark
import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources



#这个kernel没有显式声明shared memory，默认是全部使用寄存器去完成相关操作
@triton.jit
def softmax_kernel(x_ptr,
               output_ptr,
               M, N, # x的shape为(M, N) 
               BLOCK_SIZE:tl.constexpr,
):
    #划分目标：每个block处理1行
    pid=tl.program_id(0)
    block_start = x_ptr + pid*N
    
    #生成当前block负责的指针块
    block_tensor = block_start + tl.arange(0, BLOCK_SIZE)

    # 创建 Mask
    # 只有在列索引小于 N 时，指针才是有效的
    # 这里的 mask 是一个形状为 (BLOCK_SIZE,) 的布尔张量
    mask = tl.arange(0, BLOCK_SIZE) < N

    # 加载数据:
    row = tl.load(block_tensor,mask=mask, other=float('-inf')) # 负无穷在softmax中是安全的，因为exp(-inf)=0

    # 用triton在 block 维度上写 tl.max(row, axis=0)，就表示“这个 block 要对它负责的数据求 max”，
    # 最终整个 block 会得到一个结果，存到row_max中
    # 而到底用了多少 threads、怎么协作、是否向量化 —— 这些是 Triton 编译器和 GPU 硬件的事，你不用操心。
    row_max = tl.max(row, axis=0)

    #tl.max的axis的理解：
    #axis=xx，就代表对于input的tensor的xx维做压缩，最后去掉这个维度
    #比如上面的input是一个(block_size,)的tensor，axis=0就代表对这个tensor的block_size所属的那个维度求max，最后得到一个标量（0维的tensor）
    #如果input是(64, 16)：
    #axis=0代表按照行方向去压缩，也就是相当于blockIdx.y的那个往下的方向，也就是求出所有行的第一列的max，然后求出所有行的第二列的max，以此类推，最后得到一个(16,)的tensor；
    #axis=1代表按照列方向去压缩，也就是相当于blockIdx.x的那个往右的方向，求出每一行的max，最后得到一个(64,)的tensor

    #row是一个tensor，row_max是一个scaler，triton跟pytorch很像，里面的广播操作用得很多  
    row_exp = tl.exp(row - row_max) 

    row_sum = tl.sum(row_exp, axis=0)

    row_output_tensor = row_exp / row_sum

    # 最后把结果写回全局内存
    output_row_start = output_ptr + pid*N
    output_tensor = output_row_start + tl.arange(0, BLOCK_SIZE)
    tl.store(output_tensor, row_output_tensor, mask=mask)
    


#注意下面的自动overlap机制
@triton.jit
def softmax_multi_rows_kernel(x_ptr,
               output_ptr,
               M, N, 
               BLOCK_SIZE:tl.constexpr,
               num_stages:tl.constexpr):#相当于triton帮我们去自动化的模拟出double buffer，triple buffer等任意的多buffer机制，让我们能做overlap
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)


    #使用num_stages,相当于如下的cuda伪代码：
    # 比如num_stages==3:Triton 为 num_stages=3 自动生成的核心逻辑（简化版）：
    # for (int i = row_start; i < num_rows; i += row_step) {
    #     // 1. 【Stage 0】提前加载第 i 行（非阻塞）
    #     buf[i % num_stages] = async_load(global_ptr + i * N);

    #     // 2. 【Stage 1】等待第 i-row_step*num_stages 行数据就绪，并计算
    #     wait(buf[(i-row_step*num_stages) % num_stages]);          // ← 插入的同步点
    #     compute_softmax(buf[(i-row_step*num_stages) % num_stages]);

    #     // 3. 【Stage 2】存储第 i-row_step*num_stages 行结果
    #     store(global_out + (i-row_step*num_stages) * N, buf[(i-row_step*num_stages) % num_stages]);
    # }
    # // → 实现了：load(i) || compute(i-row_step*num_stages) || store(i-row_step*num_stages) 三者并行
    for row_idx in tl.range(row_start, M, row_step,num_stages=num_stages):#注意要使用tl.range而不是普通的range
        cur_row_start = x_ptr + row_idx*N
        cur_row_tensor = cur_row_start + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < N

        cur_row_tensor = tl.load(cur_row_tensor,mask=mask, other=float('-inf')) # 负无穷在softmax中是安全的，因为exp(-inf)=0
        
        cur_row_max = tl.max(cur_row_tensor, axis=0)
        cur_row_sum = tl.sum(tl.exp(cur_row_tensor - cur_row_max), axis=0)
        cur_row_output_tensor = tl.exp(cur_row_tensor - cur_row_max) / cur_row_sum
        
        output_row_start = output_ptr + row_idx*N
        output_row_tensor = output_row_start + tl.arange(0, BLOCK_SIZE)
        tl.store(output_row_tensor, cur_row_output_tensor, mask=mask)



#tl.max的底层其实就是用reduceMaxOp来实现的，然后其实就是模拟的是cuda代码：
# eg：
# //每个thread负责读取4个元素，然后再做2层规约
# __global__ void max_kernel_shuffle_float4(float* input, float* output, int N) {
#     constexpr int kWarpSize = 32;
#     __shared__ float sdata[32]; // 最多 32 warps

#     int tid = threadIdx.x;
#     int bid = blockIdx.x;
#     int global_start = (bid * blockDim.x + tid) * 4; // 每个线程负责 4 个 float

#     int warp_idx = tid / kWarpSize;
#     int lane = tid % kWarpSize;

#     // Step 1: Load 4 floats as float4 (128-bit load)
#     float4 vec= *reinterpret_cast<float4*>(input + global_start);

#     // Step 2: 求这 4 个值的最大值
#     float local_max = fmaxf(fmaxf(vec.x, vec.y), fmaxf(vec.z, vec.w));

#     // Step 3: Warp 内规约
#     local_max = warp_reduce_max_f32(local_max);

#     // Step 4: 每个 warp 的 leader 写入 shared memory
#     if (lane == 0) {
#         sdata[warp_idx] = local_max;
#     }
#     __syncthreads();

#     // Step 5: 第一个 warp 对 sdata 做最终规约
#     if (warp_idx == 0) {
#         float val = (lane < (blockDim.x + kWarpSize - 1) / kWarpSize) ? sdata[lane] : -FLT_MAX;
#         val = warp_reduce_max_f32(val);
#         if (lane == 0) {
#             atomicMax(output, val);
#         }
#     }
# }
# 在triton中，reduceMaxOp的实现也是：
# 先做thread内的规约，然后是做warp内的规约（用warp shuffle），然后是把数据放到shared_memory，再用一个warp去做warp shuffle
# 只是在调用shared_memory的时候，triton是插入一个算子，在后续的pass中进行申请，而不是能像写cuda一样显式的申请



#host侧调用的时候：让blockSize ≥ N 且是 2 的幂
def triton_softmax(x, BLOCK_SIZE=None):
    M, N = x.shape
    output = torch.empty_like(x)
    if BLOCK_SIZE is None:
        BLOCK_SIZE = 2 ** (N.bit_length() - 1) if N & (N - 1) else N
        # 确保 BLOCK_SIZE >= N
        if BLOCK_SIZE < N:
            BLOCK_SIZE *= 2
    grid = lambda meta: (M,)
    softmax_kernel[grid](x, output, M, N, BLOCK_SIZE=BLOCK_SIZE)
    return output


def triton_softmax_multi_rows(x, BLOCK_SIZE=None, num_stages=2):
    M, N = x.shape
    output = torch.empty_like(x)
    if BLOCK_SIZE is None:
        BLOCK_SIZE = 2 ** (N.bit_length() - 1) if N & (N - 1) else N
        # 确保 BLOCK_SIZE >= N
        if BLOCK_SIZE < N:
            BLOCK_SIZE *= 2
    grid = lambda meta: (M,)
    softmax_multi_rows_kernel[grid](x, output, M, N, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return output


def _next_power_of_2(n):
    return 1 << (n - 1).bit_length()


def prepare_softmax_data(n_cols, device, dtype, n_rows=1024):
    x = torch.randn(n_rows, n_cols, device=device, dtype=dtype)
    out = torch.empty_like(x)
    return {"x": x, "out": out}


def launch_softmax_single_row(data, BLOCK_SIZE):
    x = data["x"]
    out = data["out"]
    m, n = x.shape
    min_required_bs = _next_power_of_2(n)
    effective_bs = max(BLOCK_SIZE, min_required_bs)
    grid = (m,)
    softmax_kernel[grid](x, out, m, n, BLOCK_SIZE=effective_bs)


def launch_softmax_multi_rows_2stages(data, BLOCK_SIZE):
    x = data["x"]
    out = data["out"]
    m, n = x.shape
    min_required_bs = _next_power_of_2(n)
    effective_bs = max(BLOCK_SIZE, min_required_bs)
    grid = (m,)
    softmax_multi_rows_kernel[grid](
                x,
                out,
                m,
                n,
                BLOCK_SIZE=effective_bs,
                num_stages=2,
            )


def launch_softmax_multi_rows_3stages(data, BLOCK_SIZE):
    x = data["x"]
    out = data["out"]
    m, n = x.shape
    min_required_bs = _next_power_of_2(n)
    effective_bs = max(BLOCK_SIZE, min_required_bs)
    grid = (m,)
    softmax_multi_rows_kernel[grid](
                x,
                out,
                m,
                n,
                BLOCK_SIZE=effective_bs,
                num_stages=3,
            )


def launch_torch_softmax(data):
    return torch.softmax(data["x"], dim=-1)


# =============== 性能对比测试（纯kernel计时） =================
if __name__ == "__main__":
    sizes_to_test = [2**i for i in range(10, 14)]  # N: 1024 ~ 16384
    block_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

    print("=" * 80)
    print("Softmax 单行Kernel vs PyTorch（kernel-only + autotune）")
    print("=" * 80)
    auto_tune_and_benchmark(
        triton_launch=launch_softmax_single_row,
        torch_launch=launch_torch_softmax,
        sizes=sizes_to_test,
        block_sizes=block_sizes,
        prepare_data_fn=prepare_softmax_data,
        triton_output_getter=lambda data: data["out"],
        correctness_size=2048,
        atol=1e-5,
        use_do_bench=True,
    )

    print("\n" + "=" * 80)
    print("Softmax 多行Kernel （2 stages） vs PyTorch（kernel-only + autotune）")
    print("=" * 80)
    auto_tune_and_benchmark(
        triton_launch=launch_softmax_multi_rows_2stages,
        torch_launch=launch_torch_softmax,
        sizes=sizes_to_test,
        block_sizes=block_sizes,
        prepare_data_fn=prepare_softmax_data,
        triton_output_getter=lambda data: data["out"],
        correctness_size=2048,
        atol=1e-5,
        use_do_bench=True,
    )

    print("\n" + "=" * 80)
    print("Softmax 多行Kernel（3 stages） vs PyTorch（kernel-only +autotune）")
    print("=" * 80)
    auto_tune_and_benchmark(
        triton_launch=launch_softmax_multi_rows_3stages,
        torch_launch=launch_torch_softmax,
        sizes=sizes_to_test,
        block_sizes=block_sizes,
        prepare_data_fn=prepare_softmax_data,
        triton_output_getter=lambda data: data["out"],
        correctness_size=2048,
        atol=1e-5,
        use_do_bench=True,
    )