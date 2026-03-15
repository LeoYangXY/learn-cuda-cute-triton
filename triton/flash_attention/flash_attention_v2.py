import torch
import triton
import triton.language as tl


# 1个block负责一个 Qi（即 Q 的第 i 个块，大小为 Br × d）。
# 这个 Qi 会被加载到 SRAM 并一直保留，直到算完所有相关的 K, V。
@triton.jit
def _flash_attn_fwd_kernel(
    # --- 数据指针 (Pointers) ---
    Q_ptr, K_ptr, V_ptr,
    L_ptr, M_ptr,      # 辅助状态 (Logsumexp, Max)
    O_ptr,           # 输出
    
    # --- 标量参数 (Scalars) ---
    sm_scale,          # softmax scale (通常是 1/sqrt(d))
    
    # --- 逻辑维度 (Logical Dimensions) ---
    B: tl.constexpr,   # Batch Size 
    H: tl.constexpr,   # Num Heads
    N: tl.constexpr,   # Seq Len 
    d: tl.constexpr,   # Head Dim (这是每一个head的)
    
    # --- 分块大小 (Block Sizes) ---
    Br: tl.constexpr,  # 对于Q的分块
    Bc: tl.constexpr,  # 对于K和V的分块
):
    #flash-attention-v2：让每个block负责一个Q的一个块，而不是一个block负责一个Q
    
    #我们之前使用2D grid，也就是pid_BH = tl.program_id(0);pid_q = tl.program_id(1)来分解的，
    #用ncu分析发现对于L2 cache的利用率过低，发现就是这里数据划分的问题：
    #这里的写法是tl.program_id(0)对应blockIdx.x,tl.program_id(1)对应blockIdx.y
    #那么linear_block_id=blockIdx.x+gridDim.x⋅blockIdx.y
    #所以连续的block id其实是：(0,0),(1,0),(2,0),...,(127,0),(0,1),(1,1),...  也就是 BH 在连续变化，q 固定不变
    #由于我们的调度规则，一般来说在linear_block_id方向上连续的block会有较好的缓存复用：
    #因此在当前这个情况下，连续的block会使用不同BH层的Q，也就是会使用完全不同的K，V，因此会cache miss
    #下面是我们直接用1D的方式去划分，这样子更好直接理解cache hit：
    pid = tl.program_id(0)
    n_q_blocks = tl.cdiv(N, Br)
    pid_BH = pid // n_q_blocks
    pid_q  = pid %  n_q_blocks


    # 之前我们在gemm中用tl.arrange配合广播机制来表示一个2D的指针块，现在我们用这种更现代的方式去处理
    # tl.make_block_ptr(
    #     base: pointer,          # 基础指针，指向张量的起始地址
    #     shape: Tuple[int],      # 张量的整体形状，例如 (M, K)
    #     strides: Tuple[int],    # 每个维度的步长，例如 (stride_m, stride_k)
    #     offsets: Tuple[int],    # 当前块起始位置的偏移量，例如 (pid_m * BLOCK_M, 0)
    #     block_shape: Tuple[int],# 当前要加载/计算的块的大小，例如 (BLOCK_M, BLOCK_K)
    #     order: Tuple[int]       # 内存布局顺序，通常 (1, 0) 表示行优先 (Row-Major)
    # )

    Q_row_start = pid_q * Br
    cur_Q = Q_ptr+pid_BH * (N * d)
    
    cur_Q_cur_block_ptrs = tl.make_block_ptr(
        base = cur_Q,
        shape = (N, d),
        strides = (d, 1),
        offsets=(Q_row_start, 0),
        block_shape=(Br, d),
        order=(1, 0)
    )

    Q_i=tl.load(cur_Q_cur_block_ptrs)
    m_i = tl.full([Br], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([Br], dtype=tl.float32) + 1.0
    o_i = tl.zeros([Br, d], dtype=tl.float32)
    
    cur_K = K_ptr+pid_BH * (N * d)
    cur_V = V_ptr+pid_BH * (N * d)

    #跟论文的伪代码相比，论文里面关注的是串行的怎么去计算出来，而其实kernel里面由于我们只需要关注block维度，因此整个的算法逻辑是有变化的
    #我们是多轮循环每轮是不同的K_j,V_j，然后去算softmax(Q_i@K_j^T)*V_j叠加到O_i上面去

    #注意下面下面对于混合精度的处理：做tl.dot的时候保证input都是fp16类型，用于高效利用tensor core
    #tl.dot默认的output dtype是fp32的，便于直接和后续的数学函数做高精度运算
    for KV_row_start in range(0, N, Bc):
        # 加载当前块的 K 和 V
        cur_K_cur_block_ptrs = tl.make_block_ptr(
            base = cur_K,
            shape = (d, N),
            strides = (1, d),
            offsets=(0, KV_row_start),
            block_shape=(d, Bc),
            order=(0, 1)
        )#注意这里通过shape，strides，offsets，在指针层面直接做到了转置
        cur_V_cur_block_ptrs = tl.make_block_ptr(
            base = cur_V,
            shape = (N, d),
            strides = (d, 1),
            offsets=(KV_row_start, 0),
            block_shape=(Bc, d),
            order=(1, 0)
        )
        K_j_T=tl.load(cur_K_cur_block_ptrs)#我们直接使用了转置后的指针块来加载K_j_T
        V_j=tl.load(cur_V_cur_block_ptrs)

        #计算注意力分数 S = Q * K^T / sqrt(d)
        S_ij = tl.dot(Q_i, K_j_T) #我们保证输入都是fp16类型的，能够利用tensor core高效计算
        S_ij = S_ij * sm_scale

        # online softmax,注意下面这些可以进行数据复用的优化的（测下来大概能提10%左右），这里只是为了写的清晰
        # 其实数据复用的优化也是flash-attention-v2相比flash-attention-v1的一个优化点
        o_i_old = o_i
        m_i_old = m_i
        l_i_old = l_i

        m_i_hat = tl.max(S_ij,axis=1)
        l_i_hat = tl.sum(tl.exp(S_ij - m_i_hat[:, None]), axis=1)#是用[:, None]来保持维度的正确广播
        o_i_hat = tl.dot((tl.exp(S_ij - m_i_hat[:, None]) / l_i_hat[:, None]).to(tl.float16), V_j)#在使用tl.dot之前先把softmax的权重转换成fp16类型，以更好地利用tensor core

        m_i_new = tl.maximum(m_i_old, m_i_hat)#两个元素之间的比较要使用tl.maximum,不用tl.max
        l_i_new = tl.exp(m_i_old - m_i_new) * l_i_old + tl.exp(m_i_hat - m_i_new) * l_i_hat
        o_i_new = tl.exp(m_i_old - m_i_new)[:, None] * o_i_old * l_i_old[:, None] / l_i_new[:, None] + tl.exp(m_i_hat - m_i_new)[:, None] * o_i_hat * l_i_hat[:, None] / l_i_new[:, None]

        #更新给下一轮使用
        m_i = m_i_new
        l_i = l_i_new
        o_i = o_i_new
    
    cur_O = O_ptr+pid_BH * (N * d)
    cur_O_cur_block_ptrs = tl.make_block_ptr(
        base = cur_O,
        shape = (N, d),
        strides = (d, 1),
        offsets=(Q_row_start, 0),
        block_shape=(Br, d),
        order=(1, 0)
    )

    tl.store(cur_O_cur_block_ptrs, o_i.to(tl.float16))

def flash_attention(q, k, v, sm_scale):
    # Shape check
    BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
    
    # 定义 Block Size
    BLOCK_M = 128
    BLOCK_N = 64
    
    # 分配输出
    o = torch.empty_like(q)
    # 临时变量
    L = torch.empty((BATCH, N_HEADS, N_CTX), device=q.device, dtype=torch.float32)
    M = torch.empty((BATCH, N_HEADS, N_CTX), device=q.device, dtype=torch.float32)
    
    # 1D grid: 同一 BH 的 q_block 连续排列 → 更好的 K/V L2 cache 复用
    grid = (BATCH * N_HEADS * triton.cdiv(N_CTX, BLOCK_M),)
    
    # 因为 Kernel 定义中没有接收 stride，它是基于 contiguous 内存假设硬编码计算的
    # 确保输入 tensor 是 contiguous 的
    assert q.is_contiguous(), "Q must be contiguous"
    assert k.is_contiguous(), "K must be contiguous"
    assert v.is_contiguous(), "V must be contiguous"

    _flash_attn_fwd_kernel[grid](
        q, k, v,   # Pointers
        L, M,      # Aux
        o,         # Output
        sm_scale,  # Scalar
        BATCH, N_HEADS, N_CTX, HEAD_DIM, # Logical Dims
        Br=BLOCK_M, Bc=BLOCK_N        
    )
    return o

if __name__ == "__main__":
    torch.manual_seed(0)
    
    # 1. 基础参数设置
    BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
    N_CTX = 4096 
    dtype = torch.float16
    device = "cuda"
    
    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit()

    print(f"Configuration: BATCH={BATCH}, N_HEADS={N_HEADS}, HEAD_DIM={HEAD_DIM}, N_CTX={N_CTX}")
    
    # 确保 contiguous
    q = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device).contiguous()
    k = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device).contiguous()
    v = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device).contiguous()
    
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    
    # 2. 正确性验证
    print("Running consistency check...")
    
    try:
        o_tri = flash_attention(q, k, v, sm_scale)
        # PyTorch SDPA (默认 is_causal=False，与我们的 Kernel 一致)
        o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False)
        
        if torch.allclose(o_tri, o_ref, atol=1e-2, rtol=1e-2):
            print("✅ FlashAttention Verified: Triton matches PyTorch SDPA!")
        else:
            print("❌ FlashAttention Failed!")
            diff = (o_tri - o_ref).abs().max().item()
            print(f"Max Diff: {diff}")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
    # 3. 性能分析
    print("Running performance benchmark...")
    
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N_CTX'],
            x_vals=[1024, 2048, 4096, 8192],
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton', 'PyTorch'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='ms',
            plot_name='flash-attention-bench',
            args={'BATCH': BATCH, 'N_HEADS': N_HEADS, 'HEAD_DIM': HEAD_DIM, 'dtype': dtype}
        )
    )
    def bench_flash_attention(BATCH, N_HEADS, N_CTX, HEAD_DIM, dtype, provider):
        q = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").contiguous()
        k = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").contiguous()
        v = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").contiguous()
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)
        
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'torch':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False), 
                quantiles=quantiles
            )
        if provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: flash_attention(q, k, v, sm_scale), 
                quantiles=quantiles
            )
        return ms, min_ms, max_ms

    bench_flash_attention.run(show_plots=False, print_data=True)