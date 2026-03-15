import torch
import triton
import triton.language as tl

# ==============================================================================
# FlashAttention-1 Kernel Logic
# 1 个 Thread Block 负责 1 个 Head 的整个序列 (N x d)。
# 它在内部串行遍历所有的 Q 块 (i)，对每个 Q 块再遍历所有的 K/V 块 (j)。
# ==============================================================================

@triton.jit
def _flash_attn_fwd_kernel_fa1(
    # --- 数据指针 ---
    Q_ptr, K_ptr, V_ptr,
    L_ptr, M_ptr,      # 辅助状态 (FA1 通常也需要存，虽然这里我们主要在寄存器算)
    O_ptr,             # 输出
    
    # --- 标量参数 ---
    sm_scale,          # softmax scale
    
    # --- 逻辑维度 ---
    B: tl.constexpr,   
    H: tl.constexpr,   
    N: tl.constexpr,   
    d: tl.constexpr,   
    
    # --- 分块大小 ---
    Br: tl.constexpr,  # BLOCK_M
    Bc: tl.constexpr,  # BLOCK_N
):
    # 1. FA1 并行策略：一个block处理一个bh的一整个Q
    pid = tl.program_id(0)
    pid_BH = pid 
    # 注意：这里没有 pid_q！因为一个 Block 要处理所有的 q 块
    
    # 计算当前 Head 的基地址
    # 假设输入是 (B, H, N, d) 且 contiguous
    stride_bh = N * d
    cur_Q_base = Q_ptr + pid_BH * stride_bh
    cur_K_base = K_ptr + pid_BH * stride_bh
    cur_V_base = V_ptr + pid_BH * stride_bh
    cur_O_base = O_ptr + pid_BH * stride_bh

    # ==========================================================================
    # FA1 核心变化：外层循环遍历 Q 的行块 (Row Blocks)
    # 在 FA2 中，这个循环被“展开”到了 Grid 维度，由不同的 Block 并行执行。
    # 在 FA1 中，同一个 Block 串行执行这个循环。
    # ==========================================================================
    for i_start in range(0, N, Br):
        Q_row_start = i_start
        
        # 2. 加载当前的 Q 块 (Q_i)
        cur_Q_ptrs = tl.make_block_ptr(
            base=cur_Q_base,
            shape=(N, d),
            strides=(d, 1),
            offsets=(Q_row_start, 0),
            block_shape=(Br, d),
            order=(1, 0)
        )
        Q_i = tl.load(cur_Q_ptrs)
        
        # 3. 【关键】初始化当前 Q 块的 Online Softmax 状态
        # 因为每一行 (Token) 的 Attention 是独立的，
        # 当我们要计算新的 Q 块 (不同的行) 时，必须重置 m, l, o。
        m_i = tl.full([Br], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([Br], dtype=tl.float32) + 1.0
        o_i = tl.zeros([Br, d], dtype=tl.float32)
        
        # 4. 内层循环：遍历所有的 K/V 块 (Column Blocks)
        # 这部分逻辑与 FA2 完全一致
        for j_start in range(0, N, Bc):
            # 加载 K_j, V_j
            cur_K_ptrs = tl.make_block_ptr(
                base=cur_K_base,
                shape=(d, N),
                strides=(1, d),
                offsets=(0, j_start),
                block_shape=(d, Bc),
                order=(0, 1)
            )
            cur_V_ptrs = tl.make_block_ptr(
                base=cur_V_base,
                shape=(N, d),
                strides=(d, 1),
                offsets=(j_start, 0),
                block_shape=(Bc, d),
                order=(1, 0)
            )
            
            K_j_T = tl.load(cur_K_ptrs)
            V_j = tl.load(cur_V_ptrs)
            
            # 计算 S = Q * K^T * scale
            S_ij = tl.dot(Q_i, K_j_T)
            S_ij = S_ij * sm_scale
            
            # --- Online Softmax 更新逻辑 (FA1 & FA2 通用) ---
            
            # 3.1 计算当前块的 row-max
            m_ij_hat = tl.max(S_ij, axis=1)
            
            # 3.2 更新全局 max
            m_i_new = tl.maximum(m_i, m_ij_hat)
            
            # 3.3 计算修正系数 alpha
            alpha = tl.exp(m_i - m_i_new)
            
            # 3.4 计算 P_tilde = exp(S - m_new)
            P_ij = tl.exp(S_ij - m_i_new[:, None])
            
            # 3.5 更新 sum_exp
            l_ij = tl.sum(P_ij, axis=1)
            l_i_new = alpha * l_i + l_ij
            
            # 3.6 更新 output (未归一化)
            # 注意：P_ij 转 fp16 以利用 Tensor Core
            o_i = (alpha[:, None] * o_i) + tl.dot(P_ij.to(tl.float16), V_j)
            
            # 3.7 更新状态
            m_i = m_i_new
            l_i = l_i_new
            
        # 5. 【FA1 特有】当前 Q 块计算完毕，立即归一化并写回
        # FA2 是每个 Block 最后写一次；FA1 是每处理完一个 i 块就写一次
        o_i_final = o_i / l_i[:, None]
        
        cur_O_ptrs = tl.make_block_ptr(
            base=cur_O_base,
            shape=(N, d),
            strides=(d, 1),
            offsets=(Q_row_start, 0),
            block_shape=(Br, d),
            order=(1, 0)
        )
        tl.store(cur_O_ptrs, o_i_final.to(tl.float16))

# ==============================================================================
# Python 封装函数
# ==============================================================================

def flash_attention_fa1(q, k, v, sm_scale):
    BATCH, N_HEADS, N_CTX, HEAD_DIM = q.shape
    
    # 定义 Block Size (FA1 通常 BLOCK_M 可以大一点，因为寄存器压力在循环内复用)
    BLOCK_M = 128
    BLOCK_N = 64
    
    # 分配输出
    o = torch.empty_like(q)
    # 辅助变量 (FA1 其实不需要在 global 存中间态，但为了接口一致保留)
    L = torch.empty((BATCH, N_HEADS, N_CTX), device=q.device, dtype=torch.float32)
    M = torch.empty((BATCH, N_HEADS, N_CTX), device=q.device, dtype=torch.float32)
    
    # === 关键区别：Grid 大小 ===
    # FA1: 只有 Batch * Heads 个 Block
    grid = (BATCH * N_HEADS,)
    
    assert q.is_contiguous(), "Q must be contiguous"
    assert k.is_contiguous(), "K must be contiguous"
    assert v.is_contiguous(), "V must be contiguous"

    _flash_attn_fwd_kernel_fa1[grid](
        q, k, v,
        L, M,
        o,
        sm_scale,
        BATCH, N_HEADS, N_CTX, HEAD_DIM,
        Br=BLOCK_M, Bc=BLOCK_N
    )
    return o

# ==============================================================================
# 主程序：测试与 Benchmark
# ==============================================================================

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
    print("Mode: FlashAttention-1 Simulation (Serial over Sequence)")
    
    # 生成数据
    q = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device).contiguous()
    k = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device).contiguous()
    v = torch.randn((BATCH, N_HEADS, N_CTX, HEAD_DIM), dtype=dtype, device=device).contiguous()
    
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    
    # 2. 正确性验证
    print("\nRunning consistency check...")
    
    try:
        # 运行 FA1 实现
        o_tri = flash_attention_fa1(q, k, v, sm_scale)
        
        # PyTorch 参考实现
        # 注意：我们的 Kernel 没有加 causal mask，所以这里 is_causal=False
        o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale, is_causal=False)
        
        if torch.allclose(o_tri, o_ref, atol=1e-2, rtol=1e-2):
            print("✅ FlashAttention-1 Verified: Triton matches PyTorch SDPA!")
        else:
            print("❌ FlashAttention-1 Failed!")
            diff = (o_tri - o_ref).abs().max().item()
            print(f"Max Diff: {diff}")
            # 打印一些样本以便调试
            # print("Triton:", o_tri[0, 0, :5, :5])
            # print("PyTorch:", o_ref[0, 0, :5, :5])
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
    # 3. 性能分析
    print("\nRunning performance benchmark...")
    print("(Comparing FA1-style Triton vs PyTorch SDPA)")
    
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N_CTX'],
            x_vals=[1024, 2048, 4096, 8192],
            line_arg='provider',
            line_vals=['triton_fa1', 'torch'],
            line_names=['Triton (FA1 Style)', 'PyTorch SDPA'],
            styles=[('red', '--'), ('green', '-')], # FA1 用红色虚线表示可能较慢
            ylabel='ms',
            plot_name='flash-attention-fa1-bench',
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
        elif provider == 'triton_fa1':
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: flash_attention_fa1(q, k, v, sm_scale), 
                quantiles=quantiles
            )
        return ms, min_ms, max_ms

    bench_flash_attention.run(show_plots=False, print_data=True)