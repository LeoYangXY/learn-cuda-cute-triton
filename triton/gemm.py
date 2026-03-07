import torch
import triton
import triton.language as tl


DEVICE = torch.device("cuda:0")


# 每个 block 负责计算 C 的一个子块，大小为 BLOCK_SIZE_M × BLOCK_SIZE_N。
# 为了计算这个子块，它需要遍历 K 维度，每次取 A 的一个 BLOCK_SIZE_M × BLOCK_SIZE_K 的块，
# 和 B 的一个 BLOCK_SIZE_K × BLOCK_SIZE_N 的块，进行乘法累加。
# C 的一个块 [M_blk, N_blk] = Σ (A 的块 [M_blk, K_blk] × B 的块 [K_blk, N_blk])
#
# 整体要完成如下任务，只是每个block各自分配一些，然后大家并行的去做:
# for m in range(0, M, BLOCK_SIZE_M):
#     for n in range(0, N, BLOCK_SIZE_N):
#         acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#         for k in range(0, K, BLOCK_SIZE_K):
#             a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#             b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#             acc += dot(a, b)
#         C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc

# 而且注意到现代 GPU 的强大算力主要来自 Tensor Core，它一次能计算一个小矩阵块的乘法。我们下面将调用tensor core去实现
# tl.dot() 会自动使用 Tensor Core（WMMA/MMA 指令），但需满足以下条件：
#
# ✅ 硬件要求：
#   - GPU 架构 >= Volta (compute capability >= 7.0)
#
# ✅ 数据类型要求：
#   - 输入 a 和 b 必须为 Tensor Core 支持的低精度类型，如：
#       • tl.float16（fp16）
#       • tl.bfloat16（bf16，Ampere 及以上）
#       • tl.int8（整数 GEMM，特定架构）
#   - 注意：若使用 tl.float32，将无法触发 Tensor Core，回退到 CUDA Core 计算，
#     性能显著下降（通常慢 5~10 倍）。
#
# ✅ 形状对齐要求：
#   - 矩阵乘维度（M, N, K）在当前程序块（program instance）中必须满足对齐约束。
#   - 典型要求（以 Ampere 为例）：
#       • M（a 的行数） % 16 == 0
#       • N（b 的列数） % 16 == 0（或 8，取决于布局）
#       • K（a 的列数 = b 的行数） % 16 == 0
#   - 实践建议：将 BLOCK_M, BLOCK_N, BLOCK_K 设为 16 的倍数（如 64, 128, 256）。
#
# ✅ 编译器行为：
#   - Triton 编译器在满足上述条件时，会自动生成 WMMA（Volta/Turing）或
#     native MMA（Ampere+）指令。
#   - 若任一条件不满足，则回退到标量/向量化 CUDA Core 实现。
#
# 示例：
#   if a.dtype == tl.float16 and b.dtype == tl.float16:
#       # 且 BLOCK_M, BLOCK_N, BLOCK_K 均为 16 的倍数
#       c = tl.dot(a, b)  # ✅ 触发 Tensor Core 加速
#
# ⚠️ 调试提示：
#   可通过 `triton.compile(...).asm['ptx']` 查看是否包含 mma.sync 指令，
#   以确认 Tensor Core 是否被实际使用。


@triton.jit
def matmul_kernel_baseline(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A 的步长
    stride_bk, stride_bn,  # B 的步长
    stride_cm, stride_cn,  # C 的步长
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    
    # 对于矩阵 A，形状是 (M, K)，它的两个 stride 是：
    # stride_am：沿着 M 维（行）走一步，地址增加多少？
    # stride_ak：沿着 K 维（列）走一步，地址增加多少？
    # 如果 A 是 行主序（row-major）且连续（如 PyTorch 默认）：
    # stride_am = K（下一行要跳 K 个元素）
    # stride_ak = 1（下一列就是下一个元素）
    # 如果 A 是 列主序的，那么 stride_am = 1, stride_ak = M
    

    #C的大小是M*N，每个block负责计算C的一个BM*BN的子块，因此可以考虑一个简单的任务映射关系是：
    #block0负责C[0:BM, 0:BN]，block1负责C[0:BM, BN:2*BN]这样的行主序去排

    pid = tl.program_id(0)
    block_per_row = (N + BN - 1) // BN  # 每行有多少个block
    row_start = (pid // block_per_row) * BM  # 当前block负责的行起始位置
    col_start = (pid % block_per_row) * BN  # 当前block负责的列起始位置

    # 写出数学公式，注意这种分块的写法,用":"去表示分块
    # C[row_start:row_start+BM,col_start:col_start+BN]=A[row_start:row_start+BM,:]@B[:,col_start:col_start+BN]
    # 然后由于我们要在K维度上拆分，因此就是在K维度上做一个循环：
    # C[row_start:row_start+BM,col_start:col_start+BN]=Σ(A[row_start:row_start+BM,k:k+BK]@B[k:k+BK,col_start:col_start+BN])
    # 所以我们想要构造出2维的指针块，分别指向A[row_start:row_start+BM,k:k+BK]和B[k:k+BK,col_start:col_start+BN],然后每次使用tl.dot()去计算这个块的乘积，并累加到C[row_start:row_start+BM,col_start:col_start+BN]上

    #==========================================================================
    # 现在，我们要加载：
    # 行号：从 row_start 到 row_start + BM - 1 → 共 BM 个行号
    # 列号：从 k_start 到 k_start + BK - 1 → 共 BK 个列号
    # 那么总共要加载 BM × BK 个元素，也就是要有一个2D的tensor去存这BM*BK个地址，才能去tl.load

    # 比如对于A的那个指针块，我们要怎么告诉 GPU “我要加载 A 的这 BM×BK 个数”？
    # 注意GPU 内存是一长串地址（像一维数组），所以我们必须算出每个元素的地址。
    # 如果A 存成行优先（row-major）：第 i 行第 j 列的地址 = a_ptr + i * stride_am + j * stride_ak，其中 stride_am = K, stride_ak = 1

    # 如果用普通 Python，你会这么去得到这个指针块，也就是2D的tensor地址表呢：
    # a_ptrs = [[0 for _ in range(BK)] for _ in range(BM)]
    # for i in range(BM):
    #     for j in range(BK):
    #         row = row_start + i
    #         col = k_start + j
    #         a_ptrs[i][j] = a_ptr + row * stride_am + col * stride_ak
    # 这就是一个 BM × BK 的二维地址表！
    # 但 Triton 不能写 for 循环（会慢），所以它用 向量化 + 广播 来一次性算出整个表。

    # Triton 用三步完成上面的双重循环：
    # 第 1 步：生成行号列表和列号列表
    # python 里你会写成：
    # rows = [row_start, row_start+1, ..., row_start+BM-1]   # 长度 BM
    # cols = [k_start, k_start+1, ..., k_start+BK-1]         # 长度 BK
    # 在 Triton 里写成：
    # rows = row_start + tl.arange(0, BM)   # → rows
    # cols = k_start + tl.arange(0, BK)     # → cols（注意：k_start 在循环中变化）
    # 第 2 步：借助unsqueeze机制:把行号变成“列向量”，列号变成“行向量”:
    # rows[:, None] → 把 [r0, r1, r2] 变成：
    # [[r0],
    #  [r1],
    #  [r2]]
    # cols[None, :] → 把 [c0, c1] 变成：
    # [[c0, c1]]
    # 其实就是一个unsqueeze的语法糖：在哪个维度加上None，相当于就是在这个维度上做了unsqueeze，新增了1
    # 第 3 步：利用广播机制，自动算出整个地址表：
    # 广播机制介绍：对于如下的(3,1)和(1,2)的tensor进行相加，会自动广播成(3,2)的tensor：
    # [[r0]       +  [[c0, c1]   =   [[r0+c0, r0+c1],
    # [r1]                           [r1+c0, r1+c1],
    # [r2]]                          [r2+c0, r2+c1]]

    # 因此，如果我们想要计算出一个2D的地址表，第(i,j)号元素是 a_ptr + rows[i] * stride_am + cols[j] * stride_ak，那么我们就可以先算出 rows 和 cols，然后利用广播机制自动算出整个表：
    # a_ptrs = a_ptr + rows[:, None] * stride_am + cols[None, :] * stride_ak

    accumulator = tl.zeros((BM,BN), dtype=tl.float32)
    for k_start in tl.range(0, K, BK):
        a_rows = row_start + tl.arange(0, BM)  # BM
        a_cols = k_start + tl.arange(0, BK)    # BK
        a_ptrs = a_ptr + a_rows[:, None] * stride_am + a_cols[None, :] * stride_ak  # BM×BK的指针块

        # 我们想做的是有一个2D的bool tensor，标量语义是：a_mask[i][j] = (a_rows[i] < M) and (a_cols[j] < K)
        # 然后用向量+广播的形式去写就是:a_mask = (a_rows[:, None] < M) & (a_cols[None, :] < K)，得到一个BM*BK的mask
        # a_rows[:, None] < K  → [[True],
        #                        [False]]  
        # a_cols[None, :] < N  → [[True, True, False]]  
        # a_mask = 
        # [[True & True,   True & True,   True & False]   → [True,  True,  False]
        # [False & True,  False & True,  False & False]] → [False, False, False]
        a_mask = (a_rows[:, None] < M) & (a_cols[None, :] < K) #在load数据的时候使用mask，防止越界

        b_rows = k_start + tl.arange(0, BK)    # BK
        b_cols = col_start + tl.arange(0, BN)  # BN
        b_ptrs = b_ptr + b_rows[:, None] * stride_bk + b_cols[None, :] * stride_bn  # BK×BN的指针块
        b_mask = (b_rows[:, None] < K) & (b_cols[None, :] < N)

        #使用指针块去global memory中进行load,得到真正的数值块
        a_tensor = tl.load(a_ptrs, mask=a_mask, other=0.0)  # BM×BK
        b_tensor = tl.load(b_ptrs, mask=b_mask, other=0.0)  # BK×BN

        # 这里会自动调用tensor core去计算
        accumulator += tl.dot(a_tensor, b_tensor)
    
    c_rows = row_start + tl.arange(0, BM)
    c_cols = col_start + tl.arange(0, BN)
    c_ptrs = c_ptr + c_rows[:, None] * stride_cm + c_cols[None, :] * stride_cn

    c_mask = (c_rows[:, None] < M) & (c_cols[None, :] < N)  # 在store数据的时候使用mask，防止越界

    tl.store(c_ptrs, accumulator, mask=c_mask)


# 通过launch的顺序影响L2 cache的行为，从而进行优化

# ### 🧠 为什么 Launch 顺序会影响 L2 Cache 性能？—— 并行 ≠ 同时执行！
# 很多人误以为：“GPU 有上万个 CUDA cores，所有 blocks 是同时运行的”，  
# 但实际上，**GPU 的并行是严格受限的**，blocks 是 **分批、分时调度执行** 的。
# #### 🔢 硬件资源限制（以 A100 为例）：
# - **SM 数量**：108 个（流多处理器，实际计算单元）
# - **每个 SM 最多并发**：约 32–64 个 warps（即 1024–2048 个线程）
# - **典型 block 配置**：128–256 线程/block → 每个 SM 同时只能跑 4–16 个 blocks
# 因此，即使你 launch 了 1000 个 blocks：
# - GPU **先将前 ~800 个 blocks 分配到所有 SM 上**（填满硬件资源）
# - 剩下的 blocks **进入调度队列，排队等待**
# - 当某个 block 执行完毕，**调度器才从队列头部取出下一个 block**
# > ✅ 关键事实：NVIDIA GPU 使用 **FIFO 风格的全局 block 调度器**，  
# > 按 `blockIdx`（即 Triton 中的 `program_id`）**递增顺序分配 blocks**。  
# > 虽然不保证严格串行，但 **低 ID blocks 总体上比高 ID blocks 更早执行** —— 这称为 **"launch order locality"**。


# #### 📦 缓存行为示例：默认行优先 vs Grouped Launch

# ⚠️ 在分析 L2 Cache 行为（尤其是 evict）时，
# 通常可以近似假设 blocks 按 pid 顺序“串行”或“时间局部性执行”，这是一种合理且广泛使用的分析模型。

# 假设我们计算 C = A @ B，B 为 row-major，每个 block 负责 C 的一个 (BM, BN) 子块。

# ##### ❌ 默认行优先调度（低效）：
# 使用 1D grid，block ID 是 pid = 0, 1, 2, ...
# 映射规则：pid → (row_block, col_block) = (pid // num_col_blocks, pid % num_col_blocks)
# 示例（每个block负责C中的一个 BM x BN 的块）：
#   pid=0 → (0,0),  pid=1 → (0,1),  ...,  pid=7 → (0,7)
#   pid=8 → (1,0), pid=9 → (1,1),  ...,  pid=15 → (1,7)
#   ...
# 计算C的(0,0)块需要使用B[:,0:BN]，计算C的(0,1)块需要使用B[:,BN:2*BN]，...,计算C的(1,0)块又需要使用B[:,0:BN]，可见：
# - 因此，需要相同 B 列（如 B[:,0:BN]）的 blocks 是 pid=0, 8, 16, 24, ...
# - 它们在 launch 顺序上相隔很远（中间夹着 pid=1~7, 9~15 等）
# - 这些中间 blocks 加载了 B[:,BN:2*BN], B[:,2*BN:3*BN], ...
# - L2 Cache 被填满 → B[:,0:BN] 被 evict
# - 当 pid=8 执行时，B[:,0:BN] 已不在 L2 → 必须重新从 DRAM 加载 ❌

# ##### ✅ Grouped Launch（高效）：
# 通过重映射：让连续的 pid 尽可能需要使用相同的 B 的列
# 示例（GROUP_SIZE_M=4）：左边是pid号，右边是它所负责的C的块的逻辑编号
#   pid=0 → (0,0)
#   pid=1 → (1,0)
#   pid=2 → (2,0)
#   pid=3 → (3,0)
#   pid=4 → (0,1)
#   pid=5 → (1,1)
#   pid=6 → (2,1)
#   pid=7 → (3,1)
#   pid=8 → (4,0)
#   ...
# - 现在，需要 B[:,0:BN] 的 blocks 是 pid=0,1,2,3,8,9,10,11,...
# - 它们在 launch 顺序上**高度连续**
# - GPU 调度器会优先执行低 pid blocks → 这些 blocks 在时间上接近执行
# - B[:,0:BN] 只需加载一次，后续 blocks 从 L2 复用 ✅
# - 显著减少 DRAM 访问，提升带宽效率

# #### 🎯 核心洞见：
# - **你无法直接控制 L2 Cache**，但可以通过 **数据访问模式 + block 调度顺序** 间接优化
# - **Grouped launch 是一种“调度 trick”**：它不改变算法，只改变 blocks 的执行时间局部性
# - 正因 GPU 调度器具有 **launch order locality**，这种优化在实践中**稳定有效**

# ##### 🤔 为什么 Grouped Launch 不影响 A 的缓存效率？
# 在当前 kernel 实现中，A 的数据访问具有以下特性：
# 每个 block 负责 C 的一个块，比如C[row_start:row_start+BM,col_start:col_start+BN]，
# 根据矩阵乘法定义：
# C[i, j] = sum_k (A[i, k] * B[k, j])
# 所以，要计算 C 的第 i 行，只需要 A 的第 i 行。
# 因此：
# 某个block 如果负责 C 的第 m 行块 → 只需要 A 的第 m 行块
# 另一个block 如果负责 C 的第 m' 行块（m' ≠ m）→ 只需要 A 的第 m' 行块
# ✅ A 的每一行只被 唯一一个 block 使用！
#
# 因此：
# - A **没有跨-block 复用价值** —— 即使两个 block 连续执行，它们也用不到彼此的 A 数据
# - 所以，无论采用 baseline 还是 grouped launch，
#        A 的 L2 Cache 行为几乎相同：每个 A tile 被加载一次，用完就 evict
#
# 相比之下，B 的列数据被多个 blocks 重复使用（所有负责计算的元素在C中属于同一列的那些blocks 都需要 B[:, n*BN:(n+1)*BN]），
# 所以 **只有 B 能从 grouped launch 的时间局部性中获益**。

@triton.jit
def matmul_kernel_grouped(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr = 8,
):
    
    #其实就是相比baseline版本，改了一个pid到C中的块的映射关系。我们也是先直观的写出来映射，然后再写公式去分配
   
    #根据上面的分析，我们需要：把“需要相同 B 列”的 blocks 排成连续的 pid！！！
    #那么，哪些 C 的块需要相同的 B 列？
    #
    # 回忆 GEMM: C = A @ B（B 是 row-major）
    # - 计算 C 的任意元素 C[i, j] 都需要 B[:, j]
    # - 因此，**C 的同一列中的所有块（不管在哪一行）都需要相同的 B 列**
    #
    # 举例：
    #   C[0:BM,    0:BN] → 需要 B[:, 0:BN]
    #   C[BM:2*BM, 0:BN] → 也需要 B[:, 0:BN]
    #   C[2*BM:3*BM, 0:BN] → 也一样！
    #
    # 所以，要复用 B[:, 0:BN]，我们必须让C中以下逻辑块对应的gpu的 blocks 在其 pid 上连续：
    #   (m=0, n=0), (m=1, n=0), (m=2, n=0), ..., (m=*, n=0)
    #
    # 同理，下一组连续 pid 应该是：
    #   (m=0, n=1), (m=1, n=1), (m=2, n=1), ..., (m=*, n=1)  → 复用 B[:, BN:2*BN]
    #
    # 🎯 结论：我们按 **C 的列（n）分组**，每组内按 **C 的行（m）排列**
    #         即：先排完第 0 列的所有行块，再排第 1 列的所有行块，……


    # ============================================================
    # 🧩 任务划分对比：Baseline vs Naive Column-Major vs Grouped
    # ============================================================

    # 假设参数：
    #   M = 512, N = 512
    #   BM = BN = 128
    #   → num_pid_m = 4, num_pid_n = 4
    #   → 总共 16 个 blocks (pid=0~15)

    # ------------------------------------------------------------
    # 1. Baseline（行主序）：默认调度方式
    #    pid 按 C 的行优先分配
    # ------------------------------------------------------------
    # n=0         n=1         n=2         n=3
    # ┌─────────┬─────────┬─────────┬─────────┐
    # │ (0,0)   │ (0,1)   │ (0,2)   │ (0,3)   │ ← m=0
    # │ → pid=0 │ → pid=1 │ → pid=2 │ → pid=3 │
    # ├─────────┼─────────┼─────────┼─────────┤
    # │ (1,0)   │ (1,1)   │ (1,2)   │ (1,3)   │ ← m=1
    # │ → pid=4 │ → pid=5 │ → pid=6 │ → pid=7 │
    # ├─────────┼─────────┼─────────┼─────────┤
    # │ (2,0)   │ (2,1)   │ (2,2)   │ (2,3)   │ ← m=2
    # │ → pid=8 │ → pid=9 │ → pid=10│ → pid=11│
    # ├─────────┼─────────┼─────────┼─────────┤
    # │ (3,0)   │ (3,1)   │ (3,2)   │ (3,3)   │ ← m=3
    # │ → pid=12│ → pid=13│ → pid=14│ → pid=15│
    # └─────────┴─────────┴─────────┴─────────┘
    #
    # ❌ 问题：需要 B[:,0:BN] 的 blocks 是 pid=0,4,8,12 —— 不连续！
    #        L2 中 B[:,0:BN] 被中间 blocks 的 B[:,BN:2*BN] 等挤出 → 多次 DRAM 加载

    # ------------------------------------------------------------
    # 2. Naive Column-Major（错误的“整列连续”方式）：
    #    让所有 m 行连续处理同一列 n（即 pid 按列主序分配）
    # ------------------------------------------------------------
    # n=0         n=1         n=2         n=3
    # ┌─────────┬─────────┬─────────┬─────────┐
    # │ (0,0)   │ (0,1)   │ (0,2)   │ (0,3)   │ ← m=0
    # │ → pid=0 │ → pid=4 │ → pid=8 │ → pid=12│
    # ├─────────┼─────────┼─────────┼─────────┤
    # │ (1,0)   │ (1,1)   │ (1,2)   │ (1,3)   │ ← m=1
    # │ → pid=1 │ → pid=5 │ → pid=9 │ → pid=13│
    # ├─────────┼─────────┼─────────┼─────────┤
    # │ (2,0)   │ (2,1)   │ (2,2)   │ (2,3)   │ ← m=2
    # │ → pid=2 │ → pid=6 │ → pid=10│ → pid=14│
    # ├─────────┼─────────┼─────────┼─────────┤
    # │ (3,0)   │ (3,1)   │ (3,2)   │ (3,3)   │ ← m=3
    # │ → pid=3 │ → pid=7 │ → pid=11│ → pid=15│
    # └─────────┴─────────┴─────────┴─────────┘
    #
    # ✅ 表面看很好：n=0 列由 pid=0,1,2,3 连续执行 → B[:,0:BN] 可复用！
    #
    # ⚠️ 但问题出现在大矩阵场景：
    #    若 M=8192 → num_pid_m = 64，N=512 → num_pid_n=4
    #    则前 64 个 pid（0~63）全部用于 n=0 列！
    #    它们会加载 A[0:BM], A[BM:2*BM], ..., A[63*BM:64*BM]
    #    → A 的 64 个不同行块塞满 L2（A100 L2=40MB，很容易溢出）
    #    → B[:,0:BN] 被 A 的数据 evict → 后续 pid 仍需从 DRAM 重载 B！
    #    → L2 被 A “污染”，B 的复用失效！

    # ------------------------------------------------------------
    # 3. 正确方案：Grouped Launch（带 GROUP_SIZE_M）
    #    限制每组最多 GROUP_SIZE_M 行，避免 L2 被 A 污染
    #    设 GROUP_SIZE_M = 2 → 每组包含 2 行 × 所有列
    # ------------------------------------------------------------
    # GROUP #0（m=0~1）:
    # n=0         n=1         n=2         n=3
    # ┌─────────┬─────────┬─────────┬─────────┐
    # │ (0,0)   │ (0,1)   │ (0,2)   │ (0,3)   │ ← m=0
    # │ → pid=0 │ → pid=2 │ → pid=4 │ → pid=6 │
    # ├─────────┼─────────┼─────────┼─────────┤
    # │ (1,0)   │ (1,1)   │ (1,2)   │ (1,3)   │ ← m=1
    # │ → pid=1 │ → pid=3 │ → pid=5 │ → pid=7 │
    # └─────────┴─────────┴─────────┴─────────┘
    #
    # GROUP #1（m=2~3）:
    # n=0         n=1         n=2         n=3
    # ┌─────────┬─────────┬─────────┬─────────┐
    # │ (2,0)   │ (2,1)   │ (2,2)   │ (2,3)   │ ← m=2
    # │ → pid=8 │ → pid=10│ → pid=12│ → pid=14│
    # ├─────────┼─────────┼─────────┼─────────┤
    # │ (3,0)   │ (3,1)   │ (3,2)   │ (3,3)   │ ← m=3
    # │ → pid=9 │ → pid=11│ → pid=13│ → pid=15│
    # └─────────┴─────────┴─────────┴─────────┘
    #
    # ✅ 优势：
    #   - 每组内：同一列 n 的 blocks 连续执行（如 pid=0,1 → n=0）→ B 复用 ✅
    #   - 每组只涉及 GROUP_SIZE_M=2 行 A → A 数据总量小，不会挤掉 B ✅
    #   - 组间切换时 L2 自然刷新，无跨组干扰 ✅
    #
    # 📌 GROUP_SIZE_M 的本质：
    #    控制“同时活跃的 A 行数”，在 B 复用 和 A/L2 冲突 之间取得平衡。
    #    通常取 4~16（Triton 默认 8），兼顾缓存效率与并行度。

    # ⚠️ 在分析 L2 Cache 行为（尤其是 evict）时，
    # 通常可以近似假设 blocks 按 pid 顺序“串行”或“时间局部性执行”，这是一种合理且广泛使用的分析模型。


    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)



    # 一个 Group 覆盖 GROUP_SIZE_M 行，横跨整个 N 维度
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # 当前 pid 属于第几个 Group
    group_id = pid // num_pid_in_group
    # 当前 Group 的起始行 ID
    first_pid_m = group_id * GROUP_SIZE_M
    # 处理边缘情况：最后一个 Group 可能行数不足 GROUP_SIZE_M,这是去计算当前 Group 实际的行数
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    # 计算出当前 pid 在 Group 内的局部编号（0 到 num_pid_in_group-1）
    pid_in_group = pid % num_pid_in_group

    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    row_start = pid_m * BM
    col_start = pid_n * BN

    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for k_start in tl.range(0, K, BK):
        a_rows = row_start + tl.arange(0, BM)
        a_cols = k_start + tl.arange(0, BK)
        a_ptrs = a_ptr + a_rows[:, None] * stride_am + a_cols[None, :] * stride_ak
        a_mask = (a_rows[:, None] < M) & (a_cols[None, :] < K)
        a_tensor = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_rows = k_start + tl.arange(0, BK)
        b_cols = col_start + tl.arange(0, BN)
        b_ptrs = b_ptr + b_rows[:, None] * stride_bk + b_cols[None, :] * stride_bn
        b_mask = (b_rows[:, None] < K) & (b_cols[None, :] < N)
        b_tensor = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a_tensor, b_tensor)

    c_rows = row_start + tl.arange(0, BM)
    c_cols = col_start + tl.arange(0, BN)
    c_ptrs = c_ptr + c_rows[:, None] * stride_cm + c_cols[None, :] * stride_cn
    c_mask = (c_rows[:, None] < M) & (c_cols[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


# ============================================================
# Wrapper functions
# ============================================================
def matmul_baseline(a, b):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta['BM']) * triton.cdiv(N, meta['BN']),)
    matmul_kernel_baseline[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=128, BN=128, BK=32,
    )
    return c


def matmul_grouped(a, b):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta['BM']) * triton.cdiv(N, meta['BN']),)
    matmul_kernel_grouped[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM=128, BN=128, BK=32,
        GROUP_SIZE_M=8,
    )
    return c


def matmul_torch(a, b):
    return torch.matmul(a, b)


# ============================================================
# Benchmark
# ============================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[256 * i for i in range(1, 17)],  # 256 to 4096
        x_log=True,
        line_arg='provider',
        line_vals=['baseline', 'grouped', 'torch'],
        line_names=['Triton Baseline', 'Triton Grouped', 'PyTorch'],
        styles=[('blue', '-'), ('red', '-'), ('green', '-')],
        ylabel='TFLOPS',
        plot_name='matmul-comparison',
        args={},
    )
)
def benchmark(M, provider):
    K = M
    N = M
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    if provider == 'baseline':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_baseline(a, b), quantiles=quantiles)
    elif provider == 'grouped':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_grouped(a, b), quantiles=quantiles)
    elif provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul_torch(a, b), quantiles=quantiles)

    tflops = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return tflops(ms), tflops(max_ms), tflops(min_ms)


# ============================================================
# Main: run benchmark
# ============================================================
if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True)