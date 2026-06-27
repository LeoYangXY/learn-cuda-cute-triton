// ============================================================================
// Hopper (SM90) 相比 Ampere (SM80) 新增特性 —— 讲解 + 可运行 Demo
// ============================================================================
//
// 环境: NVIDIA H20 (Hopper 架构, Compute Capability 9.0, 96GB HBM3)
// 编译: nvcc -arch=sm_90a -std=c++17 -O2 -lcuda -Xcompiler -Wno-psabi -o hopper_features 01_hopper_features.cu
//
// 你之前在 RTX 5050 (Blackwell, SM100) 上用过的:
//   ✅ mma.sync (Ampere+)        — 单 Warp(32线程) 矩阵乘
//   ✅ TMA (Hopper+)             — 张量内存加速器
// 你没用过的 (本文重点):
//   ❌ Thread Block Cluster       — 多个 CTA 组成集群协作
//   ❌ WGMMA                      — Warp Group(128线程) 矩阵乘
//   ❌ Distributed Shared Memory  — Cluster 内跨 Block 共享内存
//   ❌ Async Transaction Barrier  — 硬件级异步事务屏障
//   ❌ Warp Specialization        — 生产者/消费者分工模式
//   ❌ setmaxnreg                 — 动态寄存器分配
//   ❌ Sparse WGMMA (2:4)        — 结构化稀疏加速
//   ❌ FP8 Tensor Core            — FP8 精度运算
//   ❌ TMA Multicast              — Cluster 内一次加载广播多 Block
//   ❌ TMA Store                  — SMEM→GMEM 反向 DMA (Ampere 无)
//   ❌ PDL                        — 相邻 kernel 启动延迟重叠
//   ❌ Persistent Kernel+Cluster  — 动态 tile 调度
//
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// 第一部分: 架构对比概览
// ============================================================================
//
// ┌─────────────────────┬──────────────────────┬─────────────────────────────┐
// │ 特性                │ Ampere (SM80)        │ Hopper (SM90)               │
// ├─────────────────────┼──────────────────────┼─────────────────────────────┤
// │ 内存传输            │ cp.async (线程级)    │ TMA (硬件单元, 完全异步)    │
// │ 矩阵运算            │ mma.sync (32线程)    │ WGMMA (128线程, 异步)       │
// │ 协作范围            │ 单 Thread Block      │ Thread Block Cluster (≤16)  │
// │ 共享内存可见性      │ 仅本 Block           │ DSMEM (Cluster内分布式)     │
// │ 同步机制            │ cp.async.wait_group  │ Async Transaction Barrier   │
// │ 线程角色            │ 同质 (bulk-sync)     │ Warp Specialization         │
// │ 寄存器管理          │ 静态 (编译时)        │ setmaxnreg (运行时动态)     │
// │ 稀疏                │ 无原生稀疏MMA        │ Sparse WGMMA (2:4)          │
// │ FP8                 │ 不支持               │ 原生 FP8 (e4m3/e5m2)        │
// └─────────────────────┴──────────────────────┴─────────────────────────────┘
//
// 核心设计哲学变化:
//   Ampere: 所有线程同质, "一起搬数据 → 一起算" (bulk-synchronous)
//   Hopper: 角色分工, "生产者搬数据, 消费者算", 通过 pipeline barrier 解耦
//         → 实现计算与数据传输的**深度流水线重叠**
//
// 其他硬件级改进:
//   - Shared Memory 容量: Ampere 164KB/SM → Hopper 228KB/SM (+39%)
//   - L2 Cache: Ampere 40MB (A100) → Hopper 50MB (H100)
//   - TMA Store: Hopper 新增 SMEM→GMEM 方向 (Ampere cp.async 只有 GMEM→SMEM)
//   - 第四代 NVLink: 带宽翻倍 (900 GB/s)


// ============================================================================
// 第二部分: Thread Block Cluster (跨 Block 协作)
// ============================================================================
//
// 【概念】
// Ampere 上, 每个 Thread Block 是独立调度的最小单位, Block 之间只能通过
// 全局内存通信. Hopper 引入 Cluster: 最多 16 个 Thread Block 绑定在一起
// 被调度到相邻 SM 上, 可以直接访问彼此的共享内存 (无需经过 L2/DRAM).
//
// 【为什么需要】
// FlashAttention-3 中, 一个 Cluster 内的多个 Block 可以:
//   - 共享 KV cache tile (减少重复 load)
//   - 协作做 softmax reduction
//   - 通过 DSMEM 直接传递中间结果
//
// 【launch 方式】
// 使用 cudaLaunchKernelEx 或 cooperative_groups cluster API:
//   dim3 cluster_dims(2, 1, 1);  // 2 个 block 组成 1 个 cluster
//   cudaLaunchConfig_t config = {};
//   config.gridDim = grid;
//   config.blockDim = block;
//   cudaLaunchAttribute attr[1];
//   attr[0].id = cudaLaunchAttributeClusterDimension;
//   attr[0].val.clusterDim = {2, 1, 1};
//   config.attrs = attr;
//   config.numAttrs = 1;
//   cudaLaunchKernelEx(&config, kernel, args...);

__global__ __cluster_dims__(2, 1, 1)  // 编译期指定 cluster 大小 = 2
void cluster_demo_kernel() {
    // 获取 cluster 信息
    cg::cluster_group cluster = cg::this_cluster();
    cg::thread_block block = cg::this_thread_block();

    unsigned int cluster_rank = cluster.block_rank();  // 本 block 在 cluster 中的编号
    unsigned int cluster_size = cluster.num_blocks();  // cluster 中 block 总数

    // 每个 block 有自己的 shared memory
    __shared__ int shared_data[32];

    // 本 block 的线程写入 shared memory
    if (block.thread_rank() < 32) {
        shared_data[block.thread_rank()] = blockIdx.x * 1000 + threadIdx.x;
    }

    // Cluster 级别同步 (等所有 cluster 内的 block 都写完)
    cluster.sync();

    // 现在可以访问 cluster 内其他 block 的 shared memory!
    // 使用 map_shared_rank 获取另一个 block 的共享内存地址
    unsigned int target_rank = (cluster_rank + 1) % cluster_size;
    int* remote_smem = cluster.map_shared_rank(shared_data, target_rank);

    // 读取 remote block 的 shared memory
    int remote_val = 0;
    if (block.thread_rank() == 0) {
        remote_val = remote_smem[0];
        printf("[Cluster] Block %u (cluster_rank=%u) reads from Block rank=%u: value=%d\n",
               blockIdx.x, cluster_rank, target_rank, remote_val);
    }

    cluster.sync();
}

// ============================================================================
// 第三部分: WGMMA (Warp Group Matrix Multiply-Accumulate)
// ============================================================================
//
// 【概念】
// Ampere mma.sync: 单个 Warp(32线程) 协作, 典型 tile m16n8k16 (同步, 阻塞)
// Hopper wgmma:    Warp Group(4 Warp = 128线程) 协作, 最大 tile m64n256k16 (异步)
//
// 【关键区别】
// 1. 规模: 4 倍的线程 → 可以一次算更大的 tile, 减少 instruction 数量
// 2. 异步: wgmma 是异步发射的 (fence → commit → wait), 不阻塞后续指令
//    - mma.sync: 发射后必须等完成才能用结果
//    - wgmma:    发射后可以继续发射更多 wgmma, 最后再 wait
// 3. 直接从 SMEM 读: B 操作数可以直接用 shared memory descriptor, 不需要
//    先 load 到 register (减少寄存器压力)
//
// 【WGMMA 流水线】
//   wgmma_fence()        // 声明要开始 MMA
//   wgmma.mma_async ...  // 发射 (可连续发多条)
//   wgmma_commit_group() // 提交一组
//   wgmma_wait<0>()      // 等待所有完成
//
// 【为什么异步很重要】
// 典型 GEMM K 循环:
//   for k in tiles:
//     tma_load(A_tile, B_tile)     // 加载下一个 tile
//     wgmma(current_tile)          // 计算当前 tile
// 因为 wgmma 是异步的, 加载和计算可以**真正并行** (不是假并行)
//

// WGMMA PTX wrapper (m64n64k16, f32 = f16 × f16 + f32)
// 注: 真实 wgmma 指令有 32-128 个输出寄存器, 这里演示 fence/commit/wait 流程
__device__ __forceinline__ void wgmma_fence_sync() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_all() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

// 简化演示: 展示 WGMMA 的调用骨架 (实际需要正确的 descriptor 和寄存器分配)
__global__ void wgmma_concept_kernel() {
    int tid = threadIdx.x;
    int warp_group_id = tid / 128;  // 每 128 线程是一个 warp group

    if (tid == 0) {
        printf("\n[WGMMA Concept]\n");
        printf("  Thread count: 256 (2 warp groups)\n");
        printf("  Warp Group 0: tid 0-127\n");
        printf("  Warp Group 1: tid 128-255\n");
        printf("  Each warp group can issue independent WGMMA ops\n");
        printf("  WGMMA flow: fence → mma_async → commit → wait\n");
    }

    // 每个 warp group 报告
    if (tid % 128 == 0) {
        printf("  Warp Group %d (tid=%d): ready for WGMMA\n", warp_group_id, tid);
    }
}


// ============================================================================
// 第四部分: Async Transaction Barrier (异步事务屏障)
// ============================================================================
//
// 【概念】
// Ampere 的异步拷贝用 cp.async + cp.async.wait_group 同步:
//   - 粗粒度: 只能等"第 N 组完成", 不知道具体搬了多少字节
//   - 软件管理: 需要手动计数
//
// Hopper 的 mbarrier (异步事务屏障):
//   - 硬件追踪: 设定"期望 X 字节", TMA 搬完后硬件自动递减
//   - 细粒度: 精确到字节级别的完成通知
//   - phase-based: 支持 multi-stage pipeline (类似 double/triple buffering)
//
// 【工作流程】
//   1. mbarrier.init(barrier, expected_arrivals)       // 初始化
//   2. mbarrier.arrive.expect_tx(barrier, num_bytes)   // 告知期望字节数
//   3. TMA load → 硬件自动 arrive(num_bytes)           // TMA 完成时自动通知
//   4. mbarrier.try_wait(barrier, phase)               // consumer 等待完成
//
// 【对比 Ampere】
//   Ampere:  cp.async.commit_group → cp.async.wait_group<N>  (按组等)
//   Hopper:  mbarrier.arrive_expect_tx → try_wait_parity     (按字节追踪)
//   优势: pipeline stage 可以精确同步, 不会"多等"或"少等"

// Demo: mbarrier 基本用法
__global__ void mbarrier_demo_kernel() {
    __shared__ __align__(8) uint64_t barrier;
    __shared__ float data[128];

    int tid = threadIdx.x;

    // 线程 0 初始化 barrier
    if (tid == 0) {
        // mbarrier.init: 期望 blockDim.x 个线程 arrive
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "l"(&barrier), "r"(blockDim.x) : "memory");
    }
    __syncthreads();

    // 每个线程写一些数据, 然后 arrive
    data[tid] = (float)(tid * tid);

    // arrive: 告诉 barrier "我完成了"
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n"
                 :: "l"(&barrier) : "memory");

    // 等待所有线程完成 (phase 0)
    if (tid == 0) {
        int done = 0;
        while (!done) {
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  mbarrier.try_wait.parity.shared.b64 p, [%1], 0;\n"
                "  selp.b32 %0, 1, 0, p;\n"
                "}\n"
                : "=r"(done) : "l"(&barrier) : "memory");
        }
        printf("\n[mbarrier Demo] All %d threads arrived! data[5]=%f\n",
               blockDim.x, data[5]);
    }
}


// ============================================================================
// 第五部分: TMA (Tensor Memory Accelerator) — Hopper 增强版
// ============================================================================
//
// 你在 RTX 5050 上已经用过 TMA, 但 Hopper 上 TMA 有几个重要增强:
//
// 【TMA vs cp.async 对比】
//   ┌─────────────────────┬──────────────────────────┬──────────────────────────┐
//   │                     │ cp.async (Ampere)        │ TMA (Hopper)             │
//   ├─────────────────────┼──────────────────────────┼──────────────────────────┤
//   │ 地址计算            │ 每个线程各自算           │ 硬件自动 (tensor desc)   │
//   │ 发起方式            │ 每个线程发 1 条 load     │ 1 个线程发 1 条搬整个tile│
//   │ 最大粒度            │ 16B / 线程              │ 整个 tile (如 128×64×2B) │
//   │ 多维支持            │ 手动展开                 │ 原生 2D/3D/4D/5D        │
//   │ 边界处理            │ 手动 if-else             │ 硬件自动 padding/clip    │
//   │ 配合同步            │ cp.async.wait_group      │ mbarrier (字节级追踪)    │
//   │ 搬运方向            │ gmem→smem 单向           │ 双向 (load + store)      │
//   │ Multicast           │ 不支持                   │ 支持 (Cluster内广播)     │
//   └─────────────────────┴──────────────────────────┴──────────────────────────┘
//
// 【TMA Multicast (Cluster 内广播)】
// 一次 TMA load 可以把同一个 tile 广播到 Cluster 内所有 Block 的 SMEM
// → KV cache 共享: Q 不同 head 的 Block 共享同一个 K 或 V tile
//
// 【TMA Descriptor (host 端创建)】
// CUtensorMap desc;
// cuTensorMapEncodeTiled(&desc,
//     CU_TENSOR_MAP_DATA_TYPE_FLOAT16,     // 数据类型
//     2,                                    // 维度数
//     ptr,                                  // 全局内存指针
//     {N, M},                               // tensor 形状
//     {N * sizeof(half), sizeof(half)},     // strides
//     {tile_N, tile_M},                     // tile 大小
//     ...);
//
// 【Demo: TMA 数据搬运 (需要 host 端创建 descriptor)】
// 下面用 cuTensorMapEncodeTiled 创建描述符, device 端用 PTX 发起搬运

#include <cuda.h>

// TMA Demo: Host 创建 descriptor + device 端搬运
// 注意: TMA 要求 smem 地址 128B 对齐!
__global__ void tma_load_demo_kernel(const __grid_constant__ CUtensorMap tensor_map) {
    // 用静态 shared memory, 保证对齐
    __shared__ __align__(128) char smem[64 * 64 * sizeof(half)];  // 64x64 fp16 = 8KB
    __shared__ __align__(8) uint64_t mbar;

    int tid = threadIdx.x;

    // 初始化 mbarrier (只需 1 个线程)
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "l"(&mbar), "r"(1) : "memory");
    }
    __syncthreads();

    // 只有 thread 0 发起 TMA load (搬运整个 tile, 其他线程啥都不用做!)
    if (tid == 0) {
        uint64_t* mbar_ptr = &mbar;
        uint32_t smem_addr;
        asm("{\n"
            "  .reg .u64 tmp;\n"
            "  cvta.shared.u64 tmp, %1;\n"
            "  cvt.u32.u64 %0, tmp;\n"
            "}\n" : "=r"(smem_addr) : "l"((void*)smem));

        uint32_t mbar_addr;
        asm("{\n"
            "  .reg .u64 tmp;\n"
            "  cvta.shared.u64 tmp, %1;\n"
            "  cvt.u32.u64 %0, tmp;\n"
            "}\n" : "=r"(mbar_addr) : "l"((void*)mbar_ptr));

        // 设定期望的传输字节数
        int expected_bytes = 64 * 64 * sizeof(half);  // 64x64 tile of fp16 = 8192 bytes
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(expected_bytes) : "memory");

        // TMA 2D load: 一条指令搬 64x64 tile!
        int coord_x = 0;
        int coord_y = 0;
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4}], [%2];\n"
            :: "r"(smem_addr), "l"(&tensor_map), "r"(mbar_addr),
               "r"(coord_x), "r"(coord_y)
            : "memory");
    }

    // 所有线程等待 TMA 完成
    if (tid == 0) {
        int done = 0;
        uint32_t mbar_addr;
        asm("{\n"
            "  .reg .u64 tmp;\n"
            "  cvta.shared.u64 tmp, %1;\n"
            "  cvt.u32.u64 %0, tmp;\n"
            "}\n" : "=r"(mbar_addr) : "l"((void*)&mbar));

        while (!done) {
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  mbarrier.try_wait.parity.shared.b64 p, [%1], 0;\n"
                "  selp.b32 %0, 1, 0, p;\n"
                "}\n"
                : "=r"(done) : "r"(mbar_addr) : "memory");
        }
        // 数据已在 smem 中!
        half* smem_data = (half*)smem;
        printf("\n[TMA Demo] Loaded 64x64 fp16 tile to SMEM via single TMA instruction\n");
        printf("  smem[0] = %f\n", __half2float(smem_data[0]));
        printf("  smem[63] = %f\n", __half2float(smem_data[63]));
    }
}


// ---- TMA Store (Hopper 新增: SMEM → GMEM) ----
// Ampere 只有 cp.async (GMEM→SMEM), 反向需要用普通 store
// Hopper TMA 支持双向, store 也是单指令异步完成
__device__ __forceinline__ void tma_store_2d(
    uint64_t gmem_desc,        // TMA descriptor
    uint32_t smem_addr,        // shared memory 源地址
    int coord_x, int coord_y   // tile 在 tensor 中的坐标
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
        " [%0, {%2, %3}], [%1];\n"
        :: "l"(gmem_desc), "r"(smem_addr),
           "r"(coord_x), "r"(coord_y)
        : "memory"
    );
}

// TMA store commit + wait
__device__ __forceinline__ void tma_store_arrive() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_wait() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}
// 典型用法: 
//   tma_store_2d(desc, smem_addr, x, y);
//   tma_store_arrive();
//   tma_store_wait();
//   fence.proxy.async;  // 确保 store 对其他 unit 可见


// ============================================================================
// 第六部分: setmaxnreg (动态寄存器分配)
// ============================================================================
//
// 【概念】
// Ampere: 每个 warp 的寄存器数在编译时确定, 运行时不变
// Hopper: setmaxnreg 允许在 kernel 执行中**动态调整**寄存器用量:
//   - 数据加载阶段: 减少寄存器 → SM 上能跑更多 warp → 提高带宽利用率
//   - 计算阶段: 增加寄存器 → 放得下大 accumulator → 提高计算效率
//
// 【DeepGEMM 的用法】
//   阶段1 (Producer): setmaxnreg.dec 24    // 只需少量寄存器做 TMA
//   阶段2 (Consumer): setmaxnreg.inc 232   // 需要大量寄存器做 WGMMA
//
// 【PTX 指令】
//   setmaxnreg.inc.sync.aligned.u32 <num>;  // 增加到 num 个寄存器
//   setmaxnreg.dec.sync.aligned.u32 <num>;  // 减少到 num 个寄存器
//   约束: num 必须是 8 的倍数, 范围 [24, 256]

__global__ void setmaxnreg_concept() {
    int tid = threadIdx.x;
    if (tid == 0) {
        printf("\n[setmaxnreg Concept]\n");
        printf("  Phase 1 (load): reduce regs → more active warps → better BW\n");
        printf("  Phase 2 (compute): increase regs → bigger accumulator → better FLOPS\n");
        printf("  PTX: setmaxnreg.inc.sync.aligned.u32 160\n");
        printf("  PTX: setmaxnreg.dec.sync.aligned.u32 40\n");
        printf("  Constraint: must be multiple of 8, range [24, 256]\n");
    }

    // 实际调用 (取消注释需 sm_90a):
    // asm volatile("setmaxnreg.inc.sync.aligned.u32 160;\n");
    // ... do compute ...
    // asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n");
}


// ============================================================================
// 第七部分: Warp Specialization (角色分工)
// ============================================================================
//
// 【概念】
// Ampere 模式: 所有线程做相同的事 → load 完同步 → 一起算 → 同步 → 循环
//   问题: 同步开销大, 计算和加载无法真正重叠
//
// Hopper 模式: Thread Block 内部按 Warp Group 分工:
//   Producer Warp (32线程):  负责发 TMA load, 管理 pipeline barrier
//   Consumer Warp Group (128线程): 负责做 WGMMA 计算
//
// 【Pipeline 流程 (double buffering)】
//   Stage 0:                   Stage 1:
//   ┌──────────────────────┐   ┌──────────────────────┐
//   │ Producer: TMA load   │   │ Producer: TMA load   │
//   │          tile_0      │   │          tile_1      │
//   └──────────────────────┘   └──────────────────────┘
//         │  mbarrier                │  mbarrier
//         ▼                          ▼
//   ┌──────────────────────┐   ┌──────────────────────┐
//   │ Consumer: WGMMA      │   │ Consumer: WGMMA      │
//   │        on tile_0     │   │        on tile_1     │
//   └──────────────────────┘   └──────────────────────┘
//
// Producer 不等 Consumer 算完就开始加载下一个 tile → 真并行!
//
// 【代码骨架】
//   int warp_id = threadIdx.x / 32;
//   if (warp_id < 1) {
//       // Producer: 发 TMA, 管 barrier
//       setmaxnreg_dec<40>();  // 少用寄存器
//       for (int stage : stages) {
//           mbarrier_arrive_expect_tx(bar[stage], tile_bytes);
//           tma_load(smem[stage], desc, coords);
//       }
//   } else {
//       // Consumer: 做 WGMMA
//       setmaxnreg_inc<232>();  // 多用寄存器放 accumulator
//       for (int stage : stages) {
//           mbarrier_wait(bar[stage]);
//           wgmma_fence();
//           wgmma(smem[stage]);
//           wgmma_commit();
//           wgmma_wait<0>();
//       }
//   }

__global__ void warp_specialization_concept() {
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    // 分角色
    bool is_producer = (warp_id == 0);
    bool is_consumer = (warp_id >= 1 && warp_id <= 4);  // warp 1-4 = 1 个 warp group

    if (tid == 0) {
        printf("\n[Warp Specialization]\n");
        printf("  Block has 160 threads (5 warps):\n");
        printf("  Warp 0 (tid 0-31):    Producer — issues TMA loads\n");
        printf("  Warp 1-4 (tid 32-159): Consumer — executes WGMMA\n");
        printf("  Communication: via mbarrier in shared memory\n");
        printf("  Key benefit: load and compute truly overlap!\n");
    }
    __syncthreads();

    if (is_producer && tid == 0) {
        printf("  [Producer] I would issue TMA loads here\n");
    }
    if (is_consumer && tid == 32) {
        printf("  [Consumer] I would execute WGMMA here\n");
    }
}


// ============================================================================
// 第八部分: Distributed Shared Memory (DSMEM)
// ============================================================================
//
// 【概念】
// Cluster 内所有 Block 的 Shared Memory 构成一个分布式共享地址空间.
// 任意一个 Block 可以读/写 Cluster 内其他 Block 的 Shared Memory, 延迟
// 类似本地 SMEM (几十 cycle), 远低于走 L2/DRAM (几百 cycle).
//
// 【使用方式】
//   1. cluster.map_shared_rank(local_ptr, target_rank)  // 获取远程 SMEM 地址
//   2. 直接读写该地址
//   3. 或者通过 TMA multicast 一次广播到多个 Block 的 SMEM
//
// 【Flash Attention 中的应用】
//   - Q 不同 head 的 Block 在一个 Cluster 内
//   - K/V tile 通过 TMA multicast 一次加载到所有 Block 的 SMEM
//   - 避免了每个 Block 重复加载 K/V → 节省 HBM 带宽
//
// 【PTX 指令】
//   ld.shared::cluster  // 从 cluster 内其他 block 的 smem 读
//   st.shared::cluster  // 写到 cluster 内其他 block 的 smem
//   mapa.shared::cluster  // 地址映射

// Demo 已在第二部分 (cluster_demo_kernel) 中展示了 DSMEM 的读取


// ============================================================================
// 第九部分: FP8 Tensor Core
// ============================================================================
//
// 【概念】
// Hopper 原生支持 FP8 数据类型:
//   - E4M3: 4 位指数 + 3 位尾数, 范围小精度高, 适合权重
//   - E5M2: 5 位指数 + 2 位尾数, 范围大精度低, 适合梯度
//
// 【与 FP16/BF16 对比】
//   FP16:  1 sign + 5 exp + 10 mantissa = 16 bit
//   BF16:  1 sign + 8 exp +  7 mantissa = 16 bit
//   E4M3:  1 sign + 4 exp +  3 mantissa =  8 bit  ← Hopper 新增
//   E5M2:  1 sign + 5 exp +  2 mantissa =  8 bit  ← Hopper 新增
//
// 【性能提升】
//   - 相同带宽下传输 2× 的元素 (8bit vs 16bit)
//   - Tensor Core 吞吐翻倍 (相比 FP16)
//   - WGMMA 支持: wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3
//     注意 K 维度从 k16 变成 k32 (因为元素更小)
//
// 【在 H20 上的状况】
//   H20 支持 FP8, 但 Tensor Core 数量比 H100 少很多
//   主要优势在推理场景: 减少显存占用和带宽需求


// ============================================================================
// 第十部分: TMA Multicast (Cluster 内广播加载) — 带 Demo
// ============================================================================
//
// 【概念】
// 普通 TMA load: 一次加载一个 tile 到**一个** Block 的 SMEM
// TMA Multicast: 一次加载一个 tile, 广播到 Cluster 内**多个** Block 的 SMEM
//
// 【为什么重要 — Flash Attention 3 的核心优化】
// FA3 中 Cluster 内多个 Block 处理不同的 Q head, 但共享同一个 KV tile:
//   Block 0 (head 0): 需要 K[0:64, :]
//   Block 1 (head 1): 需要 K[0:64, :]  ← 同样的数据!
// 没有 multicast: 每个 Block 各发一次 TMA load → 2× HBM 带宽
// 有 multicast:   一次 TMA load 广播给两个 Block → 1× HBM 带宽, 省一半!
//
// 【PTX 指令】
// cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
//   [smem_addr], [tensor_desc, {coord_x, coord_y}], [mbar_addr], multicast_mask;
//
// multicast_mask: 16-bit mask, bit i = 1 表示 Cluster 内 rank=i 的 Block 接收数据
//   例: 0x0003 = 0b0000_0000_0000_0011 → rank 0 和 rank 1 都接收
//
// 【使用条件】
// 1. 必须在 Cluster kernel 中使用
// 2. 发起 TMA 的 Block 必须在 multicast_mask 中
// 3. 所有目标 Block 的 mbarrier 都必须 arrive_expect_tx (预计接收的字节数)
// 4. smem 地址对 cluster 内所有 Block 相同 (相同 offset)

// TMA Multicast 2D Load 封装
__device__ __forceinline__ void tma_load_multicast_2d(
    uint32_t smem_addr,        // 本 block 的 smem 目标地址 (cast to u32)
    uint64_t gmem_desc,        // TMA tensor descriptor
    uint32_t mbar_addr,        // mbarrier shared memory 地址
    int coord_x, int coord_y,  // tile 在 tensor 中的坐标
    uint16_t multicast_mask    // 哪些 cluster rank 接收此 tile
) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%3, %4}], [%2], %5;\n"
        :: "r"(smem_addr), "l"(gmem_desc), "r"(mbar_addr),
           "r"(coord_x), "r"(coord_y), "h"(multicast_mask)
        : "memory"
    );
}

// Multicast Demo Kernel: Cluster size=2, 一个 Block 发起 TMA, 两个 Block 都收到数据
__global__ __cluster_dims__(2, 1, 1)
void tma_multicast_demo_kernel(const __grid_constant__ CUtensorMap tensor_map) {
    __shared__ __align__(128) char smem[64 * 64 * sizeof(half)];  // 8KB
    __shared__ __align__(8) uint64_t mbar;

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int cluster_rank = cluster.block_rank();
    int tid = threadIdx.x;

    // 所有 Block 都初始化自己的 mbarrier
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "l"(&mbar), "r"(1) : "memory");
    }
    __syncthreads();

    // 所有 Block 都 arrive_expect_tx (因为 multicast 会写入所有目标 Block 的 smem)
    if (tid == 0) {
        uint32_t mbar_addr;
        asm("{\n"
            "  .reg .u64 tmp;\n"
            "  cvta.shared.u64 tmp, %1;\n"
            "  cvt.u32.u64 %0, tmp;\n"
            "}\n" : "=r"(mbar_addr) : "l"((void*)&mbar));

        int expected_bytes = 64 * 64 * sizeof(half);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(expected_bytes) : "memory");
    }

    // Cluster 同步: 确保所有 Block 的 mbarrier 都已经 arrive_expect_tx
    cluster.sync();

    // 只有 rank=0 的 Block 发起 TMA multicast (一次加载, 广播到所有 rank)
    if (cluster_rank == 0 && tid == 0) {
        uint32_t smem_addr;
        asm("{\n"
            "  .reg .u64 tmp;\n"
            "  cvta.shared.u64 tmp, %1;\n"
            "  cvt.u32.u64 %0, tmp;\n"
            "}\n" : "=r"(smem_addr) : "l"((void*)smem));

        uint32_t mbar_addr;
        asm("{\n"
            "  .reg .u64 tmp;\n"
            "  cvta.shared.u64 tmp, %1;\n"
            "  cvt.u32.u64 %0, tmp;\n"
            "}\n" : "=r"(mbar_addr) : "l"((void*)&mbar));

        // multicast_mask = 0x0003: bit0=1(rank0接收), bit1=1(rank1接收)
        uint16_t mcast_mask = 0x0003;
        int coord_x = 0, coord_y = 0;

        tma_load_multicast_2d(smem_addr, (uint64_t)&tensor_map,
                              mbar_addr, coord_x, coord_y, mcast_mask);
    }

    // 所有 Block 等待自己的 mbarrier 完成 (multicast 会写入每个 Block 的 smem)
    if (tid == 0) {
        uint32_t mbar_addr;
        asm("{\n"
            "  .reg .u64 tmp;\n"
            "  cvta.shared.u64 tmp, %1;\n"
            "  cvt.u32.u64 %0, tmp;\n"
            "}\n" : "=r"(mbar_addr) : "l"((void*)&mbar));

        int done = 0;
        while (!done) {
            asm volatile(
                "{\n"
                "  .reg .pred p;\n"
                "  mbarrier.try_wait.parity.shared.b64 p, [%1], 0;\n"
                "  selp.b32 %0, 1, 0, p;\n"
                "}\n"
                : "=r"(done) : "r"(mbar_addr) : "memory");
        }

        // 验证: 两个 Block 都应该有数据!
        half* data = (half*)smem;
        printf("[TMA Multicast] Block %u (cluster_rank=%u): "
               "smem[0]=%.4f, smem[63]=%.4f  ← same data in both blocks!\n",
               blockIdx.x, cluster_rank,
               __half2float(data[0]), __half2float(data[63]));
    }
}


// ============================================================================
// 第十一部分: Programmatic Dependent Launch (PDL)
// ============================================================================
//
// 【概念】
// 传统 CUDA stream 中, kernel B 依赖 kernel A 的输出:
//   A 完全执行完毕 → GPU idle gap → B 开始
//
// PDL 允许:
//   A 执行到某个点后 → B 立即启动 (在 A 还没完成时!)
//   B 先做不依赖 A 的前置工作 (如清零 buffer, 加载常量)
//   B 需要 A 结果时 → cudaGridDependencySynchronize() 等待
//
// 【时间线对比】
//   传统: [====== kernel A ======]  [gap]  [====== kernel B ======]
//   PDL:  [====== kernel A ======]
//                        [== B preamble ==][==== B main work ====]
//                        ↑                 ↑
//              A 调用 trigger       B 调用 GridDependencySync
//
// 【API】
//   主内核: cudaTriggerProgrammaticLaunchCompletion()  // 通知可以启动次级内核
//   次内核: cudaGridDependencySynchronize()             // 等待主内核完成
//   Host:   cudaLaunchAttributeProgrammaticStreamSerialization  // 启动属性
//
// 【典型应用场景】
//   - GEMM epilogue → 下一个 GEMM prologue 重叠
//   - Softmax → 下一层 Attention 的 K/V 预加载
//   - 任何 pipeline 中相邻 kernel 之间的延迟隐藏

// PDL Demo: Primary kernel 写数据, Secondary kernel 读数据
__global__ void pdl_primary_kernel(float* output, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: 做一些计算
    if (tid < N) {
        output[tid] = (float)(tid * tid) * 0.001f;
    }

    // 通知 GPU: 本 block 的主要工作做完了
    // 当 grid 中所有 block 都 trigger 之后, GPU 才会真正启动 secondary kernel
    __syncthreads();
    cudaTriggerProgrammaticLaunchCompletion();

    // Phase 2: 可以继续做不影响 secondary 的收尾工作
    if (tid < N) {
        output[tid] += 1.0f;  // secondary 会看到这个 +1 (因为它要 sync)
    }
}

__global__ void pdl_secondary_kernel(float* input, float* result, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 1: 不依赖 primary 结果的前置工作 (与 primary 并行!)
    // 例如: 清零输出缓冲区, 预计算常量等
    if (tid < N) {
        result[tid] = 0.0f;  // 先清零 (不依赖 primary)
    }
    __syncthreads();

    // Phase 2: 需要 primary 结果 → 等待!
    cudaGridDependencySynchronize();

    // 现在 primary 已经完全完成, input[] 数据可用
    if (tid < N) {
        result[tid] = input[tid] * 2.0f;
    }

    if (tid == 0) {
        printf("[PDL Demo] secondary kernel: input[0]=%f (should be 1.0), result[0]=%f\n",
               input[0], result[0]);
    }
}


// ============================================================================
// 第十二部分: Persistent Kernel + 动态 Tile 调度
// ============================================================================
//
// 【概念】
// 传统 GEMM: grid = (M/BM) × (N/BN) 个 block, 每个 block 算一个 output tile
//   问题: 如果 grid 很大, 大量 block 排队等待 SM → 启动开销大
//         如果 grid 不整除 SM 数 → 最后一波 (wave) SM 利用率低 (wave quantization)
//
// Persistent Kernel: 只启动刚好占满 GPU 的 block 数, 每个 block 循环处理多个 tile
//   优点: 
//     - 减少 block 启动开销
//     - 可以跨 tile 复用 SMEM 中的数据 (如 GEMM 的 weight prefetch)
//     - 更好的 L2 cache 利用 (通过控制 tile 遍历顺序)
//
// 【Hopper 上的增强: Cluster + Persistent】
// 在 Hopper 上, Persistent Kernel 配合 Cluster:
//   - Cluster 内的 Block 协作处理一组 tile
//   - TMA multicast 在 Cluster 内共享数据
//   - 通过全局原子计数器动态获取下一个要处理的 tile
//
// 【Tile Scheduler 模式 (CUTLASS 3.x 中的实现)】
//   1. Raster Order: 按固定顺序遍历 tile (L 形, Z 形)
//   2. Stream-K: 将 K 维度也分割, 实现更均匀的负载均衡
//   3. Persistent + Stream-K: 结合两者优势
//
// 【注意: Cluster Launch Control (Blackwell SM100 特有)】
//   Blackwell 引入了硬件级 "Work Stealing" 机制:
//   - 一个 Block 可以尝试取消 (cancel) 另一个尚未启动的 Block
//   - 成功取消后 "窃取" 该 Block 的工作
//   - 这是 Blackwell 独有的, Hopper 上用全局原子计数器实现类似效果

// Persistent Kernel Demo: 全局原子计数器动态分发 tile
__device__ uint32_t g_tile_counter = 0;  // 全局 tile 计数器

__global__ void persistent_kernel_demo(float* output, int total_tiles,
                                        int tiles_per_row, int BM, int BN) {
    int block_id = blockIdx.x;
    int tid = threadIdx.x;

    // 每个 block 循环获取 tile, 直到所有 tile 处理完
    while (true) {
        // 原子获取下一个要处理的 tile index
        __shared__ uint32_t tile_idx;
        if (tid == 0) {
            tile_idx = atomicAdd(&g_tile_counter, 1);
        }
        __syncthreads();

        if (tile_idx >= (uint32_t)total_tiles) break;  // 所有 tile 都处理完了

        // 计算这个 tile 的坐标
        int tile_row = tile_idx / tiles_per_row;
        int tile_col = tile_idx % tiles_per_row;

        // 模拟: 每个线程处理 tile 中的一个元素
        int global_row = tile_row * BM + (tid / BN);
        int global_col = tile_col * BN + (tid % BN);
        int global_idx = global_row * (tiles_per_row * BN) + global_col;

        if (tid < BM * BN && global_idx < total_tiles * BM * BN) {
            output[global_idx] = (float)tile_idx;  // 标记由哪个 tile 写入
        }

        __syncthreads();  // 确保 tile_idx 不会被下一轮覆盖
    }

    if (tid == 0 && block_id == 0) {
        printf("[Persistent Kernel] Block 0 processed multiple tiles dynamically\n");
        printf("  Total tiles: %d, Blocks launched: just %d (persistent)\n",
               total_tiles, gridDim.x);
    }
}


// ============================================================================
// 第十三部分: 完整可运行 Demo
// ============================================================================

// --- Demo 1: Cluster 通信 ---
void run_cluster_demo() {
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Demo 1: Thread Block Cluster + DSMEM\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");

    // Launch with cluster size = 2
    // 使用 cudaLaunchKernelEx 的方式
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(4, 1, 1);   // 4 个 block
    config.blockDim = dim3(32, 1, 1); // 每个 block 32 线程
    config.dynamicSmemBytes = 0;

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = 2;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;

    cudaError_t err = cudaLaunchKernelEx(&config, cluster_demo_kernel);
    if (err != cudaSuccess) {
        printf("  Cluster launch error: %s\n", cudaGetErrorString(err));
        printf("  (Try: __cluster_dims__ attribute approach instead)\n");
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

// --- Demo 2: mbarrier 同步 ---
void run_mbarrier_demo() {
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Demo 2: Async Transaction Barrier (mbarrier)\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");

    mbarrier_demo_kernel<<<1, 32>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// --- Demo 3: TMA 数据搬运 ---
void run_tma_demo() {
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Demo 3: TMA (Tensor Memory Accelerator) Load\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");

    // 创建一个 64x64 的 fp16 tensor
    int rows = 64, cols = 64;
    size_t size = rows * cols * sizeof(half);
    half* h_data = (half*)malloc(size);
    for (int i = 0; i < rows * cols; i++) {
        h_data[i] = __float2half((float)i * 0.01f);
    }

    half* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // 创建 TMA descriptor
    CUtensorMap tensor_map;
    uint64_t global_dims[2] = {(uint64_t)cols, (uint64_t)rows};
    uint64_t global_strides[1] = {(uint64_t)(cols * sizeof(half))};  // 只需 rank-1 个 stride
    uint32_t box_dims[2] = {64, 64};  // tile 大小
    uint32_t elem_strides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2,                              // rank
        (void*)d_data,                  // 全局内存指针
        global_dims,                    // 全局维度
        global_strides,                 // strides (bytes)
        box_dims,                       // tile 维度
        elem_strides,                   // element strides
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        printf("  cuTensorMapEncodeTiled failed: %d\n", (int)res);
        printf("  (This requires CUDA 12.0+ driver)\n");
    } else {
        tma_load_demo_kernel<<<1, 32>>>(tensor_map);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

// --- Demo 4: WGMMA & Warp Specialization 概念展示 ---
void run_wgmma_demo() {
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Demo 4: WGMMA + Warp Specialization Concept\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");

    wgmma_concept_kernel<<<1, 256>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    warp_specialization_concept<<<1, 160>>>();
    CHECK_CUDA(cudaDeviceSynchronize());

    setmaxnreg_concept<<<1, 32>>>();
    CHECK_CUDA(cudaDeviceSynchronize());
}

// --- Demo 5: TMA Multicast ---
void run_tma_multicast_demo() {
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Demo 5: TMA Multicast (Cluster Broadcast Load)\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");

    // 创建 64x64 fp16 tensor
    int rows = 64, cols = 64;
    size_t size = rows * cols * sizeof(half);
    half* h_data = (half*)malloc(size);
    for (int i = 0; i < rows * cols; i++) {
        h_data[i] = __float2half((float)i * 0.01f);
    }

    half* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // 创建 TMA descriptor
    CUtensorMap tensor_map;
    uint64_t global_dims[2] = {(uint64_t)cols, (uint64_t)rows};
    uint64_t global_strides[1] = {(uint64_t)(cols * sizeof(half))};
    uint32_t box_dims[2] = {64, 64};
    uint32_t elem_strides[2] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(
        &tensor_map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        2, (void*)d_data,
        global_dims, global_strides, box_dims, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        printf("  cuTensorMapEncodeTiled failed: %d\n", (int)res);
    } else {
        // Launch with cluster size = 2
        cudaLaunchConfig_t config = {};
        config.gridDim = dim3(2, 1, 1);   // 2 blocks = 1 cluster
        config.blockDim = dim3(32, 1, 1);
        config.dynamicSmemBytes = 0;

        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = 2;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
        config.attrs = &attr;
        config.numAttrs = 1;

        cudaError_t err = cudaLaunchKernelEx(&config, tma_multicast_demo_kernel, tensor_map);
        if (err != cudaSuccess) {
            printf("  Multicast launch error: %s\n", cudaGetErrorString(err));
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
}

// --- Demo 6: PDL (Programmatic Dependent Launch) ---
void run_pdl_demo() {
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Demo 6: Programmatic Dependent Launch (PDL)\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");

    int N = 256;
    float *d_buf, *d_result;
    CHECK_CUDA(cudaMalloc(&d_buf, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_buf, 0, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_result, 0, N * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Launch primary kernel (普通方式)
    pdl_primary_kernel<<<1, 256, 0, stream>>>(d_buf, N);

    // Launch secondary kernel (带 PDL 属性)
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(1, 1, 1);
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    config.attrs = &attr;
    config.numAttrs = 1;

    cudaError_t err = cudaLaunchKernelEx(&config, pdl_secondary_kernel, d_buf, d_result, N);
    if (err != cudaSuccess) {
        printf("  PDL launch error: %s\n", cudaGetErrorString(err));
        printf("  (PDL requires compute capability 9.0+)\n");
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFree(d_result));
}

// --- Demo 7: Persistent Kernel ---
void run_persistent_kernel_demo() {
    printf("\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");
    printf("Demo 7: Persistent Kernel + Dynamic Tile Scheduling\n");
    printf("=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=" "=\n");

    int BM = 4, BN = 8;  // tile 大小 (简化)
    int M = 16, N = 32;  // 矩阵大小
    int tiles_per_row = N / BN;     // 4
    int tiles_per_col = M / BM;     // 4
    int total_tiles = tiles_per_row * tiles_per_col;  // 16 tiles

    float* d_output;
    size_t size = M * N * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_output, size));
    CHECK_CUDA(cudaMemset(d_output, 0, size));

    // 重置全局计数器
    uint32_t zero = 0;
    CHECK_CUDA(cudaMemcpyToSymbol(g_tile_counter, &zero, sizeof(uint32_t)));

    // 只启动 4 个 block (远少于 16 tiles), 每个 block 处理多个 tile
    int num_blocks = 4;
    persistent_kernel_demo<<<num_blocks, BM * BN>>>(d_output, total_tiles,
                                                     tiles_per_row, BM, BN);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 验证
    float* h_output = (float*)malloc(size);
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("  Output sample: tile assignments for first row:\n  ");
    for (int i = 0; i < N; i++) {
        printf("%.0f ", h_output[i]);
    }
    printf("\n  (Each number = tile index that wrote this element)\n");

    CHECK_CUDA(cudaFree(d_output));
    free(h_output);
}

// --- Device info ---
void print_device_info() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("============================================================\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SM Count: %d\n", prop.multiProcessorCount);
    printf("Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("Max Shared Memory/SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Cluster Launch Support: %s\n",
           prop.clusterLaunch ? "YES" : "NO");
    printf("============================================================\n\n");
}


// ============================================================================
// main
// ============================================================================
int main() {
    // 初始化 CUDA driver API (需要 cuTensorMapEncodeTiled)
    CHECK_CUDA(cudaFree(0));  // lazy init

    print_device_info();

    // 运行 demos
    run_cluster_demo();
    run_mbarrier_demo();
    run_tma_demo();
    run_wgmma_demo();
    run_tma_multicast_demo();
    run_pdl_demo();
    run_persistent_kernel_demo();

    printf("\n");
    printf("============================================================\n");
    printf("Summary: Hopper (SM90) New Features vs Ampere (SM80)\n");
    printf("============================================================\n");
    printf(" 1. Thread Block Cluster — multi-CTA cooperation (up to 16)\n");
    printf(" 2. WGMMA — 128-thread async matrix multiply\n");
    printf(" 3. DSMEM — distributed shared memory across cluster\n");
    printf(" 4. Async Transaction Barrier — hardware byte-level tracking\n");
    printf(" 5. TMA enhanced — multicast, 5D, hardware boundary handling\n");
    printf(" 6. TMA Multicast — one load broadcasts to all cluster CTAs\n");
    printf(" 7. Warp Specialization — producer/consumer decomposition\n");
    printf(" 8. setmaxnreg — dynamic register allocation at runtime\n");
    printf(" 9. FP8 Tensor Core — e4m3/e5m2 native support\n");
    printf("10. Sparse WGMMA — 2:4 structured sparsity, 2x throughput\n");
    printf("11. PDL — overlap adjacent kernel launch latency\n");
    printf("12. Persistent Kernel — dynamic tile scheduling\n");
    printf("============================================================\n");

    return 0;
}
