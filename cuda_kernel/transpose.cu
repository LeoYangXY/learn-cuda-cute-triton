#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>


// 假设 M = 1000, N = 800，你选择 blockSize = (16, 16)：
// 每个 block 覆盖 16 行 × 16 列 = 256 个元素。
// 需要的 grid size：
// x 方向（列）：⌈800 / 16⌉ = 50
// y 方向（行）：⌈1000 / 16⌉ = 63
// 总共 50 × 63 = 3150 个 blocks，总共 3150 × 256 = 806,400 个线程。
// 其中只有 1000 × 800 = 800,000 个线程做实际工作，其余 6400 个线程因越界而跳过。

//每个thread负责把原始矩阵的(row，col)处的元素复制到转置矩阵的(col，row)处
__global__ void naive_transpose(float *dst, const float *src, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {//通过此，把越界的线程跳过
    dst[col * M + row] = src[row * N + col];
  }
}

//每个thread负责4个元素,把src的(row,col),(row,col+1),(row,col+2),(row,col+3)复制到dst的(col,row),(col+1,row),(col+2,row),(col+3,row)
__global__ void transpose_4float(float *dst, const float *src, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = (blockIdx.y * blockDim.y + threadIdx.y)*4;
    if(row < M && col < N) {//通过此，把越界的线程跳过
        float4 tmp = *reinterpret_cast<const float4*>(src + row * N + col);
        dst[col * M + row] = tmp.x;
        dst[(col + 1) * M + row] = tmp.y;
        dst[(col + 2) * M + row] = tmp.z;
        dst[(col + 3) * M + row] = tmp.w;
    }
}



//合并访问（Coalesced Access）：
// 当一个 warp（32 个线程） 同时访问全局内存时，如果它们访问的地址 落在同一个或连续的内存事务（memory transaction）中，就叫“合并访问”。
// 反之，如果地址分散、跳跃，就需要多次内存事务 → 非合并（uncoalesced） → 性能暴跌。

// ✅ 什么是“读合并”？（Coalesced Read）
// 🎯 理想情况（完美合并）：
// warp 中线程 tid = 0,1,2,...,31
// 访问地址：ptr + 0, ptr + 1, ptr + 2, ..., ptr + 31（每个元素 4 字节）
// 实际地址：A, A+4, A+8, ..., A+124 → 连续 128 字节
// ✅ 1 次 128B 事务搞定！带宽利用率 100%

// ❌ 非合并读（灾难）：
// 线程访问：ptr + 0, ptr + 1024, ptr + 2048, ...
// 地址间隔很大（比如跨行）
// ❌ 需要 32 次独立事务 → 带宽利用率 ≈ 1/32！


// ✅ 什么是“写合并”？（Coalesced Write）
// 逻辑和读一样！
// 🎯 理想写合并：
// 线程 0 写 dst[0]
// 线程 1 写 dst[1]
// ...
// 线程 31 写 dst[31]
// 地址连续 → ✅ 1 次 128B 写事务


// ❌ 非合并写（矩阵转置的痛点）：
// 在转置中，如果是每个 thread 负责 float4 的场景：
// 每个 thread 负责写 dst 的 (col,row), (col+1,row), (col+2,row), (col+3,row)
// 如果 src 为 1024×2048 的矩阵（M=1024, N=2048），则 dst 为 2048×1024
// 
// - 每个 thread 的写地址为（单位：float 索引）：
//     dst[col     * M + row]   → col*1024 + row
//     dst[(col+1) * M + row]   → (col+1)*1024 + row
//     dst[(col+2) * M + row]   → (col+2)*1024 + row
//     dst[(col+3) * M + row]   → (col+3)*1024 + row
// - 考虑一个 warp（32 个线程），通常 threadIdx.y 连续 → 每个thread各自对应的col = 0,4,8,...,124
//   整个 warp 写的起始列分别为 0,4,8,...,124，对应写地址（以 float 为单位）：
//     T0: 0*1024+row, 1*1024+row, 2*1024+row, 3*1024+row
//     T1: 4*1024+row, 5*1024+row, 6*1024+row, 7*1024+row
//     ...
//     T31: 124*1024+row, ..., 127*1024+row
//   然后我们要分析整个warp的访存行为：这个warp有128个地址要访存，但是我们发现任意2个地址无法处在同一个内存事务中
// → 触发多达 128 次独立写事务。
// 结果：写带宽利用率极低（常 < 10%），成为性能瓶颈。


//而shared memory 是片上高速缓存，访问无合并要求！我们可以随机的去访问，这给了我们加速的思路
// 每个 block 处理一个 TILE_H × TILE_W 的子块,且通常 TILE_H == TILE_W（例如 32×32）
// 然后我们一般设置TileSize为warpSize的倍数，这样子block中的一行不会跨warp
#define TILE_SIZE 32
__global__ void mat_transpose_shared_kernel(
    float* __restrict__ dst,          // 输出矩阵，尺寸 N × M
    const float* __restrict__ src,    // 输入矩阵，尺寸 M × N
    int M,                            // src 的行数
    int N                             // src 的列数（= dst 的行数）
) {
    // === 1. 线程索引 ===
    const int local_col = threadIdx.x;   // block 内 x 坐标（处理列方向）
    const int local_row = threadIdx.y;   // block 内 y 坐标（处理行方向）
    
    const int block_row = blockIdx.y * TILE_SIZE;  // 当前 block 负责的起始行（在 src 中）
    const int block_col = blockIdx.x * TILE_SIZE;  // 当前 block 负责的起始列（在 src 中）

    const int global_row = block_row + local_row;    // 当前线程在 src 中的全局行
    const int global_col = block_col + local_col;  // 当前线程在 src 中的全局列

    // === 2. Shared memory tile: 存储 src 的 [block_row, block_row+TILE_SIZE) × [block_col, block_col+TILE_SIZE)
    // tile 是 src 的一个局部、未转置的拷贝

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    // === 3. 协作加载：从 global → shared memory（合并读）===
    if (global_row < M && global_col < N) {
        tile[local_row][local_col] = src[global_row * N + global_col];
    } 

    // === 4. 同步：确保整个 tile 加载完成 ===
    __syncthreads();

    // === 5. 重组写出：从 shared → global（目标：合并写！）===
    // 我们希望一个warp的元素能够写入dst的连续位置，这样才能合并写

    // 对于block中的一个thread，其一维的编号是：threadId_in_block = threadIdx.y * blockDim.x + threadIdx.x，这就是如何划分warp的
    // 所以：x 增加 1 → 线程 ID 增加 1；y 增加 1 → 线程 ID 跳过一整行（跳 blockDim.x）
    // 因此，一个warp中的所有thread，由于我们之前的设置（TileSize为warpSize的倍数），
    // 因此这32个thread会有相同的threadIdx.y,也就是相同的local_row，然后他们的local_col会在一段连续的 32 个整数中（0～31、32～63 等）
    
    // 由于我们希望连续写入dst，也就是说要访问的dst的地址需要是连续的，也就是一个warp去访问dst的同一行的元素位置
    // 注意如果我们让global_row_new = global_col;global_col_new = global_row;的话，那么便是：
    // global_row_new=blockIdx.x * TILE_SIZE+local_row:对于一个warp的32个thread是一样的
    // global_col_new=blockIdx.y * TILE_SIZE+local_col：对于一个warp的32个thread刚好是根据local_col的值是连续的32个整数
    // 所以我们每个thread负责写入这个地址是可以做到内存合并访问的


    //// 注意：转置要求 src[i][j] → dst[j][i]
    // 其中：
    //   i = global_row = block_row + local_row
    //   j = global_col = block_col + local_col
    //
    // 因此，src[block_row + local_row][block_col + local_col] 应该写入
    // dst[block_col + local_col][block_row + local_row]
    //
    // 但是，如果我们让当前线程直接写这个位置，
    // 会导致写入地址为 (block_col + local_col) * M + (block_row + local_row)，
    // 而 warp 中 local_col 连续变化 → dst 行号连续变化 → 地址跳跃（非合并写）。
    //
    // 关键洞察：关键洞察：谁来写 dst[j][i]？
    // 你可以让任何线程写 dst[j][i]，只要它能访问到 src[i][j]。
    // 在 shared memory 模型中，整个 block 共享 tile由于整个 block 共享 tile，我们可以重新分配写任务：
    // 让线程 (local_row, local_col) 不写它自己加载的元素，
    // 而是把 tile[local_col][local_row] 这个的元素写到其在dst中对应的位置。
    //
    // 具体地：
    //   tile[local_col][local_row] = src[block_row + local_col][block_col + local_row]
    //   根据转置规则，它应写入：
    //        dst_row = block_col + local_row
    //        dst_col = block_row + local_col
    //
    // 此时，warp 中 local_row 固定、local_col 连续 →
    //   dst_row 固定，dst_col 连续 → 写入地址连续 → 合并写达成！

    //直观理解：
    // 如果简单化考虑：一个block是32*32的话，
    // 那么warp5在填写阶段，从src的某一行的连续位置读取元素，负责把这个block的shared memory的第5行填满，
    // 然后在写入阶段，这个warp要把这个shared memory的第5列元素写出去，经过转置也就是会写到dst的某一行的连续位置
    // 这样子读取和写入都做到了合并访问
    // 在具体实现的时候，就用我们是调度线程的，因此就是用local_row,local_col去表述即可

    const int global_row_new = block_col + local_row;   // = blockIdx.x * TILE_SIZE + local_row
    const int global_col_new = block_row + local_col;   // = blockIdx.y * TILE_SIZE + local_col

    if(global_row_new < N && global_col_new < M){
        dst[global_row_new * M + global_col_new] = tile[local_col][local_row];
    }

}




// bank conflict:

// Shared memory 被划分为 32 个 bank，每个 bank 负责一段连续的 4 字节（32-bit）地址空间，
// 按 4B 一块、4B 一块地轮流分配给 bank 0 到 bank 31。
// 每个 bank 每周期只能服务一次 32-bit（4 字节）的访问

// [0～3B] → bank 0  
// [4～7B] → bank 1  
// [8～11B] → bank 2  
// ...  
// [124～127B] → bank 31  
// [128～131B] → bank 0 （循环回来）  
// [132～135B] → bank 1  
// ...

// 因此，计算bank_id的数学公式为：
// byte_addr = 元素的字节偏移
// word_index = byte_addr / 4          // 整数除法，即“第几个 4B 单元”
// bank_id = word_index % 32

// 所以，在我们的tile中存储的元素为fp32的场景下：
// tile[i][j] 的 字节偏移 = (i * 32 + j) * 4
// word_index = byte_offset / 4 = i * 32 + j
// bank_id = word_index % 32 = (i * 32 + j) % 32 = j % 32
// 也就是说bank_id由j决定

// 上面的例子中有dst[global_row_new * M + global_col_new] = tile[local_col][local_row]
// 一个warp中的32个thread的local-col是连续的32个数字，但是local_row是相同的，因此这32个thread会访问同一个bank，所以这 32 次访问必须串行执行，耗时 ≈ 32 倍

//一个经典的处理方法就是padding：
//将tile的列数增加到33，这样每个warp访问的bank就不会冲突,原因在于：
// tile[i][j] → byte_offset = (i * 33 + j) * 4   // 因为每行有 33 个元素！
// → word_index = i * 33 + j
// → bank_id = (i * 33 + j) % 32 = (i + j) % 32
// 因此，当最后访问tile[local_col][local_row]时，local_col从0~31连续变化，local_row是相同的，因此bank_id也是从0~31连续变化，无bank_conflict

__global__ void mat_transpose_shared_kernel_padding(
    float* __restrict__ dst,          
    const float* __restrict__ src,   
    int M,                           
    int N                            
) {
    const int local_col = threadIdx.x;   
    const int local_row = threadIdx.y;  
    
    const int block_row = blockIdx.y * TILE_SIZE;  
    const int block_col = blockIdx.x * TILE_SIZE;  

    const int global_row = block_row + local_row;    
    const int global_col = block_col + local_col;  

    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];//此处padding，别的代码都不用动

    if (global_row < M && global_col < N) {
        tile[local_row][local_col] = src[global_row * N + global_col];
    } 

    __syncthreads();

    const int global_row_new = block_col + local_row;   
    const int global_col_new = block_row + local_col;   

    if(global_row_new < N && global_col_new < M){
        dst[global_row_new * M + global_col_new] = tile[local_col][local_row];
    }

}



// CPU reference transpose
void cpu_transpose(const std::vector<float>& src, std::vector<float>& dst, int M, int N) {
    dst.resize(N * M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            dst[j * M + i] = src[i * N + j];
        }
    }
}

// Check correctness
bool check_correctness(const std::vector<float>& gpu_result, const std::vector<float>& cpu_ref, int size) {
    const float eps = 1e-5f;
    for (int i = 0; i < size; ++i) {
        if (std::abs(gpu_result[i] - cpu_ref[i]) > eps) {
            printf("Mismatch at %d: GPU=%f, CPU=%f\n", i, gpu_result[i], cpu_ref[i]);
            return false;
        }
    }
    return true;
}

// Timing helper
float time_kernel(cudaEvent_t start, cudaEvent_t stop, dim3 grid, dim3 block, 
                  void (*kernel)(float*, const float*, int, int),
                  float* d_dst, const float* d_src, int M, int N) {
    cudaEventRecord(start);
    kernel<<<grid, block>>>(d_dst, d_src, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}
int main() {
    // Matrix dimensions
    const int M = 10240;
    const int N = 20480;

    printf("Testing matrix transpose: M=%d, N=%d\n", M, N);

    // Host data
    std::vector<float> h_src(M * N);
    std::srand(42);
    for (int i = 0; i < M * N; ++i) {
        h_src[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    std::vector<float> h_dst_cpu;
    cpu_transpose(h_src, h_dst_cpu, M, N);

    // Device memory
    float *d_src = nullptr, *d_dst = nullptr;
    cudaMalloc(&d_src, M * N * sizeof(float));
    cudaMalloc(&d_dst, N * M * sizeof(float));
    cudaMemcpy(d_src, h_src.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    bool all_passed = true;

    // -----------------------------
    // 1. naive_transpose
    // -----------------------------
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((M + blockSize.x - 1) / blockSize.x,
                      (N + blockSize.y - 1) / blockSize.y);
        float ms = time_kernel(start, stop, gridSize, blockSize, naive_transpose, d_dst, d_src, M, N);
        std::vector<float> h_result(N * M);
        cudaMemcpy(h_result.data(), d_dst, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = check_correctness(h_result, h_dst_cpu, N * M);
        printf("[naive_transpose] Time: %.3f ms | Correct: %s\n", ms, ok ? "YES" : "NO");
        if (!ok) all_passed = false;
    }

    // -----------------------------
    // 2. transpose_4float
    // -----------------------------
    {
        dim3 blockSize(16, 16);
        dim3 gridSize((M + blockSize.x - 1) / blockSize.x,
                      (N + 4 * blockSize.y - 1) / (4 * blockSize.y));
        float ms = time_kernel(start, stop, gridSize, blockSize, transpose_4float, d_dst, d_src, M, N);
        std::vector<float> h_result(N * M);
        cudaMemcpy(h_result.data(), d_dst, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = check_correctness(h_result, h_dst_cpu, N * M);
        printf("[transpose_4float] Time: %.3f ms | Correct: %s\n", ms, ok ? "YES" : "NO");
        if (!ok) all_passed = false;
    }

    // -----------------------------
    // 3. shared memory (no padding)
    // -----------------------------
    {
        const int TILE = 32;
        dim3 blockSize(TILE, TILE);
        dim3 gridSize((N + TILE - 1) / TILE,
                      (M + TILE - 1) / TILE);
        float ms = time_kernel(start, stop, gridSize, blockSize, mat_transpose_shared_kernel, d_dst, d_src, M, N);
        std::vector<float> h_result(N * M);
        cudaMemcpy(h_result.data(), d_dst, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = check_correctness(h_result, h_dst_cpu, N * M);
        printf("[shared_no_pad] Time: %.3f ms | Correct: %s\n", ms, ok ? "YES" : "NO");
        if (!ok) all_passed = false;
    }

    // -----------------------------
    // 4. shared memory with padding
    // -----------------------------
    {
        const int TILE = 32;
        dim3 blockSize(TILE, TILE);
        dim3 gridSize((N + TILE - 1) / TILE,
                      (M + TILE - 1) / TILE);
        float ms = time_kernel(start, stop, gridSize, blockSize, mat_transpose_shared_kernel_padding, d_dst, d_src, M, N);
        std::vector<float> h_result(N * M);
        cudaMemcpy(h_result.data(), d_dst, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        bool ok = check_correctness(h_result, h_dst_cpu, N * M);
        printf("[shared_padding] Time: %.3f ms | Correct: %s\n", ms, ok ? "YES" : "NO");
        if (!ok) all_passed = false;
    }

    // Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (all_passed) {
        printf("\n✅ All tests passed!\n");
    } else {
        printf("\n❌ Some tests failed!\n");
        return 1;
    }

    return 0;
}