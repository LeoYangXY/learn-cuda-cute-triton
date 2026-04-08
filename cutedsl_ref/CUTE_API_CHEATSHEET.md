# CuTeDSL API 复习速查表

这份表从 `cutedsl_ref` 目录下 11 个示例文件逐个提取而来，覆盖本目录里实际用到的 CuTeDSL、`cute.nvgpu`、`cpasync`、pipeline、运行时测试 API，以及少量和 CuTe kernel 紧密配套的 `cutlass` DSL 类型/工具。

涉及文件：

- `add.py`
- `transpose.py`
- `reduce_max.py`
- `sgemv.py`
- `embedding.py`
- `layer_norm.py`
- `gemm_boundary_handling.py`
- `sgemm.py`
- `hgemm.py`
- `data_movement_hardware_paths.py`
- `flash_attention_v4.py`

## 1. 装饰器与编译启动

| API | 背诵含义 | 常见文件 |
| --- | --- | --- |
| `@cute.kernel` | 声明 GPU device kernel 函数体。 | 几乎全部 |
| `@cute.jit` | 声明 host 侧 JIT 入口，通常内部配置 `.launch(...)`。 | 几乎全部 |
| `@cute.struct` | 定义 CuTeDSL 可识别的结构体，常用于 shared memory storage。 | `sgemm.py`, `hgemm.py`, `data_movement_hardware_paths.py`, `flash_attention_v4.py` |
| `cute.compile(...)` | 编译 `@cute.jit` 函数/类实例为可调用 kernel。 | 几乎全部 |
| `kernel(...).launch(grid=..., block=...)` | 指定 grid/block 启动 kernel。 | 几乎全部 |

## 2. 类型注解与 DSL 类型

| API | 背诵含义 | 常见文件 |
| --- | --- | --- |
| `cute.Tensor` | CuTe 张量/视图类型，可表示 GMEM、SMEM、RMEM 视图。 | 全部 |
| `cute.Layout` | 多维坐标到线性地址的映射。 | 多数文件 |
| `cute.Shape` | 逻辑 shape/tiler 类型注解。 | `add.py`, `embedding.py`, `layer_norm.py`, `data_movement_hardware_paths.py` |
| `cute.CopyAtom` | copy atom 的类型注解。 | `data_movement_hardware_paths.py` |
| `cute.TiledMma` | tiled MMA 对象类型注解。 | `flash_attention_v4.py` |
| `cute.TiledCopy` | tiled copy 对象类型注解。 | `flash_attention_v4.py` |
| `cute.ComposedLayout` | composed layout 类型，常用于 swizzle + base layout。 | `flash_attention_v4.py` |
| `cutlass.Numeric` | CUTLASS 数值类型泛型。 | `flash_attention_v4.py` |
| `cutlass.Float16` | fp16 DSL 数值类型。 | GEMM/data movement |
| `cutlass.Float32` | fp32 DSL 数值类型。 | 多数文件 |
| `cutlass.Int32` | int32 DSL 类型。 | `flash_attention_v4.py` |
| `cutlass.Int64` | int64 DSL 类型，常用于 mbarrier storage。 | TMA 相关 |
| `cutlass.Boolean` | DSL bool 类型。 | `flash_attention_v4.py` |
| `cutlass.Constexpr[...]` | 编译期常量类型注解。 | `gemm_boundary_handling.py`, `data_movement_hardware_paths.py` |

## 3. Layout 与坐标代数

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `cute.make_layout(shape, stride=...)` | 直接用 shape/stride 构造 layout。 | 基础布局、TV layout、staged SMEM |
| `cute.make_ordered_layout(shape, order=...)` | 按维度顺序构造 layout，常表达行主/列主和 padding。 | SMEM layout |
| `cute.make_composed_layout(...)` | 组合 swizzle 和基础 layout。 | FlashAttention SMEM |
| `cute.make_swizzle(...)` | 构造 XOR/swizzle 地址置换，减少 bank conflict。 | `hgemm.py`, `flash_attention_v4.py` |
| `cute.tile_to_shape(atom_layout, shape, order=...)` | 把小 layout atom 平铺到目标 shape。 | FlashAttention |
| `cute.logical_divide(...)` | 对 layout/tensor 做逻辑切分。 | FlashAttention |
| `cute.composition(...)` | 组合布局/张量视图，改变索引解释。 | FlashAttention |
| `cute.ceil_div(x, y)` | 向上整除，常用于 grid 维度。 | GEMM/data movement |
| `cute.cosize(layout)` | layout 覆盖的元素容量，常用于 SMEM buffer 大小。 | struct SMEM |
| `cute.size(x)` | 张量/fragment 元素个数。 | fragment loop |
| `cute.size(x, mode=[...])` | 查询指定 mode 的 size。 | FlashAttention |
| `cute.size_in_bytes(dtype, layout)` | dtype + layout 对应字节数，常用于 TMA transaction bytes。 | TMA |
| `cute.slice_(tensor_or_layout, coord)` | 静态切片，常切 stage 或 operand view。 | GEMM/TMA |
| `cute.group_modes(tensor, begin, end)` | 合并 mode，给 TMA 分区形成二维/分组视图。 | TMA |

## 4. 张量视图、分块与谓词

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `cute.local_tile(tensor, tiler, coord, proj=...)` | 按 CTA tile 和 block 坐标取全局子张量。 | GEMM、向量算子 |
| `cute.local_partition(tile, thr_layout, tid, shape=...)` | 按线程 layout 把 tile 分给单线程。 | `add.py` 等基础向量化 |
| `cute.make_tensor(iterator, layout)` | 用已有 iterator + 新 layout 包装张量视图。 | embedding、FlashAttention |
| `tensor.iterator` | 取张量底层 iterator，用于重塑视图。 | embedding、LayerNorm、FlashAttention |
| `cute.make_identity_tensor(shape)` | 构造坐标身份张量，用于生成 mask/predicate。 | FlashAttention |
| `cute.domain_offset(coord, tensor)` | 坐标域平移。 | FlashAttention |
| `cute.elem_less(lhs, rhs)` | 坐标逐元素比较，生成谓词。 | FlashAttention |
| `tensor[None, ...]` | 用 CuTe 多 mode 下标取子视图。 | 全部 GEMM |
| `tensor.fill(value)` | 填充 fragment/RMEM tensor。 | GEMM、FlashAttention |
| `tensor.load()` | 从 CuTe scalar/tensor SSA 位置读取值。 | FlashAttention |
| `tensor.store(value)` | 写入 CuTe scalar/tensor SSA 位置。 | FlashAttention |
| `tensor.to(dtype)` | 类型转换。 | FlashAttention |
| `tensor.reduce(op, init=...)` | 对 tensor/fragment 做归约。 | FlashAttention |
| `cute.ReductionOp.MAX` | 最大值归约操作。 | FlashAttention softmax |
| `cute.ReductionOp.ADD` | 求和归约操作。 | FlashAttention softmax |

## 5. Copy Atom 与 TiledCopy

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `cute.make_copy_atom(op, dtype, num_bits_per_copy=...)` | 构造一次 copy 的原子操作，绑定 op、dtype、位宽。 | 向量化 copy |
| `cute.nvgpu.CopyUniversalOp()` | 通用同步 LD/ST copy op。 | GMEM/SMEM/RMEM 普通拷贝 |
| `cute.make_tiled_copy(atom, thr_layout, val_layout)` | 构造 block 内协作 tiled copy。 | 基础向量化例子 |
| `cute.make_tiled_copy_tv(atom, thr_layout, val_layout)` | TV 形式构造 tiled copy，常用于每线程向量化搬运。 | GEMM G2S |
| `cute.make_tiled_copy_A(atom, tiled_mma)` | 生成匹配 TiledMMA A 操作数布局的 copy。 | LdMatrix A |
| `cute.make_tiled_copy_B(atom, tiled_mma)` | 生成匹配 TiledMMA B 操作数布局的 copy。 | LdMatrix B |
| `cute.make_tiled_copy_C(atom, tiled_mma)` | 生成匹配 TiledMMA C 操作数布局的 copy。 | FlashAttention |
| `tiled_copy.get_slice(tid)` | 当前线程在 tiled copy 里的切片句柄。 | copy 分区 |
| `thr_copy.partition_S(src)` | 对 copy 源张量分区。 | G2S/S2R/R2G |
| `thr_copy.partition_D(dst)` | 对 copy 目的张量分区。 | G2S/S2R/R2G |
| `thr_copy.retile(fragment)` | 将寄存器 fragment 重排成 copy 视角。 | LdMatrix/R2G |
| `cute.copy(atom_or_tiled_copy, src, dst)` | 执行 copy。 | 全部 copy 路径 |
| `cute.copy(..., pred=mask)` | 带谓词的 guarded copy。 | FlashAttention 边界 |
| `cute.copy(..., tma_bar_ptr=barrier)` | TMA copy 时绑定 mbarrier。 | TMA |
| `cute.autovec_copy(src, dst)` | 自动向量化 copy。 | data movement、FlashAttention |
| `cute.make_fragment_like(view, dtype=...)` | 创建与 view 同形的寄存器 fragment。 | copy staging、输出转换 |
| `cute.make_rmem_tensor(shape, dtype=...)` | 创建寄存器内存 tensor。 | LayerNorm、FlashAttention |

## 6. MMA 与 Tensor Core

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `cute.nvgpu.MmaUniversalOp(dtype)` | 通用 MMA op，示例里用于 fp32 CuTe GEMM。 | `sgemm.py` V3 |
| `cute.nvgpu.warp.MmaF16BF16Op(...)` | warp 级 fp16/bf16 Tensor Core MMA op。 | HGEMM/WMMA |
| `warp.MmaF16BF16Op(...)` | 同上，直接从 `cutlass.cute.nvgpu import warp` 后使用。 | FlashAttention |
| `cute.make_tiled_mma(op, atom_layout_mnk=..., permutation_mnk=...)` | 构造多 warp/多 atom 协作的 TiledMMA。 | WMMA GEMM |
| `cute.make_tiled_mma(op, atoms_layout)` | 用 atom layout 构造 TiledMMA。 | `sgemm.py` V3 |
| `tiled_mma.get_slice(tid)` | 当前线程在 TiledMMA 中负责的分片。 | GEMM |
| `thr_mma.partition_A(tensor)` | 按 MMA 布局划分 A 操作数。 | GEMM |
| `thr_mma.partition_B(tensor)` | 按 MMA 布局划分 B 操作数。 | GEMM |
| `thr_mma.partition_C(tensor)` | 按 MMA 布局划分 C 操作数。 | GEMM |
| `thr_mma.partition_shape_C(shape)` | 获取 C 分区形状。 | FlashAttention |
| `tiled_mma.make_fragment_A(partition)` | 创建 A 寄存器 fragment。 | GEMM |
| `tiled_mma.make_fragment_B(partition)` | 创建 B 寄存器 fragment。 | GEMM |
| `tiled_mma.make_fragment_C(partition)` | 创建 C accum fragment。 | GEMM |
| `thr_mma.make_fragment_A(partition)` | 线程 MMA 句柄创建 A fragment。 | data movement |
| `thr_mma.make_fragment_B(partition)` | 线程 MMA 句柄创建 B fragment。 | FlashAttention |
| `cute.gemm(tiled_mma, d, a, b, c)` | 执行 MMA/GEMM 累加。 | GEMM、FlashAttention |
| `cute.nvgpu.warp.LdMatrix8x8x16bOp(...)` | warp 级 ldmatrix，从 SMEM 读 Tensor Core 格式矩阵。 | WMMA |
| `warp.LdMatrix8x8x16bOp(...)` | 同上，直接导入 `warp` 后使用。 | FlashAttention |

## 7. `cute.arch` 硬件原语

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `cute.arch.thread_idx()` | 读取当前 threadIdx 三元组。 | 全部 kernel |
| `cute.arch.block_idx()` | 读取当前 blockIdx 三元组。 | 全部 kernel |
| `cute.arch.grid_dim()` | 读取 gridDim 三元组。 | block swizzle |
| `cute.arch.warp_idx()` | 读取当前 warp id。 | warp specialization |
| `cute.arch.make_warp_uniform(x)` | 让 warp 内值一致，便于分支。 | TMA/pipeline |
| `cute.arch.WARP_SIZE` | warp 大小常量，一般为 32。 | 线程数计算 |
| `cute.arch.sync_threads()` | block 内同步，相当于 `__syncthreads()`。 | SMEM 协作 |
| `cute.arch.shuffle_sync_bfly(mask, val, lane_mask)` | warp 内 butterfly shuffle。 | reduce、sgemv、LayerNorm |
| `cute.arch.cp_async_commit_group()` | 提交一组 cp.async。 | 手写 cp.async pipeline |
| `cute.arch.cp_async_wait_group(n)` | 等待 cp.async，只允许最多 n 组未完成。 | 手写 cp.async pipeline |
| `cute.arch.mbarrier_init(ptr, cnt=...)` | 初始化 mbarrier。 | TMA |
| `cute.arch.mbarrier_init_fence()` | mbarrier 初始化后的 fence。 | TMA |
| `cute.arch.mbarrier_arrive_and_expect_tx(ptr, bytes)` | 到达 mbarrier 并登记期望传输字节数。 | TMA |
| `cute.arch.mbarrier_wait(ptr, phase)` | 等待 mbarrier 指定 phase 完成。 | TMA |
| `cute.arch.fmax(a, b)` | 设备侧 max。 | FlashAttention |
| `cute.arch.rcp_approx(x)` | 近似倒数。 | FlashAttention softmax |

## 8. cp.async 与 TMA

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `cpasync.CopyG2SOp()` | cp.async 全局内存到共享内存 copy op。 | `sgemm.py`, data movement, FlashAttention |
| `cpasync.LoadCacheMode.GLOBAL` | cp.async 全局缓存模式。 | FlashAttention |
| `cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()` | TMA bulk tensor tile G2S 操作。 | TMA |
| `cpasync.CopyBulkTensorTileG2SOp()` | 同上，直接导入 `cpasync` 后使用。 | data movement |
| `cute.nvgpu.cpasync.make_tiled_tma_atom(op, tensor, smem_layout, smem_tile)` | 创建 TMA atom/descriptor。 | TMA GEMM |
| `cpasync.make_tiled_tma_atom(...)` | 同上，直接导入 `cpasync` 后使用。 | data movement |
| `cute.nvgpu.cpasync.tma_partition(...)` | 为 TMA 切出 SMEM/GMEM 分区。 | TMA GEMM |
| `cpasync.tma_partition(...)` | 同上，直接导入 `cpasync` 后使用。 | data movement |
| `cute.nvgpu.cpasync.prefetch_descriptor(tma_atom)` | 预取 TMA descriptor。 | TMA |

## 9. Pipeline API

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `PipelineCpAsync.create(...)` | 创建 cp.async 生产者-消费者 pipeline。 | `sgemm.py` V10 |
| `PipelineTmaAsync.create(...)` | 创建 TMA producer + async consumer pipeline。 | `sgemm.py` V11 |
| `PipelineAsync` | 通用 pipeline 类型；本目录中主要 import/注释提到，未直接实例化。 | `sgemm.py` |
| `CooperativeGroup(Agent.Thread, n)` | 描述 producer/consumer 线程组。 | pipeline |
| `Agent.Thread` | pipeline participant 的线程 agent 类型。 | pipeline |
| `mainloop_pipeline.make_participants()` | 生成 producer/consumer 参与者对象。 | `PipelineCpAsync` |
| `producer.acquire_and_advance()` | producer 获取可写 stage 并前进。 | `PipelineCpAsync` |
| `producer.tail()` | producer 收尾，通知 pipeline 结束。 | `PipelineCpAsync` |
| `consumer.wait_and_advance()` | consumer 等待可读 stage 并前进。 | `PipelineCpAsync` |
| `handle.index` | 当前 pipeline stage 索引。 | pipeline |
| `handle.commit()` | producer 提交 stage。 | `PipelineCpAsync` |
| `handle.release()` | consumer 释放 stage。 | pipeline |
| `make_pipeline_state(PipelineUserType.Producer, stages)` | 创建 producer 侧状态机。 | `PipelineTmaAsync` |
| `make_pipeline_state(PipelineUserType.Consumer, stages)` | 创建 consumer 侧状态机。 | `PipelineTmaAsync` |
| `producer_state.index` | producer 当前 stage 索引。 | `PipelineTmaAsync` |
| `producer_state.count` | producer 已处理 tile 计数。 | `PipelineTmaAsync` |
| `producer_state.advance()` | producer 状态前进。 | `PipelineTmaAsync` |
| `consumer_state.index` | consumer 当前 stage 索引。 | `PipelineTmaAsync` |
| `consumer_state.advance()` | consumer 状态前进。 | `PipelineTmaAsync` |
| `mainloop_pipeline.producer_acquire(state)` | TMA producer 获取可写 stage。 | `PipelineTmaAsync` |
| `mainloop_pipeline.producer_get_barrier(state)` | 取当前 stage 对应 mbarrier。 | `PipelineTmaAsync` |
| `mainloop_pipeline.producer_commit(state)` | TMA producer 提交；TMA 场景通常由硬件 tx_count 完成 arrive。 | `PipelineTmaAsync` |
| `mainloop_pipeline.consumer_wait(state)` | consumer 等待 stage ready。 | `PipelineTmaAsync` |
| `mainloop_pipeline.consumer_release(state)` | consumer 释放 stage。 | `PipelineTmaAsync` |
| `PipelineUserType.Producer` | pipeline producer 角色枚举。 | `PipelineTmaAsync` |
| `PipelineUserType.Consumer` | pipeline consumer 角色枚举。 | `PipelineTmaAsync` |
| `cutlass.pipeline.NamedBarrier` | 命名 barrier，用于 CTA 内更细同步。 | `flash_attention_v4.py` |

## 10. `cute.struct` 与共享内存

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `cute.struct.MemRange[dtype, n]` | 结构体字段：连续 n 个 dtype 元素。 | SharedStorage |
| `cute.struct.Align[memrange, align]` | 给 MemRange 添加对齐约束。 | TMA/SMEM |
| `cutlass.utils.SmemAllocator()` | kernel 内 shared memory 分配器。 | 多数 SMEM 示例 |
| `allocator.allocate_tensor(dtype, layout, align, swizzle=None)` | 分配一个 SMEM tensor。 | SMEM tile |
| `allocator.allocate(SharedStorage)` | 按 `@cute.struct` 分配整块 storage。 | TMA/pipeline |
| `storage.sA.get_tensor(layout=...)` | 从 storage 字段按 layout 取 tensor。 | pipeline |
| `storage.sA.get_tensor(outer, swizzle=inner)` | 从 storage 字段取 swizzled tensor。 | TMA |
| `storage.mbar_ptr.data_ptr()` | 取 mbarrier 原始指针。 | TMA |
| `storage.pipeline_mbar_ptr.data_ptr()` | 取 pipeline mbarrier 原始指针。 | pipeline |

## 11. 数学、编译期控制与常量

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `cute.math.rsqrt(x)` | 计算 `1 / sqrt(x)`。 | LayerNorm |
| `cute.math.exp2(x, fastmath=True)` | 计算 `2^x`，softmax 快速指数。 | FlashAttention |
| `cutlass.range_constexpr(...)` | 编译期展开循环。 | reduce、LayerNorm、FlashAttention |
| `cutlass.const_expr(expr)` | 编译期条件/表达式。 | FlashAttention |
| `cutlass.max(a, b)` | DSL max。 | FlashAttention |
| `cutlass.min(a, b)` | DSL min。 | FlashAttention |
| `cutlass.Float32.inf` | fp32 正无穷常量。 | FlashAttention |
| `cutlass.Float32.zero` | fp32 零常量。 | FlashAttention |

## 12. 运行时、测试与张量标注

| API | 背诵含义 | 常见用途 |
| --- | --- | --- |
| `from_dlpack(torch_tensor, assumed_align=...)` | 将 PyTorch/DLPack tensor 包装为 CuTe runtime tensor。 | 全部测试 |
| `benchmark(compiled, kernel_arguments=JitArguments(...))` | 对 compiled kernel 做微基准计时。 | 全部测试 |
| `JitArguments(...)` | benchmark 参数包装。 | 全部测试 |
| `tensor.mark_layout_dynamic(...)` | 标注 runtime tensor 的 layout 动态属性。 | FlashAttention |
| `tensor.mark_compact_shape_dynamic(...)` | 标注 compact shape 动态属性。 | FlashAttention |
| `utils.LayoutEnum.from_tensor(tensor)` | 从 tensor 推断 layout enum。 | TMA GEMM |
| `sm90_utils.make_smem_layout_a(...)` | 生成 SM90 A 操作数 swizzled/staged SMEM layout。 | TMA GEMM |
| `sm90_utils.make_smem_layout_b(...)` | 生成 SM90 B 操作数 swizzled/staged SMEM layout。 | TMA GEMM |

## 13. 按文件记忆版

### `add.py`

重点 API：

- `@cute.kernel`, `@cute.jit`, `cute.Tensor`, `cute.Layout`, `cute.Shape`
- `cute.arch.thread_idx`, `cute.arch.block_idx`
- `cute.local_tile`, `cute.local_partition`
- `cute.make_layout`, `cute.make_copy_atom`, `cute.make_tiled_copy`
- `cute.nvgpu.CopyUniversalOp`, `cute.copy`
- `tiled_copy.get_slice`, `partition_S`, `partition_D`
- `cute.make_fragment_like`, `cute.size`
- `from_dlpack`, `cute.compile`, `benchmark`, `JitArguments`

背诵定位：最基础的标量/向量化 load-store、thread 到数据分区、copy atom/tiled copy 入门。

### `transpose.py`

重点 API：

- `cute.arch.sync_threads`
- `cute.make_ordered_layout`
- `cutlass.utils.SmemAllocator`, `allocate_tensor`
- `cutlass.Float32`

背诵定位：SMEM tile + padding + block 内同步。

### `reduce_max.py`

重点 API：

- `cute.arch.shuffle_sync_bfly`
- `cutlass.range_constexpr`
- `SmemAllocator.allocate_tensor`
- `cute.make_layout`, `cute.arch.sync_threads`

背诵定位：block reduce 与 warp shuffle reduce。

### `sgemv.py`

重点 API：

- `cute.arch.shuffle_sync_bfly`
- `cutlass.range_constexpr`
- `cute.arch.thread_idx`, `cute.arch.block_idx`

背诵定位：warp 负责一行/多行的 GEMV，核心是 warp 内规约。

### `embedding.py`

重点 API：

- `cute.local_tile`
- `cute.make_tensor`
- `tensor.iterator`
- `cute.make_tiled_copy`
- `cute.nvgpu.CopyUniversalOp`

背诵定位：通过 iterator + layout 重塑向量视图，再做向量化拷贝。

### `layer_norm.py`

重点 API：

- `cute.arch.shuffle_sync_bfly`, `cutlass.range_constexpr`
- `cute.math.rsqrt`
- `cute.make_rmem_tensor`
- `cute.make_tensor`, `tensor.iterator`
- `cute.make_copy_atom`, `cute.make_tiled_copy`, `cute.copy`

背诵定位：规约、寄存器缓存、向量化读取、归一化数学函数。

### `gemm_boundary_handling.py`

重点 API：

- `cute.Tensor`, `cute.Layout`, `cutlass.Constexpr`
- `cute.make_ordered_layout`
- `cutlass.utils.SmemAllocator`, `allocate_tensor`
- `cute.arch.sync_threads`

背诵定位：GEMM 边界处理、SMEM 填零/越界判断。

### `sgemm.py`

重点 API：

- `cute.nvgpu.MmaUniversalOp`
- `cute.nvgpu.warp.MmaF16BF16Op`
- `cute.nvgpu.warp.LdMatrix8x8x16bOp`
- `cute.make_tiled_mma`, `cute.gemm`
- `cute.make_tiled_copy_tv`, `make_tiled_copy_A`, `make_tiled_copy_B`
- `cpasync.CopyG2SOp`
- `cute.arch.cp_async_commit_group`, `cute.arch.cp_async_wait_group`
- `cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp`
- `cute.nvgpu.cpasync.make_tiled_tma_atom`
- `cute.nvgpu.cpasync.tma_partition`
- `cute.arch.mbarrier_*`
- `PipelineCpAsync`, `PipelineTmaAsync`, `CooperativeGroup`, `Agent`, `make_pipeline_state`, `PipelineUserType`
- `sm90_utils.make_smem_layout_a/b`, `utils.LayoutEnum.from_tensor`

背诵定位：GEMM 优化总复习，从 naive/tiled 到 WMMA、TMA、cp.async、pipeline、warp specialization。

### `hgemm.py`

重点 API：

- `cute.make_swizzle`
- `cute.make_ordered_layout`
- `allocate_tensor(..., swizzle=...)`
- `cute.arch.grid_dim`
- `cute.nvgpu.warp.MmaF16BF16Op`
- `cute.nvgpu.warp.LdMatrix8x8x16bOp`
- `cute.nvgpu.cpasync` TMA 系列

背诵定位：HGEMM 专项，SMEM swizzle、block swizzle、WMMA/TMA。

### `data_movement_hardware_paths.py`

重点 API：

- `cute.autovec_copy`
- `cpasync.CopyG2SOp`
- `cute.arch.cp_async_commit_group`, `cute.arch.cp_async_wait_group`
- `cpasync.CopyBulkTensorTileG2SOp`
- `cpasync.make_tiled_tma_atom`
- `cpasync.tma_partition`
- `cute.arch.mbarrier_*`
- `cute.struct.MemRange`, `cute.struct.Align`

背诵定位：硬件数据搬运路径总览：普通 LD/ST、cp.async、ldmatrix、TMA。

### `flash_attention_v4.py`

重点 API：

- `cute.make_composed_layout`, `cute.make_swizzle`, `cute.tile_to_shape`
- `cute.logical_divide`, `cute.composition`
- `cute.make_identity_tensor`, `cute.domain_offset`, `cute.elem_less`
- `cute.copy(..., pred=...)`
- `warp.MmaF16BF16Op`, `warp.LdMatrix8x8x16bOp`
- `cute.make_tiled_copy_A/B/C`, `cute.gemm`
- `cute.make_rmem_tensor`, `tensor.reduce`, `cute.ReductionOp.MAX`, `cute.ReductionOp.ADD`
- `cute.math.exp2`, `cute.arch.fmax`, `cute.arch.rcp_approx`
- `cutlass.pipeline.NamedBarrier`
- `mark_layout_dynamic`, `mark_compact_shape_dynamic`

背诵定位：综合应用题，包含 swizzled SMEM、mask/predicate、MMA、softmax reduce、寄存器张量、动态 layout 标注。

## 14. 考前最该背的主线

1. Kernel 壳：`@cute.jit` 里建 layout/copy/mma，然后 `.launch`；`@cute.kernel` 里用 `thread_idx`/`block_idx` 做实际工作。
2. Layout 主线：`make_layout`、`make_ordered_layout`、`local_tile`、`slice_`、`group_modes`。
3. Copy 主线：`make_copy_atom` -> `make_tiled_copy(_tv)` -> `get_slice` -> `partition_S/D` -> `cute.copy`。
4. MMA 主线：`MmaF16BF16Op` -> `make_tiled_mma` -> `partition_A/B/C` -> `make_fragment_A/B/C` -> `LdMatrix` -> `cute.gemm`。
5. 异步主线：`cpasync.CopyG2SOp` + `commit/wait_group`，或 TMA `make_tiled_tma_atom` + `tma_partition` + `mbarrier`。
6. Pipeline 主线：`PipelineCpAsync` 是 SM 线程发 cp.async；`PipelineTmaAsync` 是 TMA 硬件搬运 + consumer warp 计算。
7. FlashAttention 主线：layout/swizzle + predicated copy + MMA + online softmax reduce + epilogue。
