"""
=============================================================================
教程 15: @dsl_user_op —— 自定义 PTX 内联汇编
=============================================================================

当 CuTeDSL 没有提供你需要的硬件指令时，@dsl_user_op 让你直接嵌入 PTX 汇编。

什么是 @dsl_user_op？
  一个装饰器，允许在 CuTeDSL 编译流中插入自定义的 LLVM IR / PTX 操作。
  最典型的用法是通过 llvm.inline_asm() 嵌入原始 PTX 指令。

使用场景：
  1. 读取 GPU 特殊寄存器（如 %globaltimer, %smid, %clock）
  2. Blackwell tcgen05 fence 指令（CuTeDSL 可能未封装）
  3. Cluster 级跨 CTA 操作（mapa, st.async.shared::cluster）
  4. TMA Bulk Copy S2G（SMEM → GMEM 异步批量拷贝）
  5. 任何 CuTeDSL 未直接提供的 PTX 指令

基本签名：
  @dsl_user_op
  def my_op(arg1, arg2, *, loc=None, ip=None) -> ReturnType:
      # loc, ip 是 MLIR 编译器基础设施要求的关键字参数
      # 函数体中使用 llvm.inline_asm() 嵌入 PTX
      ...

PTX 寄存器约束：
  "=r" — 输出 32 位通用寄存器
  "=l" — 输出 64 位通用寄存器
  "r"  — 输入 32 位通用寄存器
  "l"  — 输入 64 位通用寄存器
  "f"  — 输入 32 位浮点寄存器

关键 API：
  @dsl_user_op                     — 装饰器
  llvm.inline_asm(result, ops, asm, constraints, ...)  — PTX 内联汇编
  T.i32() / T.i64()               — MLIR 整数类型
  val.ir_value()                   — 获取 DSL 值的 MLIR IR 表示
=============================================================================
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


# =============================================================================
# 示例 1: 读取 GPU 全局计时器 —— %globaltimer
# =============================================================================
# PTX: mov.u64 $0, %globaltimer;
# 返回 GPU 全局纳秒级计时器，用于 kernel 内部性能分析

@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> cutlass.Int64:
    """读取 GPU 全局计时器（纳秒精度）"""
    t = llvm.inline_asm(
        T.i64(),           # 返回类型：64 位整数
        [],                # 无输入操作数
        "mov.u64 $0, %globaltimer;",  # PTX 汇编：读特殊寄存器
        "=l",              # 约束：输出到 64 位寄存器
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int64(t)


# =============================================================================
# 示例 2: 读取 SM ID —— %smid
# =============================================================================
# PTX: mov.u32 $0, %smid;
# 获取当前线程所在的 SM 编号

@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> cutlass.Int32:
    """读取当前 SM 的编号"""
    t = llvm.inline_asm(
        T.i32(),           # 返回类型：32 位整数
        [],                # 无输入操作数
        "mov.u32 $0, %smid;",
        "=r",              # 约束：输出到 32 位寄存器
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


# =============================================================================
# 示例 3: 读取 Warp 级时钟 —— %clock
# =============================================================================

@dsl_user_op
def clock_u32(*, loc=None, ip=None) -> cutlass.Int32:
    """读取 per-SM 时钟计数器"""
    t = llvm.inline_asm(
        T.i32(), [],
        "mov.u32 $0, %clock;",
        "=r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(t)


# =============================================================================
# 示例 4: 带输入参数 —— 整数加法（演示 PTX 输入/输出约束）
# =============================================================================

@dsl_user_op
def ptx_add_i32(
    a: cutlass.Int32, b: cutlass.Int32,
    *, loc=None, ip=None,
) -> cutlass.Int32:
    """
    用 PTX 内联汇编实现 32 位整数加法（纯演示用途）。
    展示如何传递输入参数和获取输出。
    """
    result = llvm.inline_asm(
        T.i32(),                          # 返回类型
        [a.ir_value(), b.ir_value()],     # 输入操作数列表
        "add.s32 $0, $1, $2;",           # PTX: $0 = $1 + $2
        "=r,r,r",                         # 约束：输出=r, 输入r,r
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip,
    )
    return cutlass.Int32(result)


# =============================================================================
# 在 kernel 中使用 @dsl_user_op
# =============================================================================

@cute.kernel
def user_op_demo_kernel(gOutput: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    if tidx == 0 and bidx == 0:
        # 示例 1: 读取全局计时器
        t_start = globaltimer_u64()

        # 示例 2: 读取 SM ID
        sm = smid_u32()

        # 示例 3: 读取时钟
        clk = clock_u32()

        # 示例 4: PTX 加法
        a = cutlass.Int32(42)
        b = cutlass.Int32(58)
        c = ptx_add_i32(a, b)

        t_end = globaltimer_u64()

        cute.printf("SM ID: %d\n", sm)
        cute.printf("Clock: %d\n", clk)
        cute.printf("PTX add: %d + %d = %d\n", a, b, c)
        cute.printf("Timer: start=%ld, end=%ld, diff=%ld ns\n",
                     t_start, t_end, t_end - t_start)

        # 写入结果供 host 验证
        gOutput[0] = cutlass.Float32(c)  # 应该是 100.0
        gOutput[1] = cutlass.Float32(sm)


@cute.jit
def user_op_demo(mOutput: cute.Tensor):
    user_op_demo_kernel(mOutput).launch(
        grid=(1, 1, 1), block=(32, 1, 1)
    )


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()

    print("=" * 60)
    print("教程 15: @dsl_user_op 自定义 PTX 操作")
    print("=" * 60)
    print()

    output = torch.zeros(4, device="cuda", dtype=torch.float32)
    output_ = from_dlpack(output, assumed_align=16)

    compiled = cute.compile(user_op_demo, output_)
    compiled(output_)
    torch.cuda.synchronize()

    # 验证 PTX add 结果
    ptx_add_result = output[0].item()
    sm_id = output[1].item()
    assert ptx_add_result == 100.0, f"PTX add 失败: {ptx_add_result} != 100"
    print(f"\n  PTX add 验证通过! (42 + 58 = {int(ptx_add_result)})")
    print(f"  运行在 SM {int(sm_id)} 上")

    print()
    print("=" * 60)
    print("@dsl_user_op 使用要点")
    print("=" * 60)
    print()
    print("  1. 函数必须接受 *, loc=None, ip=None 关键字参数")
    print("  2. 用 llvm.inline_asm() 嵌入 PTX 汇编")
    print("  3. 约束字符串: =r(输出32位), =l(输出64位), r(输入32位), l(输入64位), f(浮点)")
    print("  4. 用 val.ir_value() 获取 DSL 值的 MLIR IR 表示")
    print("  5. 用 ptr.toint().ir_value() 获取指针的整数 IR 表示")
    print()
    print("  实际应用场景:")
    print("    - globaltimer / smid: kernel 内部性能分析")
    print("    - tcgen05.fence: Blackwell MMA 前的同步屏障")
    print("    - mapa.shared::cluster: Hopper 跨 CTA 地址映射")
    print("    - st.async.shared::cluster: 跨 CTA 异步存储")
    print("    - cp.async.bulk.global.shared: TMA S2G 批量拷贝")
    print()
    print("  教程 15 完成!")
