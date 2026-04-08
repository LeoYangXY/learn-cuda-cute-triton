"""
=============================================================================
教程 13: Constexpr、cute.compile 与 Lambda 泛型
=============================================================================

CuTeDSL 的三个编译期特性，让 Python 拥有 C++ 模板级别的代码生成能力：

1. cute.compile(fn, *args) — 预编译
   - 将 @cute.jit 函数 + 所有参数（Tensor、标量、lambda）一次性编译为 CUDA kernel
   - 编译只发生一次，后续调用零开销
   - 参数的类型/形状在编译时完全特化

2. cutlass.Constexpr — 编译期常量
   - 标量（如 alpha=2.5）在编译时被烘焙为 PTX 立即数
   - Lambda 函数在编译时被 AST 内联展开
   - 运行时无任何额外开销

3. cutlass.range_constexpr(N) — 编译期循环展开
   - 循环变量在编译时已知，编译器完全展开
   - 等价于 C++ 的 #pragma unroll 或模板元编程

关键 API：
  cute.compile(fn, *args)          — 预编译 @cute.jit 函数
  cutlass.Constexpr                — 编译期常量类型标注
  cutlass.range_constexpr(N)       — 编译期循环展开
=============================================================================
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.testing import benchmark, JitArguments

N = 10000
TPB = 256


# =============================================================================
# 第一部分：cute.compile — 预编译分离编译和执行
# =============================================================================

@cute.kernel
def vecadd_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * TPB + tidx
    if idx < cute.size(gA):
        gC[idx] = gA[idx] + gB[idx]


@cute.jit
def vecadd(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
    n = cute.size(mA)
    num_blocks = (n + TPB - 1) // TPB
    vecadd_kernel(mA, mB, mC).launch(
        grid=(num_blocks, 1, 1), block=(TPB, 1, 1)
    )


# =============================================================================
# 第二部分：Constexpr — 编译期常量
# =============================================================================
# C = alpha * A + beta * B（AXPBY）
# alpha, beta 作为 Constexpr 传入 → 编译时被内联为常量

@cute.kernel
def axpby_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    alpha: cutlass.Constexpr, beta: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * TPB + tidx
    if idx < cute.size(gA):
        gC[idx] = alpha * gA[idx] + beta * gB[idx]


@cute.jit
def axpby(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    alpha: cutlass.Constexpr, beta: cutlass.Constexpr,
):
    n = cute.size(mA)
    num_blocks = (n + TPB - 1) // TPB
    axpby_kernel(mA, mB, mC, alpha, beta).launch(
        grid=(num_blocks, 1, 1), block=(TPB, 1, 1)
    )


# =============================================================================
# 第三部分：Lambda 泛型 — 一个 kernel 多种操作
# =============================================================================
# 通过 Constexpr 传入 lambda，编译时内联展开 → 零成本抽象

@cute.kernel
def binary_kernel(
    gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor,
    op: cutlass.Constexpr,  # lambda 通过 Constexpr 传入
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    idx = bidx * TPB + tidx
    if idx < cute.size(gA):
        gC[idx] = op(gA[idx], gB[idx])  # 像普通函数一样调用


@cute.jit
def binary_op(
    mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
    op: cutlass.Constexpr,
):
    n = cute.size(mA)
    num_blocks = (n + TPB - 1) // TPB
    binary_kernel(mA, mB, mC, op).launch(
        grid=(num_blocks, 1, 1), block=(TPB, 1, 1)
    )


# =============================================================================
# 第四部分：range_constexpr — 编译期循环展开
# =============================================================================

@cute.jit
def constexpr_loop_demo():
    """
    range_constexpr 生成编译期已知的循环，编译器完全展开。
    循环变量在每次迭代中是编译期常量。
    """
    layout = cute.make_layout(shape=(4, 3), stride=(3, 1))
    cute.printf("range_constexpr 循环展开演示:\n")
    cute.printf("Layout: {}\n", layout)

    # cutlass.range_constexpr 的循环变量是编译期常量
    # 可以用于需要常量表达式的场景（如 layout 索引）
    for i in cutlass.range_constexpr(4):
        for j in cutlass.range_constexpr(3):
            offset = cute.crd2idx((i, j), layout)
            cute.printf("  layout(%d, %d) = %d\n", i, j, offset)


# =============================================================================
# 主函数
# =============================================================================

if __name__ == "__main__":
    cutlass.cuda.initialize_cuda_context()

    # ---- Part 1: cute.compile ----
    print("=" * 60)
    print("第一部分：cute.compile 预编译")
    print("=" * 60)

    A = torch.randn(N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, device="cuda", dtype=torch.float32)
    C = torch.zeros(N, device="cuda", dtype=torch.float32)

    A_ = from_dlpack(A)
    B_ = from_dlpack(B)
    C_ = from_dlpack(C)

    # 编译（只发生一次）
    compiled_vecadd = cute.compile(vecadd, A_, B_, C_)
    # 执行（后续调用零开销）
    compiled_vecadd(A_, B_, C_)

    assert torch.allclose(C, A + B, atol=1e-5)
    print("  cute.compile 向量加法验证通过!")

    # 性能对比
    t = benchmark(compiled_vecadd, kernel_arguments=JitArguments(A_, B_, C_))
    print(f"  编译后执行耗时: {t:.2f} us")

    # ---- Part 2: Constexpr ----
    print("\n" + "=" * 60)
    print("第二部分：Constexpr 编译期常量")
    print("=" * 60)

    alpha, beta = 2.5, -1.3
    C.zero_()

    # alpha, beta 作为 Constexpr 传入 → 编译时被内联为常量
    compiled_axpby = cute.compile(axpby, A_, B_, C_, alpha, beta)
    compiled_axpby(A_, B_, C_)

    expected = alpha * A + beta * B
    assert torch.allclose(C, expected, atol=1e-4)
    print(f"  AXPBY 验证通过! (alpha={alpha}, beta={beta})")
    print("  alpha 和 beta 在编译时被内联为 PTX 立即数常量。")

    # ---- Part 3: Lambda 泛型 ----
    print("\n" + "=" * 60)
    print("第三部分：Lambda 泛型")
    print("=" * 60)

    # 加法
    C.zero_()
    add_op = lambda a, b: a + b
    compiled_add = cute.compile(binary_op, A_, B_, C_, add_op)
    compiled_add(A_, B_, C_)
    assert torch.allclose(C, A + B, atol=1e-5)
    print("  Lambda 加法验证通过!")

    # 乘法（需要重新编译，因为 lambda 不同）
    C.zero_()
    mul_op = lambda a, b: a * b
    compiled_mul = cute.compile(binary_op, A_, B_, C_, mul_op)
    compiled_mul(A_, B_, C_)
    assert torch.allclose(C, A * B, atol=1e-4)
    print("  Lambda 乘法验证通过!")

    print("  同一个 binary_kernel，通过传入不同 lambda 生成了 2 个特化版本。")
    print("  每个 lambda 在编译时被 AST 内联，运行时零开销。")

    # ---- Part 4: range_constexpr ----
    print("\n" + "=" * 60)
    print("第四部分：range_constexpr 编译期循环展开")
    print("=" * 60)
    constexpr_loop_demo()
    torch.cuda.synchronize()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print()
    print("  cute.compile(fn, *args):")
    print("    - 编译一次，执行多次")
    print("    - 参数类型/形状在编译时完全特化")
    print()
    print("  cutlass.Constexpr:")
    print("    - 标量 → PTX 立即数（无内存读取）")
    print("    - Lambda → 编译时 AST 内联（无间接调用）")
    print()
    print("  cutlass.range_constexpr(N):")
    print("    - 编译时完全展开循环")
    print("    - 循环变量是编译期常量")
    print()
    print("  教程 13 完成!")
