"""
=============================================================================
tv_layout 映射表验证
=============================================================================
验证三种 tv_layout 的映射关系：

  1. 1D 元素 256个，64 线程，连续分配：thread0→[0,1,2,3], thread1→[4,5,6,7]
     tv_layout = (64, 4):(4, 1)

  2. 1D 元素 256个，64 线程，交错分配：thread0→[0,64,128,192], thread1→[1,65,129,193]
     tv_layout = (64, 4):(1, 64)

  3. 2D 元素 16×16=256个，8 线程，每线程 2行×16列
     tv_layout = (8, (16, 2)):(32, (1, 16))
=============================================================================
"""

import cutlass
import cutlass.cute as cute


# ============================================================================
# 案例 1: 1D 连续分配
# ============================================================================
@cute.jit
def case1_contiguous():
    cute.printf("======================================================================\n")
    cute.printf("  案例1: 1D元素, 64线程, 连续分配\n")
    cute.printf("  tv_layout = make_layout((64, 4), (4, 1))\n")
    cute.printf("======================================================================\n")

    tv = cute.make_layout(shape=(64, 4), stride=(4, 1))
    cute.printf("tv_layout = {}\n\n", tv)

    cute.printf("映射表 (t_idx -> v_idx 对应的元素编号):\n")
    cute.printf("           v=0   v=1   v=2   v=3\n")
    for t in range(5):
        v0 = cute.crd2idx((t, 0), tv)
        v1 = cute.crd2idx((t, 1), tv)
        v2 = cute.crd2idx((t, 2), tv)
        v3 = cute.crd2idx((t, 3), tv)
        cute.printf("  t=%2d :  %4d  %4d  %4d  %4d\n", t, v0, v1, v2, v3)
    cute.printf("  ...  \n")
    for t in range(62, 64):
        v0 = cute.crd2idx((t, 0), tv)
        v1 = cute.crd2idx((t, 1), tv)
        v2 = cute.crd2idx((t, 2), tv)
        v3 = cute.crd2idx((t, 3), tv)
        cute.printf("  t=%2d :  %4d  %4d  %4d  %4d\n", t, v0, v1, v2, v3)


# ============================================================================
# 案例 2: 1D 交错分配
# ============================================================================
@cute.jit
def case2_interleaved():
    cute.printf("\n======================================================================\n")
    cute.printf("  案例2: 1D元素, 64线程, 交错分配\n")
    cute.printf("  tv_layout = make_layout((64, 4), (1, 64))\n")
    cute.printf("======================================================================\n")

    tv = cute.make_layout(shape=(64, 4), stride=(1, 64))
    cute.printf("tv_layout = {}\n\n", tv)

    cute.printf("映射表 (t_idx -> v_idx 对应的元素编号):\n")
    cute.printf("           v=0   v=1   v=2   v=3\n")
    for t in range(5):
        v0 = cute.crd2idx((t, 0), tv)
        v1 = cute.crd2idx((t, 1), tv)
        v2 = cute.crd2idx((t, 2), tv)
        v3 = cute.crd2idx((t, 3), tv)
        cute.printf("  t=%2d :  %4d  %4d  %4d  %4d\n", t, v0, v1, v2, v3)
    cute.printf("  ...  \n")
    for t in range(62, 64):
        v0 = cute.crd2idx((t, 0), tv)
        v1 = cute.crd2idx((t, 1), tv)
        v2 = cute.crd2idx((t, 2), tv)
        v3 = cute.crd2idx((t, 3), tv)
        cute.printf("  t=%2d :  %4d  %4d  %4d  %4d\n", t, v0, v1, v2, v3)


# ============================================================================
# 案例 3: 2D 元素 (16×16), 8 线程, 每线程 2行×16列
# ============================================================================
@cute.jit
def case3_2d_elements():
    cute.printf("\n======================================================================\n")
    cute.printf("  案例3: 2D元素(16x16), 8线程, 每线程2行x16列\n")
    cute.printf("  tv_layout = make_layout((8, (16, 2)), (32, (1, 16)))\n")
    cute.printf("======================================================================\n")

    tv = cute.make_layout(shape=(8, (16, 2)), stride=(32, (1, 16)))
    cute.printf("tv_layout = {}\n\n", tv)

    cute.printf("映射表: 每个 t_idx 拥有 2行x16列, 显示 1D 编号:\n\n")

    for t in range(8):
        cute.printf("  t=%d ->\n", t)
        # row_off=0: v_coord = (col, 0), col = 0..15
        cute.printf("    row%2d: [", t * 2)
        for col in range(16):
            v0 = cute.crd2idx((t, (col, 0)), tv)
            if col > 0:
                cute.printf(", ")
            cute.printf("%3d", v0)
        cute.printf("]\n")

        # row_off=1: v_coord = (col, 1), col = 0..15
        cute.printf("    row%2d: [", t * 2 + 1)
        for col in range(16):
            v1 = cute.crd2idx((t, (col, 1)), tv)
            if col > 0:
                cute.printf(", ")
            cute.printf("%3d", v1)
        cute.printf("]\n\n")

    # 验证: 把 1D 编号还原成 (row, col) 坐标
    cute.printf("验证: 1D编号 -> (row, col), 元素排布 16x16 行优先\n\n")
    for t in range(3):  # 只打印前 3 个线程
        cute.printf("  t=%d ->\n", t)
        for row_off in range(2):
            cute.printf("    [")
            for col in range(16):
                elem = cute.crd2idx((t, (col, row_off)), tv)
                r = elem / 16
                c = elem % 16
                if col > 0:
                    cute.printf(", ")
                cute.printf("(%2d,%2d)", r, c)
            cute.printf("]\n")
        cute.printf("\n")


# ============================================================================
# 主函数
# ============================================================================
cutlass.cuda.initialize_cuda_context()

case1_contiguous()
case2_interleaved()
case3_2d_elements()

print("\n✅ 所有案例验证完成!")
