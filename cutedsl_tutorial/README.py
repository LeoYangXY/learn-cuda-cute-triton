"""
=============================================================================
CuTeDSL 学习教程 —— 从零到 Flash Attention
=============================================================================

教程目录：
  01_hello_and_vecadd.py    ✅ 可运行  基础：线程模型、向量加法、cute.compile
  02_layout.py              ✅ 可运行  核心：Layout 布局代数、坐标映射、转置
  03_tiling_and_gemm.py     ✅ 可运行  关键：Tiling 分块、local_tile、MMA 分区
  04_smem_and_pipeline.py   ✅ 可运行  进阶：Shared Memory、异步流水线
  05_tensor_core_wmma.py    ✅ 可运行  硬件：WMMA Tensor Core、向量化拷贝、LdMatrix
  06_hopper_tma_wgmma.py    📖 概念    Hopper：TMA、WGMMA、Warp 特化（需 SM90）
  07_blackwell_tcgen05.py   📖 概念    Blackwell：tcgen05 UMMA、TMEM、2CTA（需 SM100）
  08_flash_attention.py     ✅ 可运行  应用：Naive SDPA、Flash Attention V1/V2（Online Softmax）
  09_advanced_techniques.py ✅ 可运行  进阶：Persistent Kernel、Dynamic Shape、zipped_divide

性能对比（RTX 5050, SM120）：
  教程 01 向量加法：  CuTeDSL 向量化版 ≈ 0.92x PyTorch
  教程 05 WMMA GEMM： CuTeDSL ≈ 0.93x PyTorch（26 vs 28 TFLOPS）
  教程 08 Flash Attn： V1=0.17x, V2=0.39x PyTorch SDPA

知识体系图：

  ┌─────────────────────────────────────────────────────────────┐
  │                    CuTeDSL 知识体系                          │
  ├─────────────────────────────────────────────────────────────┤
  │                                                             │
  │  基础层（01-02）                                             │
  │  ├── @cute.kernel / @cute.jit 装饰器                        │
  │  ├── thread_idx / block_idx / block_dim                     │
  │  ├── cute.printf vs print                                   │
  │  ├── from_dlpack / cute.compile / benchmark                 │
  │  ├── Layout = (Shape, Stride) 布局代数                      │
  │  ├── make_layout / make_ordered_layout                      │
  │  ├── crd2idx / idx2crd 坐标映射                             │
  │  └── make_fragment / print_tensor                           │
  │                                                             │
  │  分块层（03-04）                                             │
  │  ├── local_tile(tensor, tiler, coord, proj)                 │
  │  ├── zipped_divide / flat_divide                            │
  │  ├── MmaUniversalOp + make_tiled_mma                        │
  │  ├── partition_A / partition_B / partition_C                 │
  │  ├── make_fragment_A/B/C                                    │
  │  ├── SmemAllocator / allocate_tensor                        │
  │  ├── sync_threads / bank conflict / padding                 │
  │  └── PipelineAsync（生产者-消费者模型）                      │
  │                                                             │
  │  硬件层（05-07）                                             │
  │  ├── WMMA: MmaF16BF16Op + LdMatrix + 向量化拷贝             │
  │  ├── TiledCopy: make_tiled_copy_tv / make_tiled_copy_A/B    │
  │  ├── TMA: CopyBulkTensorTileG2SOp + mbarrier               │
  │  ├── WGMMA: Warp Group MMA（SM90, 从 SMEM 直读）            │
  │  ├── tcgen05 UMMA: 统一 MMA（SM100, 单线程发射）            │
  │  ├── TMEM: 张量内存（SM100, MMA 累加器存储）                 │
  │  ├── Swizzle: 消除 bank conflict                            │
  │  ├── Warp Specialization: TMA/MMA/Epilogue 分工             │
  │  └── 2CTA 协作 MMA（SM100）                                 │
  │                                                             │
  │  应用层（08-09）                                              │
  │  ├── Scaled Dot-Product Attention                           │
  │  ├── Flash Attention（Online Softmax + Warp Reduce）        │
  │  ├── Persistent Kernel（持久化 kernel）                     │
  │  ├── Dynamic Shape（动态形状，编译一次多次复用）             │
  │  ├── zipped_divide（层次化分块）                             │
  │  └── Flash Attention v1→v4 演进                             │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘

GPU 架构演进与 CuTE 特性对应：

  ┌──────────┬────────┬─────────────────────────────────────────┐
  │ 架构      │ SM     │ CuTE 关键特性                           │
  ├──────────┼────────┼─────────────────────────────────────────┤
  │ Ampere   │ SM80   │ WMMA, LdMatrix, cp.async               │
  │ Hopper   │ SM90   │ TMA, WGMMA, mbarrier, Cluster          │
  │ Blackwell│ SM100  │ tcgen05 UMMA, TMEM, 2CTA MMA           │
  │ (消费级) │ SM120  │ WMMA, TMA, Block-Scaled MMA(FP4)        │
  │          │        │ 不支持 WGMMA/tcgen05 UMMA               │
  └──────────┴────────┴─────────────────────────────────────────┘

运行方式：
  conda activate cutedsl
  python 01_hello_and_vecadd.py
  python 02_layout.py
  ...

环境要求：
  - NVIDIA GPU (SM80+)
  - CUDA Toolkit 12.0+
  - nvidia-cutlass-dsl >= 4.4.2
  - PyTorch >= 2.0
=============================================================================
"""

if __name__ == "__main__":
    import os
    tutorial_dir = os.path.dirname(os.path.abspath(__file__))
    files = sorted([f for f in os.listdir(tutorial_dir) if f.endswith('.py') and f != '__init__.py' and f != 'README.py'])

    print("=" * 60)
    print("CuTeDSL 学习教程 —— 从零到 Flash Attention")
    print("=" * 60)
    print()
    for f in files:
        print(f"  📄 {f}")
    print()
    print("运行方式: conda activate cutedsl && python <文件名>")
    print()
    print("建议学习顺序: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09")
    print("其中 01-05、08-09 可以在你的 GPU 上运行验证")
    print("06-07 是概念讲解，阅读代码和注释即可")