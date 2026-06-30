# hopper_fa3

最简的 FA3 baseline：用同一份输入，把**你写的 CUDA kernel** 和**官方 FA3**（或 torch SDPA）做对比。

## 文件
- `csrc/my_fa3_kernel.cu` — ★ **你只改这一个文件**（CUDA kernel + pybind）
- `run.py` — 编译你的扩展 + 跑对比
- `setup_env.sh` — 新机器一键配环境
- `requirements.txt` — Python 依赖

## 怎么用
新机器：
```bash
bash setup_env.sh
```
它依次：装系统依赖 → 装 Python 依赖 → 装官方 flash-attn（Hopper 上即 FA3）→ 编译你的扩展 → 跑一次对比。
（flash-attn 装不上会自动回退到 torch SDPA，不卡流程。）

日常改完 kernel 后：
```bash
python3 run.py                # dummy 模式，默认用官方 FA3 作基准
python3 run.py --official sdpa  # 不想用 FA3，退回到 torch SDPA 基准
python3 run.py --real         # 当你在 csrc 里写了真 attention，开启正确性对比
```

可选参数：`--batch --heads --seqlen --head-dim --dtype --causal --official auto|fa3|sdpa`

## 关于"官方 FA3"
官方 `flash-attn` 这个包在 Hopper 上 `flash_attn_func` 默认就走 FA3 内核——不是单独一个包，
而是同一个包在 SM90 上自动 dispatch 到 FA3。`setup_env.sh` 默认就装它；没装上时对比自动回退到 `torch_sdpa`。

## 装不上 FA3 的两个坑（已修好）
1. **构建隔离**：`pip install flash-attn` 默认会建一个没装 torch 的临时环境，而 flash-attn
   的 `setup.py` 一开头就要 `import torch` → 报 `ModuleNotFoundError: No module named 'torch'`。
   修法：`setup_env.sh` 用 `--no-build-isolation` 复用系统已装的 torch。
2. **torch 版本 + cudnn 路径**：FA3 在 flash-attn>=2.6 才引入，需要 torch>=2.2；而 pip 版 torch
   把 cudnn/cublas 等拆到 `nvidia-*` 包，其 `.so` 不在链接器默认路径 → `import torch` 报
   `libcudnn.so.9: cannot open shared object file`。修法：`setup_env.sh` 把 nvidia 各 lib 目录
   加入 `LD_LIBRARY_PATH`；为让 `run.py` 完全自包含，它会在启动时自动 re-exec 自己来设好该变量。
3. **编译器**：torch 2.4+ 头文件要 GCC 9+，而 nvcc 12.1 不认 GCC 13。`run.py` 会自动挑选
   `[9, 12]` 区间内的 gcc-toolset，并在 nvcc 上加 `-allow-unsupported-compiler` 放行。
   `setup_env.sh` 会顺手装好 `gcc-toolset-12-gcc-c++`。

> 注：本机默认 `python3` 是 3.6，torch 装在 `python3.9` 下，脚本统一用 `python3.9`。
