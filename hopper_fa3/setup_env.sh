#!/usr/bin/env bash
# =============================================================================
# hopper_fa3 一键环境配置（无 docker，新机器直接 bash setup_env.sh）
#
# 会做：系统依赖 -> 构建依赖 -> 装 torch(FA3 需要 >=2.2) -> 装官方 flash-attn(含 FA3)
#       -> 编译你的 CUDA 扩展 -> 跑一次对比
#
# 关于"装不上"的两个坑（本脚本已修复）：
#   1) pip 默认会建一个隔离环境，里面没有 torch，而 flash-attn 的 setup.py 一开头
#      就要 import torch -> 报 ModuleNotFoundError: No module named 'torch'。
#      解决：用 --no-build-isolation 复用系统已装的 torch。
#   2) FA3 在 flash-attn>=2.6 才引入，需要 torch>=2.2；而 pip 版 torch 把
#      cudnn/cublas 等拆到 nvidia-* 包，其 .so 不在链接器默认搜索路径 ->
#      import torch 报 libcudnn.so.9: cannot open shared object file。
#      解决：把 nvidia 各 lib 目录加入 LD_LIBRARY_PATH（本脚本与 run.py 都会设置）。
# =============================================================================
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 你的 torch 装在 python3.9 下，优先用它
if command -v python3.9 >/dev/null 2>&1; then PY=python3.9; else PY=python3; fi

echo "===== hopper_fa3 环境配置 ====="

# 1) 系统依赖：Python 开发头 + 编译器（含 GCC 9+，torch 2.4+ 头文件需要）+ git 及 bash 补全
echo "[1/5] 系统依赖..."
if   command -v apt-get >/dev/null 2>&1; then sudo apt-get install -y python3-dev gcc g++ make git bash-completion 2>/dev/null || apt-get install -y python3-dev gcc g++ make git bash-completion
elif command -v dnf     >/dev/null 2>&1; then sudo dnf install -y python39-devel gcc-c++ make git bash-completion 2>/dev/null || dnf install -y python39-devel gcc-c++ make git bash-completion
                                              dnf install -y gcc-toolset-12-gcc-c++ 2>/dev/null || true
elif command -v yum     >/dev/null 2>&1; then sudo yum install -y python39-devel gcc-c++ make git bash-completion 2>/dev/null || yum install -y python39-devel gcc-c++ make git bash-completion
                                              yum install -y gcc-toolset-12-gcc-c++ 2>/dev/null || true
fi

# 2) 构建依赖（ninja / packaging）
echo "[2/5] 构建依赖 (ninja / packaging)..."
${PY} -m pip install -r requirements.txt

# 3) 装一个带 FA3 的 torch（Hopper 上 flash-attn>=2.6 才含 FA3，需 torch>=2.2）
echo "[3/5] 安装 torch 2.4.0 (FA3 需要 torch>=2.2)..."
${PY} -m pip install "torch==2.4.0"

# 4) 把 nvidia 的 cudnn/cublas 等 .so 目录加入 LD_LIBRARY_PATH
#    （pip 版 torch 拆包，链接器默认找不到；flash-attn 编译时 import torch 需要它）
NVIDIA_LIBS="$(${PY} - <<'PY'
import glob, os, site
dirs=[]
for sp in site.getsitepackages():
    for d in glob.glob(os.path.join(sp,'nvidia','*')):
        lib=os.path.join(d,'lib')
        if os.path.isdir(lib): dirs.append(lib)
print(':'.join(dirs))
PY
)"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
echo "    LD_LIBRARY_PATH 已加入 nvidia lib 目录"

# 5) 官方 flash-attn（>=2.6 含 FA3），用 --no-build-isolation 复用系统 torch
echo "[4/5] 安装官方 flash-attn (>=2.6 含 FA3)..."
${PY} -m pip install --no-build-isolation "flash-attn==2.7.4.post1"

# 6) 编译你的 CUDA 扩展 + 跑一次对比
echo "[5/5] 编译扩展并跑 baseline..."
${PY} -c "import run; run._get_extension()" || echo "   [warn] 扩展编译失败，运行时将回退 SDPA。"
OFFICIAL=fa3 ${PY} run.py

echo ""
echo "===== 完成 ====="
echo "以后改 kernel 后只需:  ${PY} run.py"
echo "写真 attention 后加 --real 开启正确性对比:  ${PY} run.py --real"
echo "用官方 FA3 作为基准:  ${PY} run.py --official fa3   (没装上时自动回退 SDPA)"
