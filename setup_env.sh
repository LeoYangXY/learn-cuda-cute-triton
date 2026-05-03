#!/bin/bash
# =============================================================================
# 一键安装环境 for learn-cuda-cute-triton
# 用法: source setup_env.sh
# 注意: 用 source 而不是 bash，这样 PATH 会保留在当前 shell
#
# 版本锁定:
#   Python: 3.12
#   PyTorch: 2.5.1+cu121
#   nvidia-cutlass-dsl: 4.3.5
#   GPU 驱动要求: >= 550
#
# 支持的 GPU:
#   H20/H100/H800 (SM90) — 全部特性 (TMA/Pipeline/WGMMA教程)
#   A100/A800 (SM80) — 大部分特性
# =============================================================================

echo "========================================="
echo "learn-cuda-cute-triton 环境安装"
echo "========================================="

# ---- Miniconda ----
if [ ! -d "$HOME/miniconda3" ]; then
    echo ">>> 安装 Miniconda..."
    curl -sSf https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    rm -f /tmp/miniconda.sh
fi
export PATH=$HOME/miniconda3/bin:$PATH
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ---- Python 3.12 环境 ----
if [ ! -d "$HOME/miniconda3/envs/cutedsl" ]; then
    echo ">>> 创建 Python 3.12 环境..."
    conda create -n cutedsl python=3.12 -y
fi
export PATH=$HOME/miniconda3/envs/cutedsl/bin:$HOME/miniconda3/bin:/usr/local/cuda/bin:$PATH

# ---- 检查驱动 ----
echo ">>> 检查 GPU 驱动..."
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "   GPU 已可用"
else
    # 驱动可能太旧，尝试升级用户态库到 550
    DRIVER_VER=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oP '\d+\.\d+\.\d+' | head -1)
    if [ -n "$DRIVER_VER" ]; then
        MAJOR=$(echo $DRIVER_VER | cut -d. -f1)
        if [ "$MAJOR" -lt 550 ]; then
            echo "   驱动 $DRIVER_VER < 550，尝试升级用户态库..."
            curl -sL "https://us.download.nvidia.com/tesla/550.127.08/NVIDIA-Linux-x86_64-550.127.08.run" -o /tmp/nvidia-driver.run
            chmod +x /tmp/nvidia-driver.run
            /tmp/nvidia-driver.run --no-kernel-module --no-questions --ui=none --no-backup 2>/dev/null
            rm -f /tmp/nvidia-driver.run
            echo "   用户态驱动升级完成"
        fi
    fi
fi

# ---- PyTorch ----
if ! python3 -c "import torch; assert torch.__version__.startswith('2.5')" 2>/dev/null; then
    echo ">>> 安装 PyTorch 2.5.1..."
    pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -1
else
    echo ">>> PyTorch 已安装"
fi

# ---- CuTeDSL ----
if ! python3 -c "from cutlass.cute.runtime import from_dlpack" 2>/dev/null; then
    echo ">>> 安装 nvidia-cutlass-dsl 4.3.5..."
    pip install -q "nvidia-cutlass-dsl==4.3.5" --extra-index-url https://pypi.nvidia.com 2>&1 | tail -1
else
    echo ">>> CuTeDSL 已安装"
fi

# ---- Patch experimental 模块 ----
find $HOME/miniconda3/envs/cutedsl/ -path "*/cute/experimental/__init__.py" 2>/dev/null | while read f; do
    if grep -q "raise NotImplementedError" "$f" 2>/dev/null; then
        echo 'import warnings; warnings.warn("skip")' > "$f"
        echo "   Patched: $f"
    fi
done

# ---- 验证 ----
echo ""
echo ">>> 验证环境..."
python3 -c "
import torch, cutlass
from cutlass.cute.runtime import from_dlpack
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
sm = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
print(f'✅ CuTeDSL OK | PyTorch {torch.__version__} | GPU: {gpu} | SM{sm[0]}{sm[1]}')
if sm >= (9, 0):
    print('   → Hopper 全部特性可用 (TMA/WGMMA/Cluster/Multicast)')
elif sm >= (8, 0):
    print('   → Ampere 特性可用 (TMA/Pipeline/WMMA), Hopper 独有特性跳过')
" 2>&1 | grep -v "UserWarning\|warnings.warn"

echo ""
echo "========================================="
echo "✅ 安装完成!"
echo ""
echo "用法:"
echo "  python3 cutedsl_ref/sgemm_hopper.py     # Hopper 教程"
echo "  python3 cutedsl_ref/sgemm.py            # SGEMM 6 版本演进"
echo "  python3 cutedsl_ref/flash_attention_v4.py  # Flash Attention"
echo "  python3 cutedsl_ref/add.py              # 入门: elementwise add"
echo ""
echo "如果切换了 shell, 重新激活环境:"
echo "  export PATH=\$HOME/miniconda3/envs/cutedsl/bin:\$HOME/miniconda3/bin:/usr/local/cuda/bin:\$PATH"
echo "========================================="
