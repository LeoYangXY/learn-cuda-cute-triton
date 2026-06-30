"""FA3 baseline：用同一份输入，对比你的 CUDA kernel 和官方 FA3 / torch SDPA。

用法:
  python3 run.py                 # 默认 dummy 模式（只跑通流程，不比正确性）
  python3 run.py --real          # 当你在 csrc 里实现了真 attention，开启正确性对比
  OFFICIAL=fa3 python3 run.py    # 强制用官方 FA3 作为对比基准
"""
import argparse
import glob
import importlib
import os
import shutil
import site
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _ensure_nvidia_libs():
    """pip 版 torch 把 cudnn/cublas 等拆到 nvidia-* 包，其 .so 不在链接器默认
    搜索路径。必须在 python 启动前把它们的 lib 目录加入 LD_LIBRARY_PATH；
    若缺失则 re-exec 自身（新进程启动时链接器才能读到）。"""
    dirs = []
    for sp in site.getsitepackages():
        for d in sorted(glob.glob(os.path.join(sp, "nvidia", "*"))):
            lib = os.path.join(d, "lib")
            if os.path.isdir(lib):
                dirs.append(lib)
    if not dirs:
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    cur_set = set(cur.split(":")) if cur else set()
    if all(d in cur_set for d in dirs):
        return
    os.environ["LD_LIBRARY_PATH"] = ":".join(dirs) + (":" + cur if cur else "")
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_nvidia_libs()


def _pick_cxx():
    """torch 2.4+ 头文件要求 GCC 9+；nvcc 12.1 不认 GCC 13。
    挑一个主版本落在 [9, 12] 区间内的编译器，写进 CXX/CC
    （torch 的 cpp_extension 会读这个环境变量；nvcc 用 -allow-unsupported-compiler 放行）。"""
    def ver(cxx):
        try:
            out = subprocess.run([cxx, "-dumpversion"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
            parts = [int(x) for x in out.split(".")[:2]]
            return tuple(parts + [0] * (2 - len(parts)))
        except Exception:
            return (0, 0)

    def ok(v):
        return (9, 0) <= v <= (12, 99)

    cur = os.environ.get("CXX", "g++")
    if ok(ver(cur)):
        return
    # 1) gcc-toolset：选最高且主版本 <= 12 的
    for cand in sorted(glob.glob("/opt/rh/gcc-toolset-*/root/usr/bin/c++"), reverse=True):
        if ok(ver(cand)):
            os.environ["CXX"] = cand
            os.environ["CC"] = cand.replace("/c++", "/gcc")
            return
    # 2) 常见命名 g++-N
    for name in ("g++-12", "g++-11", "g++-10", "g++-9"):
        p = shutil.which(name)
        if p and ok(ver(p)):
            os.environ["CXX"] = p
            os.environ["CC"] = p.replace("g++", "gcc")
            return
    # 3) 系统默认 g++，若恰好落在 [9, 12] 区间内则直接用
    if ok(ver("g++")):
        return


_pick_cxx()

import torch
from torch.utils.cpp_extension import load

SRC = Path(__file__).resolve().parent / "csrc" / "my_fa3_kernel.cu"
BUILD_DIR = Path(__file__).resolve().parent / "build"
MODULE_NAME = "hopper_fa3_my_kernel"

_MY_EXT = None
_MY_BACKEND = "python_sdpa_fallback"
_BUILD_TRIED = False


@lru_cache(maxsize=1)
def _get_extension():
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    return load(
        name=MODULE_NAME,
        sources=[str(SRC)],
        extra_cuda_cflags=[
            "-O3", "-std=c++17", "--use_fast_math",
            "-allow-unsupported-compiler",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
        ],
        extra_cflags=["-O3", "-std=c++17"],
        build_directory=str(BUILD_DIR),
    )


def my_kernel(q, k, v, softmax_scale, causal, dummy=False):
    """调用你的 CUDA 扩展；编译失败则回退到 SDPA。"""
    global _MY_EXT, _MY_BACKEND, _BUILD_TRIED
    if _MY_EXT is None and not _BUILD_TRIED:
        _BUILD_TRIED = True
        try:
            _MY_EXT = _get_extension()
            _MY_BACKEND = "cuda_pybind_extension"
        except Exception as e:
            print(f"[WARN] 你的 CUDA 扩展编译失败，回退 SDPA: {e}")
            _MY_BACKEND = "python_sdpa_fallback"

    if _MY_EXT is not None:
        return _MY_EXT.my_fa3_forward(q, k, v, float(softmax_scale), bool(causal))
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=softmax_scale, is_causal=causal
    )


def official_impl():
    """优先官方 flash-attn (FA3)；Hopper 上 flash_attn_func 自动 dispatch 到 FA3 内核。
    装不上时回退 torch SDPA。"""
    try:
        from flash_attn import flash_attn_func
        return "flash_attn_fa3", lambda q, k, v, s, c: flash_attn_func(
            q, k, v, dropout_p=0.0, softmax_scale=s, causal=c
        )
    except Exception:
        pass
    return "torch_sdpa", lambda q, k, v, s, c: (
        torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=s, is_causal=c)
    )


def _bench(fn, q, k, v, scale, causal, warmup, iters):
    for _ in range(warmup):
        fn(q, k, v, scale, causal)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(q, k, v, scale, causal)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--official", default=os.environ.get("OFFICIAL", "auto"),
                    choices=["auto", "fa3", "sdpa"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--real", action="store_true",
                    help="csrc 里已实现真 attention 时，开启正确性对比")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    q = torch.randn(args.batch, args.heads, args.seqlen, args.head_dim, device="cuda", dtype=dtype).contiguous()
    k = torch.randn_like(q).contiguous()
    v = torch.randn_like(q).contiguous()
    scale = args.head_dim ** -0.5

    pref = "sdpa" if args.official == "sdpa" else "fa3"
    if pref == "sdpa":
        official_name, official_fn = "torch_sdpa", lambda q, k, v, s, c: (
            torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=s, is_causal=c)
        )
    else:
        official_name, official_fn = official_impl()

    out_off = official_fn(q, k, v, scale, args.causal)
    out_mine = my_kernel(q, k, v, scale, args.causal)

    max_abs = (out_mine - out_off).abs().max().item()
    ok = torch.allclose(out_mine, out_off, atol=2e-2, rtol=2e-2)

    flops = 4.0 * args.batch * args.heads * args.seqlen * args.seqlen * args.head_dim
    if args.causal:
        flops *= 0.5
    ms_off = _bench(official_fn, q, k, v, scale, args.causal, args.warmup, args.iters)
    ms_mine = _bench(lambda q, k, v, s, c: my_kernel(q, k, v, s, c), q, k, v, scale, args.causal, args.warmup, args.iters)

    print("=" * 72)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"shape=(B={args.batch}, H={args.heads}, N={args.seqlen}, D={args.head_dim}) "
          f"dtype={args.dtype} causal={args.causal}")
    print(f"official_backend={official_name}  |  my_kernel_backend={_MY_BACKEND}")
    print("-" * 72)
    if args.real:
        print(f"correctness: {'PASS' if ok else 'FAIL'} | max_abs_diff={max_abs:.6f}")
    else:
        print(f"correctness: SKIP (dummy) | max_abs_diff={max_abs:.6f}  "
              f"(写真 attention 后加 --real 开启对比)")
    print("-" * 72)
    print(f"{official_name:>16} | {ms_off:8.3f} ms | {flops / (ms_off / 1000) / 1e12:8.2f} TFLOPS")
    print(f"{'my_kernel':>16} | {ms_mine:8.3f} ms | {flops / (ms_mine / 1000) / 1e12:8.2f} TFLOPS")
    print("-" * 72)
    print(f"speedup(my_kernel vs {official_name}) = {ms_off / ms_mine:.3f}x")
    print("=" * 72)

    if args.real and not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
