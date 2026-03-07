import torch
import triton

DEVICE = torch.device("cuda:0")


def _benchmark_ms(fn, num_warmup, num_iter, use_do_bench=False):
    if use_do_bench:
        return triton.testing.do_bench(fn, warmup=num_warmup, rep=num_iter)

    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / num_iter


def _default_prepare_data(size, device, dtype):
    x = torch.randn(size, device=device, dtype=dtype)
    y = torch.randn(size, device=device, dtype=dtype)
    out = torch.empty_like(x)
    return {"x": x, "y": y, "out": out}


def auto_tune_and_benchmark(
    triton_launch,
    torch_launch,
    sizes,
    block_sizes=None,
    num_warmup_tune=5,
    num_iter_tune=20,
    num_warmup_bench=10,
    num_iter_bench=100,
    dtype=torch.float32,
    device=DEVICE,
    correctness_size=98432,
    prepare_data_fn=None,
    triton_output_getter=None,
    atol=1e-6,
    use_do_bench=False,
):
    """
    1) 正确性检查（Triton kernel output vs PyTorch）
    2) 按 BLOCK_SIZE 自动调优（仅测 kernel launch）
    3) 用最优 BLOCK_SIZE 做 Triton vs PyTorch benchmark

    参数约定：
    - prepare_data_fn(size, device, dtype) -> dict
    - triton_launch(data, BLOCK_SIZE=...): 只 launch kernel，结果写入 data
    - torch_launch(data) -> torch.Tensor: PyTorch 参考输出
    - triton_output_getter(data) -> torch.Tensor: 默认取 data["out"]
    """
    if prepare_data_fn is None:
        prepare_data_fn = _default_prepare_data
    if triton_output_getter is None:
        triton_output_getter = lambda data: data["out"]

    if block_sizes is None:
        block_sizes = [64, 128, 256, 512, 1024]

    # ======================
    # 正确性检查
    # ======================
    print("🔍 Running correctness check...")
    data_test = prepare_data_fn(correctness_size, device, dtype)
    candidate_bs = block_sizes[0]
    triton_launch(data_test, BLOCK_SIZE=candidate_bs)
    triton_out = triton_output_getter(data_test)
    torch_out = torch_launch(data_test)

    if not torch.allclose(triton_out, torch_out, atol=atol):
        max_diff = torch.max(torch.abs(triton_out - torch_out)).item()
        raise RuntimeError(f"❌ Correctness check failed! Max diff: {max_diff}")
    print("✅ Correctness verified.\n")

    # ======================
    # 自动调优 + Benchmark
    # ======================
    print(f"{'Size':>10} | {'Best BS':>8} | {'Triton (ms)':>12} | {'Torch (ms)':>12} | {'Speedup':>8}")
    print("-" * 70)

    for size in sizes:
        data = prepare_data_fn(size, device, dtype)

        best_bs = block_sizes[0]
        best_time = float("inf")

        for bs in block_sizes:
            if bs <= 0:
                continue

            avg_ms = _benchmark_ms(
                lambda: triton_launch(data, BLOCK_SIZE=bs),
                num_warmup=num_warmup_tune,
                num_iter=num_iter_tune,
                use_do_bench=use_do_bench,
            )

            if avg_ms < best_time:
                best_time = avg_ms
                best_bs = bs

        triton_ms = _benchmark_ms(
            lambda: triton_launch(data, BLOCK_SIZE=best_bs),
            num_warmup=num_warmup_bench,
            num_iter=num_iter_bench,
            use_do_bench=use_do_bench,
        )

        torch_ms = _benchmark_ms(
            lambda: torch_launch(data),
            num_warmup=num_warmup_bench,
            num_iter=num_iter_bench,
            use_do_bench=use_do_bench,
        )

        speedup = torch_ms / triton_ms if triton_ms > 0 else float("inf")
        print(f"{size:>10} | {best_bs:>8} | {triton_ms:>12.3f} | {torch_ms:>12.3f} | {speedup:>8.2f}x")
