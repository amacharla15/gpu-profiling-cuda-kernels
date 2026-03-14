import torch
import time
import json
import os
from datetime import datetime


def benchmark_gpu(fn, num_warmup=5, num_runs=20, device="cuda"):
    """
    Benchmark a GPU function with warmup and timed runs.

    Args:
        fn: Function to run for one benchmark iteration.
        num_warmup: Number of warmup runs.
        num_runs: Number of timed runs.
        device: CUDA device name.

    Returns:
        Dictionary with timing statistics in milliseconds.
    """

    for _ in range(num_warmup):
        fn()

    torch.cuda.synchronize()

    times_ms = []

    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times_ms.append(elapsed * 1000)

    times_sorted = sorted(times_ms)
    n = len(times_sorted)

    mean = sum(times_ms) / n
    variance = sum((t - mean) ** 2 for t in times_ms) / n
    std = variance ** 0.5

    p50 = times_sorted[int(n * 0.50)]
    p95 = times_sorted[int(n * 0.95)]

    return {
        "mean_ms": round(mean, 4),
        "std_ms": round(std, 4),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "min_ms": round(times_sorted[0], 4),
        "max_ms": round(times_sorted[-1], 4),
        "num_runs": num_runs,
        "num_warmup": num_warmup,
        "all_times_ms": [round(t, 4) for t in times_ms]
    }


def get_gpu_info():
    """
    Get basic GPU and software information.
    """
    props = torch.cuda.get_device_properties("cuda")
    return {
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "vram_gb": round(props.total_memory / (1024 ** 3), 1),
        "num_sms": props.multi_processor_count,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def save_results(results, filename):
    """
    Save benchmark results to a JSON file.
    """
    output = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": get_gpu_info(),
        "results": results
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("=" * 60)
    print("BENCHMARK HARNESS TEST")
    print("=" * 60)
    print()

    info = get_gpu_info()
    for key, val in info.items():
        print(f"  {key}: {val}")
    print()

    device = torch.device("cuda")
    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)

    # Benchmark only the matrix multiply, not tensor creation.
    matmul_fn = lambda: torch.mm(a, b)

    print("Benchmarking 2000x2000 matrix multiply (FP32)...")
    result = benchmark_gpu(matmul_fn, num_warmup=5, num_runs=20)

    print(f"\n  Mean:   {result['mean_ms']:.3f} ms")
    print(f"  Std:    {result['std_ms']:.3f} ms")
    print(f"  P50:    {result['p50_ms']:.3f} ms")
    print(f"  P95:    {result['p95_ms']:.3f} ms")
    print(f"  Min:    {result['min_ms']:.3f} ms")
    print(f"  Max:    {result['max_ms']:.3f} ms")

    save_results(
        {"test_matmul_2000x2000_fp32": result},
        "profiling/results/harness_test.json"
    )

    print("\nHarness is working! Ready for real workloads.")
