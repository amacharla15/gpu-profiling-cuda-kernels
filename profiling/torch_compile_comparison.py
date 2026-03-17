import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import benchmark_gpu, get_gpu_info, save_results


def benchmark_eager_vs_compiled(batch_size=64, dtype=torch.float32):
    device = "cuda"
    results = {}

    print("Benchmarking EAGER mode...")
    model_eager = models.resnet50(weights=None).to(device=device, dtype=dtype).eval()
    images = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

    with torch.no_grad():
        fn_eager = lambda: model_eager(images)
        results["eager"] = benchmark_gpu(fn_eager, num_warmup=5, num_runs=30)

    print("Compiling model with torch.compile (inductor backend)...")
    print("  First run will be SLOW (compilation happens here)...")

    model_compiled = models.resnet50(weights=None).to(device=device, dtype=dtype).eval()
    model_compiled = torch.compile(model_compiled, backend="cudagraphs")

    with torch.no_grad():
        print("  Warmup (includes compilation)...")
        for i in range(5):
            _ = model_compiled(images)
            torch.cuda.synchronize()
            if i == 0:
                print("  Compilation complete!")

        fn_compiled = lambda: model_compiled(images)
        print("  Benchmarking compiled model...")
        results["compiled"] = benchmark_gpu(fn_compiled, num_warmup=3, num_runs=30)

    return results


def profile_kernel_counts(batch_size=64, dtype=torch.float32):
    device = "cuda"
    images = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

    results = {}

    for mode_name, use_compile in [("eager", False), ("compiled", True)]:
        model = models.resnet50(weights=None).to(device=device, dtype=dtype).eval()

        if use_compile:
            model = torch.compile(model, backend="cudagraphs")

        with torch.no_grad():
            for _ in range(5):
                _ = model(images)
                torch.cuda.synchronize()

        with torch.no_grad():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
            ) as prof:
                _ = model(images)
                torch.cuda.synchronize()

        events = prof.key_averages(group_by_input_shape=False)
        cuda_events = [e for e in events if e.self_device_time_total > 0]

        total_launches = sum(e.count for e in cuda_events)
        unique_kernels = len(cuda_events)
        total_device_time = sum(e.self_device_time_total for e in cuda_events)

        top5 = sorted(cuda_events, key=lambda e: e.self_device_time_total, reverse=True)[:5]

        top5_info = []
        for e in top5:
            pct = e.self_device_time_total / total_device_time * 100 if total_device_time > 0 else 0
            top5_info.append({
                "name": e.key[:60],
                "calls": e.count,
                "device_time_us": round(e.self_device_time_total, 1),
                "pct": round(pct, 1),
            })

        results[mode_name] = {
            "total_kernel_launches": total_launches,
            "unique_kernels": unique_kernels,
            "total_device_time_us": round(total_device_time, 1),
            "top5_kernels": top5_info,
        }

    return results


if __name__ == "__main__":
    print("=" * 65)
    print("TORCH.COMPILE COMPARISON: Eager vs Compiled")
    print("=" * 65)

    info = get_gpu_info()
    print(f"GPU: {info['gpu_name']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print()

    print("PART 1: Latency Benchmark")
    print("-" * 65)

    timing_results = benchmark_eager_vs_compiled(batch_size=64)

    eager_ms = timing_results["eager"]["mean_ms"]
    compiled_ms = timing_results["compiled"]["mean_ms"]
    speedup = eager_ms / compiled_ms

    print(f"\n{'Mode':>12} {'Mean (ms)':>12} {'Std (ms)':>10} {'P95 (ms)':>10}")
    print("-" * 50)
    print(f"{'Eager':>12} {eager_ms:>12.3f} {timing_results['eager']['std_ms']:>10.3f} "
          f"{timing_results['eager']['p95_ms']:>10.3f}")
    print(f"{'Compiled':>12} {compiled_ms:>12.3f} {timing_results['compiled']['std_ms']:>10.3f} "
          f"{timing_results['compiled']['p95_ms']:>10.3f}")
    print(f"\n  Speedup: {speedup:.2f}x")

    print(f"\n\nPART 2: Kernel Analysis")
    print("-" * 65)

    kernel_results = profile_kernel_counts(batch_size=64)

    for mode in ["eager", "compiled"]:
        kr = kernel_results[mode]
        print(f"\n  {mode.upper()} mode:")
        print(f"    Total kernel launches: {kr['total_kernel_launches']}")
        print(f"    Unique kernel types:   {kr['unique_kernels']}")
        print(f"    Total device time:     {kr['total_device_time_us'] / 1000:.2f} ms")
        print(f"    Top 5 kernels:")
        for i, k in enumerate(kr["top5_kernels"]):
            print(f"      {i+1}. [{k['pct']:5.1f}%] {k['name']} (x{k['calls']})")

    eager_launches = kernel_results["eager"]["total_kernel_launches"]
    compiled_launches = kernel_results["compiled"]["total_kernel_launches"]

    print(f"\n\nSUMMARY")
    print("=" * 65)
    print(f"  Latency improvement:   {speedup:.2f}x ({eager_ms:.1f}ms → {compiled_ms:.1f}ms)")
    print(f"  Kernel launch reduction: {eager_launches} → {compiled_launches} "
          f"({(1 - compiled_launches/eager_launches)*100:.0f}% fewer)")

    all_results = {
        "timing": {k: {kk: vv for kk, vv in v.items() if kk != "all_times_ms"}
                   for k, v in timing_results.items()},
        "kernel_analysis": kernel_results,
        "summary": {
            "speedup": round(speedup, 2),
            "eager_launches": eager_launches,
            "compiled_launches": compiled_launches,
        }
    }

    output = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": info,
        "config": {"batch_size": 64, "dtype": "fp32"},
        "results": all_results,
    }

    os.makedirs("profiling/results", exist_ok=True)
    with open("profiling/results/torch_compile_comparison.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to profiling/results/torch_compile_comparison.json")