import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import get_gpu_info


def profile_resnet50(batch_size=64, dtype=torch.float32):
    device = "cuda"
    model = models.resnet50(weights=None).to(device=device, dtype=dtype).eval()
    images = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(5):
            _ = model(images)
    torch.cuda.synchronize()

    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            _ = model(images)
            torch.cuda.synchronize()

    return prof


def extract_top_kernels(prof, top_n=10):
    events = prof.key_averages(group_by_input_shape=False)

    events_sorted = sorted(
        events,
        key=lambda e: e.self_device_time_total,
        reverse=True
    )

    total_cuda_us = sum(e.self_device_time_total for e in events_sorted)

    results = []
    for e in events_sorted[:top_n]:
        if e.self_device_time_total == 0:
            continue

        pct = (e.self_device_time_total / total_cuda_us * 100) if total_cuda_us > 0 else 0

        results.append({
            "kernel_name": e.key,
            "calls": e.count,
            "total_cuda_us": round(e.self_device_time_total, 1),
            "avg_cuda_us": round(e.self_device_time_total / max(e.count, 1), 1),
            "pct_of_total": round(pct, 1),
            "cpu_time_us": round(e.self_cpu_time_total, 1),
        })

    return results, total_cuda_us


def extract_memory_info():
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def classify_kernel(kernel_name):
    name_lower = kernel_name.lower()

    compute_keywords = ["gemm", "conv", "wgrad", "dgrad", "implicit"]
    memory_keywords = ["batch_norm", "relu", "elementwise", "pool",
                       "add_", "copy", "fill", "cat", "adaptive"]

    for kw in compute_keywords:
        if kw in name_lower:
            return "compute-bound"
    for kw in memory_keywords:
        if kw in name_lower:
            return "memory-bound"
    return "unknown"


def run_analysis(batch_size=64, dtype=torch.float32):
    dtype_name = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16"
    }[dtype]

    print(f"\nProfiling ResNet-50: batch_size={batch_size}, dtype={dtype_name}")
    print("-" * 65)

    prof = profile_resnet50(batch_size=batch_size, dtype=dtype)
    top_kernels, total_cuda_us = extract_top_kernels(prof, top_n=10)

    print(f"\nTotal CUDA time: {total_cuda_us / 1000:.2f} ms")
    print(f"\nTop 10 CUDA Kernels:")
    print(f"{'#':>3} {'Kernel':>45} {'Calls':>6} {'Total(us)':>10} "
          f"{'%':>6} {'Type':>15}")
    print("-" * 95)

    compute_time = 0
    memory_time = 0

    for i, k in enumerate(top_kernels):
        classification = classify_kernel(k["kernel_name"])

        if classification == "compute-bound":
            compute_time += k["total_cuda_us"]
        elif classification == "memory-bound":
            memory_time += k["total_cuda_us"]

        name = k["kernel_name"]
        if len(name) > 45:
            name = name[:42] + "..."

        print(f"{i+1:>3} {name:>45} {k['calls']:>6} "
              f"{k['total_cuda_us']:>10.1f} {k['pct_of_total']:>5.1f}% "
              f"{classification:>15}")

    print(f"\nBottleneck Classification:")

    classified_total = compute_time + memory_time
    if classified_total > 0:
        compute_pct = compute_time / classified_total * 100
        memory_pct = memory_time / classified_total * 100
        print(f"  Compute-bound kernels: {compute_pct:.1f}% of classified time")
        print(f"  Memory-bound kernels:  {memory_pct:.1f}% of classified time")

        if compute_pct > memory_pct:
            print(f"  → This workload is COMPUTE-BOUND at bs={batch_size}")
            print(f"    (FP16/BF16 will help — Tensor Cores speed up the GEMM/conv)")
        else:
            print(f"  → This workload is MEMORY-BOUND at bs={batch_size}")
            print(f"    (Bigger batch or FP16 helps by reducing memory traffic)")

    mem = extract_memory_info()
    print(f"\nGPU Memory:")
    print(f"  Allocated: {mem['allocated_mb']:.1f} MB")
    print(f"  Reserved:  {mem['reserved_mb']:.1f} MB")
    print(f"  Peak:      {mem['max_allocated_mb']:.1f} MB")

    return {
        "batch_size": batch_size,
        "dtype": dtype_name,
        "total_cuda_ms": round(total_cuda_us / 1000, 2),
        "top_kernels": top_kernels,
        "bottleneck": {
            "compute_pct": round(compute_time / max(classified_total, 1) * 100, 1),
            "memory_pct": round(memory_time / max(classified_total, 1) * 100, 1),
        },
        "memory": mem,
    }


if __name__ == "__main__":
    print("=" * 65)
    print("PYTORCH PROFILER ANALYSIS")
    print("=" * 65)

    info = get_gpu_info()
    print(f"GPU: {info['gpu_name']}")
    print()

    all_results = {}

    for bs in [1, 64]:
        result = run_analysis(batch_size=bs, dtype=torch.float32)
        all_results[f"bs{bs}_fp32"] = result

    result = run_analysis(batch_size=64, dtype=torch.float16)
    all_results["bs64_fp16"] = result

    output = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": info,
        "results": all_results,
    }
    os.makedirs("profiling/results", exist_ok=True)
    with open("profiling/results/profiler_analysis.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 65)
    print("Results saved to profiling/results/profiler_analysis.json")
    print("=" * 65)
