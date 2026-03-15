import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from benchmark import benchmark_gpu, get_gpu_info, save_results
from workloads import make_resnet50, make_input


def run_sweep():
    device = "cuda"

    batch_sizes = [1, 4, 16, 64, 128]
    dtypes = [
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ]

    all_results = {}

    print(f"{'Batch':>6} {'Dtype':>6} {'Latency(ms)':>12} {'Std(ms)':>10} "
          f"{'P95(ms)':>10} {'Throughput':>12} {'imgs/sec':>10}")
    print("-" * 78)

    for dtype_name, dtype in dtypes:
        model = make_resnet50(dtype=dtype, device=device)

        for bs in batch_sizes:
            try:
                images = make_input(bs, dtype=dtype, device=device)

                with torch.no_grad():
                    fn = lambda: model(images)
                    result = benchmark_gpu(fn, num_warmup=5, num_runs=30)

                throughput = round(bs * 1000 / result["mean_ms"], 1)
                result["batch_size"] = bs
                result["dtype"] = dtype_name
                result["throughput_imgs_per_sec"] = throughput

                key = f"resnet50_bs{bs}_{dtype_name}"
                all_results[key] = result

                print(f"{bs:>6} {dtype_name:>6} {result['mean_ms']:>12.3f} "
                      f"{result['std_ms']:>10.3f} {result['p95_ms']:>10.3f} "
                      f"{'':>12} {throughput:>10.1f}")

                del images
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"{bs:>6} {dtype_name:>6} {'OOM':>12}")
                torch.cuda.empty_cache()

    return all_results


def print_summary(results):
    print("\n")
    print("=" * 70)
    print("SUMMARY: Key Findings")
    print("=" * 70)

    print("\n1) Throughput scaling with batch size (FP32):")
    print(f"   {'Batch Size':>10} {'Throughput (img/s)':>20} {'vs BS=1':>10}")

    bs1_throughput = None
    for bs in [1, 4, 16, 64, 128]:
        key = f"resnet50_bs{bs}_fp32"
        if key in results:
            tp = results[key]["throughput_imgs_per_sec"]
            if bs == 1:
                bs1_throughput = tp
            speedup = f"{tp / bs1_throughput:.1f}x" if bs1_throughput else "—"
            print(f"   {bs:>10} {tp:>20.1f} {speedup:>10}")

    print(f"\n2) Dtype effect at batch_size=64:")
    print(f"   {'Dtype':>10} {'Latency (ms)':>15} {'Throughput':>15} {'vs FP32':>10}")

    fp32_latency = None
    for dtype_name in ["fp32", "fp16", "bf16"]:
        key = f"resnet50_bs64_{dtype_name}"
        if key in results:
            lat = results[key]["mean_ms"]
            tp = results[key]["throughput_imgs_per_sec"]
            if dtype_name == "fp32":
                fp32_latency = lat
            speedup = f"{fp32_latency / lat:.2f}x" if fp32_latency else "—"
            print(f"   {dtype_name:>10} {lat:>15.3f} {tp:>15.1f} {speedup:>10}")


if __name__ == "__main__":
    print("=" * 70)
    print("RESNET-50 SWEEP: Batch Sizes x Dtypes")
    print("=" * 70)

    info = get_gpu_info()
    print(f"GPU: {info['gpu_name']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print()

    results = run_sweep()

    print_summary(results)

    save_results(results, "profiling/results/resnet50_sweep.json")

    print("\nSweep complete! Next: add convolution and GEMM workloads.")
