import torch
import torchvision.models as models
from benchmark import benchmark_gpu, get_gpu_info, save_results


def make_resnet50(dtype=torch.float32, device="cuda"):
    """
    Create a ResNet-50 model and move it to the target device.
    """
    model = models.resnet50(weights=None)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def make_input(batch_size, dtype=torch.float32, device="cuda"):
    """
    Create a random input tensor with ImageNet-style shape.
    """
    return torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)


def benchmark_resnet50(batch_size=1, dtype=torch.float32):
    """
    Run one ResNet-50 benchmark configuration and return the results.
    """
    device = "cuda"
    model = make_resnet50(dtype=dtype, device=device)
    images = make_input(batch_size, dtype=dtype, device=device)

    with torch.no_grad():
        fn = lambda: model(images)
        result = benchmark_gpu(fn, num_warmup=5, num_runs=30)

    result["batch_size"] = batch_size
    result["dtype"] = str(dtype)
    result["throughput_imgs_per_sec"] = round(
        batch_size * 1000 / result["mean_ms"], 1
    )

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("RESNET-50 BENCHMARK")
    print("=" * 60)

    # Basic environment info
    info = get_gpu_info()
    print(f"GPU: {info['gpu_name']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print()

    # Single test configuration
    batch_size = 1
    dtype = torch.float32

    print(f"Config: batch_size={batch_size}, dtype={dtype}")
    print("Running benchmark (5 warmup + 30 timed runs)...")
    print()

    result = benchmark_resnet50(batch_size=batch_size, dtype=dtype)

    # Print summary
    print("Results:")
    print(f"  Latency (mean):  {result['mean_ms']:.3f} ms")
    print(f"  Latency (std):   {result['std_ms']:.3f} ms")
    print(f"  Latency (p50):   {result['p50_ms']:.3f} ms")
    print(f"  Latency (p95):   {result['p95_ms']:.3f} ms")
    print(f"  Throughput:      {result['throughput_imgs_per_sec']:.1f} images/sec")
    print()

    # Save benchmark output
    save_results(
        {"resnet50_bs1_fp32": result},
        "profiling/results/resnet50_single.json"
    )

    print("Done! Next step: sweep across batch sizes and dtypes.")
