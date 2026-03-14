"""
Smoke test: Verify GPU access, PyTorch CUDA, and basic timing methodology.
Run this FIRST before building anything else.
"""
import torch
import time

def main():
    # ---- Device check ----
    print("=" * 60)
    print("GPU SMOKE TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available.")
        return

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)

    print(f"GPU:              {props.name}")
    print(f"Compute cap:      {props.major}.{props.minor}")
    print(f"VRAM:             {props.total_memory / (1024**3):.1f} GB")
    print(f"SMs:              {props.multi_processor_count}")
    print(f"PyTorch:          {torch.__version__}")
    print(f"CUDA (PyTorch):   {torch.version.cuda}")
    print()

    # ---- Basic GPU operation ----
    print("Running basic GPU operation...")
    a = torch.randn(1000, 1000, device=device)
    b = torch.randn(1000, 1000, device=device)
    c = torch.mm(a, b)
    print(f"Matrix multiply: {list(a.shape)} x {list(b.shape)} = {list(c.shape)}")
    print(f"Result sample (top-left 3x3):\n{c[:3, :3]}")
    print()

    # ---- GPU timing: WRONG vs RIGHT ----
    # KEY CONCEPT: GPU operations are ASYNCHRONOUS.
    # When you call torch.mm(), Python gets control back IMMEDIATELY
    # while the GPU is still computing. So time.perf_counter() without
    # synchronize() measures how fast Python can LAUNCH the work,
    # not how long the GPU takes to FINISH it.
    
    print("Timing demo: WRONG way vs RIGHT way")
    print("-" * 40)

    # WRONG: no sync — measures launch overhead only
    start = time.perf_counter()
    _ = torch.mm(a, b)
    wrong_time = time.perf_counter() - start
    print(f"WRONG (no sync):   {wrong_time*1000:.3f} ms  <-- just launch overhead")

    # RIGHT: synchronize before and after
    torch.cuda.synchronize()
    start = time.perf_counter()
    _ = torch.mm(a, b)
    torch.cuda.synchronize()
    right_time = time.perf_counter() - start
    print(f"RIGHT (with sync): {right_time*1000:.3f} ms  <-- actual GPU time")
    print()

    # ---- Warmup demo ----
    # FIRST GPU operation is slow because CUDA must:
    #   1. Load the kernel binary from disk into GPU
    #   2. Allocate GPU memory for the first time
    #   3. Initialize internal caches
    # This is called "cold start". After 2-3 runs, times stabilize.
    # In our profiling harness, we'll always do warmup runs and
    # exclude them from measurements.

    print("Warmup demo: first run vs warmed-up runs")
    print("-" * 40)

    x = torch.randn(2000, 2000, device=device)
    y = torch.randn(2000, 2000, device=device)

    times = []
    for i in range(7):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = torch.mm(x, y)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
        label = " <-- cold start" if i == 0 else ""
        print(f"  Run {i}: {elapsed*1000:.3f} ms{label}")

    warm_times = times[1:]
    mean_ms = sum(warm_times) / len(warm_times)
    print(f"\n  Mean (excluding warmup): {mean_ms:.3f} ms")
    print()

    # ---- Dtype support check ----
    # A100 supports all three. FP16 and BF16 use Tensor Cores,
    # which are specialized hardware for matrix math — up to
    # 16x faster than FP32 for the same operation.
    # We'll measure this precisely in Phase 2.

    print("Dtype support check:")
    print("-" * 40)
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        try:
            t = torch.randn(100, 100, device=device, dtype=dtype)
            r = torch.mm(t, t)
            print(f"  {str(dtype):20s} OK")
        except Exception as e:
            print(f"  {str(dtype):20s} FAIL: {e}")

    print()
    print("=" * 60)
    print("SMOKE TEST PASSED - ready for Phase 2!")
    print("=" * 60)

if __name__ == "__main__":
    main()
