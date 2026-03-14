# GPU Profiling & CUDA Kernel Optimization

GPU performance profiling suite and hand-written CUDA kernels, built on an NVIDIA A100 80GB PCIe.

**Author:** Akshith Macharla | M.S. Computer Science, CSU Chico

## What This Project Covers

**Part A — Python GPU Profiling Harness**
- Benchmark CV workloads (ResNet-50, convolutions, GEMM) across batch sizes and dtypes (FP32/FP16/BF16)
- Proper GPU benchmarking: warmup, CUDA synchronization, statistical reporting (mean, std, p50, p95)
- PyTorch Profiler integration with kernel-level timing
- Roofline analysis: classify workloads as compute-bound vs memory-bound

**Part B — CUDA C++ Programs**
- Vector addition — first `.cu` file with `cudaMalloc`, `cudaMemcpy`, kernel launch
- Matrix multiplication — naive vs shared-memory tiled (benchmarked)
- Image processing kernel — 2D convolution on GPU
- Nsight Compute profiling with roofline analysis

**Part C — PyTorch vs Hand-Written CUDA**
- Compare hand-written matmul against `torch.mm` (cuBLAS)
- Analysis of when custom kernels make sense vs using libraries

## Hardware

| Spec | Value |
|------|-------|
| GPU | NVIDIA A100 80GB PCIe |
| Compute Capability | 8.0 |
| SMs | 108 |
| VRAM | 79.2 GB HBM2e |
| CUDA Toolkit | 12.9 |
| PyTorch | 2.10.0+cu128 |

## Project Structure
```
gpu-profiling-cuda-kernels/
├── profiling/           # Python benchmarking harness
│   ├── benchmark.py     # Core timing infrastructure
│   └── results/         # JSON benchmark outputs
├── cuda/                # CUDA C++ exercises (.cu files)
├── plots/               # Generated charts and figures
├── smoke_test.py        # GPU environment verification
└── README.md
```

## Quick Start
```bash
# SSH into GPU machine and activate environment
source ~/gpu-env/bin/activate

# Run smoke test
python3 smoke_test.py

# Run benchmark harness test
python3 profiling/benchmark.py
```

## Status

- [x] Phase 1: Environment setup & smoke test
- [ ] Phase 2: Core benchmarking harness with ResNet-50
- [ ] Phase 3: Batch size & dtype sweep
- [ ] Phase 4: PyTorch Profiler integration
- [ ] Phase 5: Roofline analysis & plots
- [ ] Phase 6: CUDA Exercise 1 — Vector addition
- [ ] Phase 7: CUDA Exercise 2 — Matrix multiplication
- [ ] Phase 8: CUDA Exercise 3 — Image processing kernel
- [ ] Phase 9: CUDA Exercise 4 — Nsight Compute profiling
- [ ] Phase 10: PyTorch vs CUDA comparison & README polish

---

*Part of a 6-project AI inference engineering portfolio. See also: [CPU Inference Server](https://github.com/amacharla15/CPUinference)
