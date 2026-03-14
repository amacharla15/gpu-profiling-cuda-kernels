# GPU Profiling & CUDA Kernel Optimization

GPU performance profiling suite and custom CUDA kernels benchmarked on an NVIDIA A100 80GB PCIe.

**Author:** Akshith Macharla | M.S. Computer Science, CSU Chico

## Overview

**Part A — GPU Profiling Harness (Python)**
- Benchmarked ResNet-50, convolution, and GEMM workloads across batch sizes (1–128) and dtypes (FP32/FP16/BF16)
- GPU-correct methodology: warmup, CUDA synchronization, statistical reporting (mean, std, p50, p95)
- PyTorch Profiler integration with kernel-level timing and memory analysis
- Roofline analysis classifying workloads as compute-bound vs memory-bound

**Part B — Custom CUDA C++ Kernels**
- Tiled matrix multiplication with shared memory optimization
- 2D convolution kernel for image processing
- All kernels profiled with Nsight Compute — occupancy, memory throughput, roofline position

**Part C — cuBLAS vs Custom Kernel Analysis**
- Benchmarked hand-written CUDA matmul against PyTorch's `torch.mm` (cuBLAS) on identical workloads
- Quantified the performance gap and analyzed when custom kernels are justified vs library calls

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
├── cuda/                # CUDA C++ kernels (.cu files)
├── plots/               # Benchmark charts and roofline figures
└── smoke_test.py        # GPU environment verification
```

## Quick Start
```bash
source ~/gpu-env/bin/activate
python3 smoke_test.py              # verify GPU access
python3 profiling/benchmark.py     # run benchmarks
```

---

*Part of a 6-project AI inference engineering portfolio. See also: [CPU Inference Server](https://github.com/amacharla15/CPUinference)*
