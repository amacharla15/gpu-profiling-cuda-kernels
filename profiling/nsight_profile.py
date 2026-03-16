import torch
import torchvision.models as models
import subprocess
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from benchmark import get_gpu_info


def create_nsys_target_script():
    script = '''
import torch
import torchvision.models as models

device = "cuda"
dtype = torch.float32
batch_size = 64

model = models.resnet50(weights=None).to(device=device, dtype=dtype).eval()
images = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)

with torch.no_grad():
    for _ in range(3):
        _ = model(images)
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("resnet50_inference")
with torch.no_grad():
    for _ in range(5):
        _ = model(images)
        torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("nsys target script complete.")
'''
    path = "profiling/nsys_target.py"
    with open(path, "w") as f:
        f.write(script)
    return path


def create_ncu_target_script():
    script = '''
import torch
import torchvision.models as models

device = "cuda"
batch_size = 64

model = models.resnet50(weights=None).to(device=device).eval()
images = torch.randn(batch_size, 3, 224, 224, device=device)

with torch.no_grad():
    for _ in range(3):
        _ = model(images)
torch.cuda.synchronize()

with torch.no_grad():
    _ = model(images)
torch.cuda.synchronize()

print("ncu target script complete.")
'''
    path = "profiling/ncu_target.py"
    with open(path, "w") as f:
        f.write(script)
    return path


def run_nsys_profile():
    print("=" * 65)
    print("NSIGHT SYSTEMS — Full GPU Timeline")
    print("=" * 65)
    print()
    print("Running nsys on ResNet-50 (bs=64, FP32)...")
    print()

    script_path = create_nsys_target_script()
    output_name = "profiling/results/resnet50_nsys"

    cmd = [
        "nsys", "profile",
        "--stats=true",
        "--force-overwrite", "true",
        "-o", output_name,
        "python3", script_path
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        output = result.stdout + "\n" + result.stderr

        with open("profiling/results/nsys_output.txt", "w") as f:
            f.write(output)

        print("--- nsys output ---\n")

        lines = output.split("\n")
        in_kernel_section = False
        kernel_lines = []

        for line in lines:
            if "CUDA Kernel Statistics" in line or "cuda_gpu_kern_sum" in line:
                in_kernel_section = True
            if in_kernel_section:
                kernel_lines.append(line)
                if len(kernel_lines) > 25:
                    break

        if kernel_lines:
            for line in kernel_lines:
                print(line)
        else:
            for line in lines[-40:]:
                print(line)

        print(f"\nFull output saved to: profiling/results/nsys_output.txt")
        print(f"Timeline file: {output_name}.nsys-rep")

    except FileNotFoundError:
        print("ERROR: 'nsys' not found.")
    except subprocess.TimeoutExpired:
        print("WARNING: nsys timed out.")
    except Exception as e:
        print(f"ERROR: {e}")


def run_ncu_profile():
    print("\n")
    print("=" * 65)
    print("NSIGHT COMPUTE — Deep Kernel Analysis")
    print("=" * 65)
    print()

    script_path = create_ncu_target_script()

    cmd = [
        "ncu",
        "--set", "full",
        "--target-processes", "all",
        "--launch-skip", "100",
        "--launch-count", "10",
        "-o", "profiling/results/resnet50_ncu",
        "--force-overwrite",
        "python3", script_path
    ]

    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        output = result.stdout + "\n" + result.stderr

        with open("profiling/results/ncu_output.txt", "w") as f:
            f.write(output)

        print("--- ncu output ---\n")
        lines = output.strip().split("\n")

        for line in lines[-60:]:
            print(line)

        print(f"\nFull output saved to: profiling/results/ncu_output.txt")
        print(f"Report file: profiling/results/resnet50_ncu.ncu-rep")

    except FileNotFoundError:
        print("ERROR: 'ncu' not found.")
    except subprocess.TimeoutExpired:
        print("WARNING: ncu timed out.")
    except Exception as e:
        print(f"ERROR: {e}")


def parse_and_summarize():
    print("\n")
    print("=" * 65)
    print("PROFILING SUMMARY")
    print("=" * 65)

    results_dir = "profiling/results"

    nsys_exists = os.path.exists(f"{results_dir}/nsys_output.txt")
    ncu_exists = os.path.exists(f"{results_dir}/ncu_output.txt")
    profiler_exists = os.path.exists(f"{results_dir}/profiler_analysis.json")

    print(f"\n  Nsight Systems output:  {'YES' if nsys_exists else 'NO'}")
    print(f"  Nsight Compute output:  {'YES' if ncu_exists else 'NO'}")
    print(f"  PyTorch Profiler data:  {'YES' if profiler_exists else 'NO'}")


if __name__ == "__main__":
    info = get_gpu_info()
    print(f"GPU: {info['gpu_name']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print()

    nsys_available = os.system("which nsys > /dev/null 2>&1") == 0
    ncu_available = os.system("which ncu > /dev/null 2>&1") == 0

    print(f"nsys available: {nsys_available}")
    print(f"ncu available:  {ncu_available}")
    print()

    if nsys_available:
        run_nsys_profile()
    else:
        print("SKIPPING nsys")

    if ncu_available:
        run_ncu_profile()
    else:
        print("\nSKIPPING ncu")

    parse_and_summarize()