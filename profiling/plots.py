import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os

plt.style.use("dark_background")

COLORS = {
    "fp32": "#4CAF50",
    "fp16": "#2196F3",
    "bf16": "#FF9800",
}

def load_sweep_data():
    path = "profiling/results/resnet50_sweep.json"
    with open(path) as f:
        data = json.load(f)
    return data["results"]

def plot_latency_vs_batch(results, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    batch_sizes = [1, 4, 16, 64, 128]

    for dtype_name, color in COLORS.items():
        latencies = []
        p95s = []
        valid_bs = []

        for bs in batch_sizes:
            key = f"resnet50_bs{bs}_{dtype_name}"
            if key in results:
                latencies.append(results[key]["mean_ms"])
                p95s.append(results[key]["p95_ms"])
                valid_bs.append(bs)

        ax.plot(valid_bs, latencies, "o-", color=color, linewidth=2, markersize=8, label=f"{dtype_name.upper()} (mean)")
        ax.plot(valid_bs, p95s, "s--", color=color, linewidth=1, markersize=5, alpha=0.5, label=f"{dtype_name.upper()} (p95)")

    ax.set_xlabel("Batch Size", fontsize=13)
    ax.set_ylabel("Latency (ms)", fontsize=13)
    ax.set_title("ResNet-50 Latency vs Batch Size — A100 80GB", fontsize=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])

    plt.tight_layout()
    filepath = os.path.join(output_dir, "latency_vs_batch.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")

def plot_throughput_vs_batch(results, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    batch_sizes = [1, 4, 16, 64, 128]

    for dtype_name, color in COLORS.items():
        throughputs = []
        valid_bs = []

        for bs in batch_sizes:
            key = f"resnet50_bs{bs}_{dtype_name}"
            if key in results:
                throughputs.append(results[key]["throughput_imgs_per_sec"])
                valid_bs.append(bs)

        ax.plot(valid_bs, throughputs, "o-", color=color, linewidth=2.5, markersize=10, label=f"{dtype_name.upper()}")

        if throughputs:
            peak_idx = throughputs.index(max(throughputs))
            ax.annotate(
                f"{throughputs[peak_idx]:.0f}",
                (valid_bs[peak_idx], throughputs[peak_idx]),
                textcoords="offset points",
                xytext=(0, 15),
                fontsize=10,
                color=color,
                fontweight="bold",
                ha="center",
            )

    ax.set_xlabel("Batch Size", fontsize=13)
    ax.set_ylabel("Throughput (images/sec)", fontsize=13)
    ax.set_title("ResNet-50 Throughput vs Batch Size — A100 80GB", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    ax.set_xscale("log", base=2)
    ax.set_xticks(batch_sizes)
    ax.set_xticklabels([str(b) for b in batch_sizes])

    plt.tight_layout()
    filepath = os.path.join(output_dir, "throughput_vs_batch.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")

def plot_bottleneck_shift(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    path = "profiling/results/profiler_analysis.json"
    with open(path) as f:
        data = json.load(f)

    profiler_results = data["results"]

    configs = []
    compute_pcts = []
    memory_pcts = []

    for key, label in [
        ("bs1_fp32", "BS=1\nFP32"),
        ("bs64_fp32", "BS=64\nFP32"),
        ("bs64_fp16", "BS=64\nFP16"),
    ]:
        if key in profiler_results:
            configs.append(label)
            b = profiler_results[key]["bottleneck"]
            compute_pcts.append(b["compute_pct"])
            memory_pcts.append(b["memory_pct"])

    if not configs:
        print("No profiler data found, skipping bottleneck plot")
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    y_pos = range(len(configs))

    ax.barh(y_pos, compute_pcts, height=0.5, color="#FF6B6B", label="Compute-bound")
    ax.barh(y_pos, memory_pcts, height=0.5, left=compute_pcts, color="#4ECDC4", label="Memory-bound")

    for i in range(len(configs)):
        if compute_pcts[i] > 10:
            ax.text(compute_pcts[i] / 2, i, f"{compute_pcts[i]:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold")
        if memory_pcts[i] > 10:
            ax.text(compute_pcts[i] + memory_pcts[i] / 2, i, f"{memory_pcts[i]:.0f}%", ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(configs, fontsize=12)
    ax.set_xlabel("% of Classified Kernel Time", fontsize=13)
    ax.set_title("Bottleneck Shift: Compute → Memory as FP16 Speeds Up Math", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    filepath = os.path.join(output_dir, "bottleneck_shift.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    print()

    results = load_sweep_data()

    plot_latency_vs_batch(results)
    plot_throughput_vs_batch(results)
    plot_bottleneck_shift()

    print()
    print("All plots saved to plots/ directory.")
    print("Download them with: scp cscigpu:~/gpu-profiling-cuda-kernels/plots/*.png .")
