import argparse
import csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise SystemExit("matplotlib/numpy not installed. Run: pip install matplotlib numpy")

DARK_BG = "#0c0f0f"
AX_BG = "#121414"
TEXT = "#e2e2e2"
MUTED = "#c4c7c7"
OUTLINE = "#444748"
GRID = "#2a2d2e"


def apply_dark_style(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(OUTLINE)
    legend = ax.get_legend()
    if legend:
        frame = legend.get_frame()
        frame.set_facecolor(AX_BG)
        frame.set_edgecolor(OUTLINE)
        for text in legend.get_texts():
            text.set_color(TEXT)


def finalize_figure(fig, ax):
    fig.patch.set_facecolor(DARK_BG)
    apply_dark_style(ax)


def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def pick_max_thread(rows):
    by_size = {}
    for r in rows:
        size = int(r["vertices"])
        threads = int(r["threads"])
        if size not in by_size or threads > by_size[size]["threads"]:
            by_size[size] = r
    return by_size


def plot_cpu_gpu_bars(cpu_rows, gpu_rows, out_path, value_key, ylabel, title):
    cpu = pick_max_thread(cpu_rows)
    gpu = pick_max_thread(gpu_rows)
    sizes = sorted(set(cpu.keys()) & set(gpu.keys()))
    if not sizes:
        print(f"[Warn] No overlapping sizes for {title}")
        return

    cpu_vals = [cpu[s][value_key] for s in sizes]
    gpu_vals = [gpu[s][value_key] for s in sizes]

    x = np.arange(len(sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, cpu_vals, width, label="CPU (OpenMP)", color="#4C72B0")
    ax.bar(x + width / 2, gpu_vals, width, label="GPU (CUDA)", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:,}" for s in sizes])
    ax.set_xlabel("Graph vertices", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    finalize_figure(fig, ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[Plot] Saved: {out_path}")
    plt.close(fig)


def plot_efficiency_curve(rows, out_path):
    sizes = sorted(set(int(r["vertices"]) for r in rows))
    fig, ax = plt.subplots(figsize=(9, 6))

    for size in sizes:
        size_rows = sorted(
            [r for r in rows if int(r["vertices"]) == size],
            key=lambda r: int(r["threads"])
        )
        threads = [int(r["threads"]) for r in size_rows]
        eff = [r["efficiency"] for r in size_rows]
        ax.plot(threads, eff, "o-", lw=2, label=f"V={size:,}")

    ax.set_xlabel("Thread count", fontsize=12)
    ax.set_ylabel("Parallel efficiency", fontsize=12)
    ax.set_title("Parallel Efficiency vs Threads (CPU)", fontsize=13, fontweight="bold")
    ax.set_xticks(sorted(set(int(r["threads"]) for r in rows)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, color=GRID)
    ax.legend(fontsize=9)
    finalize_figure(fig, ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[Plot] Saved: {out_path}")
    plt.close(fig)


def plot_dataset_results(rows, out_dir):
    datasets = sorted(set(r["dataset"] for r in rows))
    modes = ["seq", "cpu", "gpu"]

    def value_map(key):
        data = {d: {m: 0.0 for m in modes} for d in datasets}
        for r in rows:
            data[r["dataset"]][r["mode"]] = float(r[key])
        return data

    time_map = value_map("total_ms")
    acc_map = value_map("accuracy")

    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"seq": "#4C72B0", "cpu": "#55A868", "gpu": "#DD8452"}

    for idx, mode in enumerate(modes):
        vals = [time_map[d][mode] for d in datasets]
        ax.bar(x + (idx - 1) * width, vals, width, label=mode.upper(), color=colors[mode])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Total time (ms)", fontsize=12)
    ax.set_title("Dataset Runtime Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    finalize_figure(fig, ax)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "dataset_runtime.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[Plot] Saved: {Path(out_dir) / 'dataset_runtime.png'}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, mode in enumerate(modes):
        vals = [acc_map[d][mode] * 100 for d in datasets]
        ax.bar(x + (idx - 1) * width, vals, width, label=mode.upper(), color=colors[mode])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Dataset Accuracy Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    finalize_figure(fig, ax)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "dataset_accuracy.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[Plot] Saved: {Path(out_dir) / 'dataset_accuracy.png'}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate presentation plots")
    parser.add_argument("--cpu", required=True, help="CPU perf_summary CSV")
    parser.add_argument("--gpu", required=True, help="GPU perf_summary CSV")
    parser.add_argument("--cpu-eff", required=True, help="CPU perf_efficiency CSV")
    parser.add_argument("--datasets", required=True, help="dataset_results CSV")
    parser.add_argument("--outdir", required=True, help="output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cpu_rows = read_csv(args.cpu)
    gpu_rows = read_csv(args.gpu)
    cpu_eff = read_csv(args.cpu_eff)
    dataset_rows = read_csv(args.datasets)

    plot_cpu_gpu_bars(cpu_rows, gpu_rows, outdir / "cpu_gpu_total_time.png",
                      "avg_total_ms", "Average total time (ms)",
                      "CPU vs GPU Total Time (Max Threads)")
    plot_cpu_gpu_bars(cpu_rows, gpu_rows, outdir / "cpu_gpu_throughput.png",
                      "throughput_edges_per_ms", "Edges per ms",
                      "CPU vs GPU Throughput (Max Threads)")
    plot_efficiency_curve(cpu_eff, outdir / "cpu_efficiency.png")
    plot_dataset_results(dataset_rows, outdir)


if __name__ == "__main__":
    main()
