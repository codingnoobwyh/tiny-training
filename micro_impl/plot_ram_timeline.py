#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path


DEFAULT_SOURCES = [
    ("inference", Path("inference/ram_segments.py")),
    ("training", Path("training/ram_segments.py")),
]

PALETTE = {
    "free": "#f1f5f9",
    "gap": "#e2e8f0",
    "old fp32": "#cbd5e1",
    "old tmp": "#cbd5e1",
    "old FC1": "#cbd5e1",
    "old int8": "#cbd5e1",
    "prev grads": "#dbeafe",
    "all grads": "#dbeafe",
    "forward": "#a7f3d0",
    "fp32 input": "#38bdf8",
    "q input": "#0ea5e9",
    "conv1 out": "#f97316",
    "pool1 out": "#fb923c",
    "conv2 out": "#f59e0b",
    "pool2 out": "#fbbf24",
    "NCHW tmp": "#94a3b8",
    "FC1 input": "#818cf8",
    "FC1 out": "#6366f1",
    "ReLU out": "#8b5cf6",
    "FC2 input": "#7c3aed",
    "logits": "#ef4444",
    "fp32 logits": "#f87171",
    "sum": "#14b8a6",
    "CE": "#fda4af",
    "CE old": "#fecdd3",
    "labels": "#fb7185",
    "prob": "#f43f5e",
    "grad logits": "#e11d48",
    "fc2 grads": "#c084fc",
    "fc2 grad in": "#d8b4fe",
    "fc2 grad weight": "#c084fc",
    "fc2 grad bias": "#a855f7",
    "fc1 grads": "#93c5fd",
    "fc1 grad out": "#bfdbfe",
    "fc1 grad in": "#93c5fd",
    "fc1 grad bias": "#60a5fa",
    "fc1 grad weight": "#3b82f6",
    "flatten grad": "#6ee7b7",
    "pool2 grad": "#34d399",
    "pool2 grad input": "#34d399",
    "relu2 grad": "#2dd4bf",
    "relu2 grad input": "#2dd4bf",
    "conv2 grads": "#facc15",
    "conv2 grad input": "#fde047",
    "conv2 grad weight": "#eab308",
    "conv2 grad bias": "#ca8a04",
    "pool1 grad": "#fdba74",
    "pool1 grad input": "#fdba74",
    "conv1 out grad": "#fb923c",
    "conv1 grads": "#f97316",
    "conv1 grad input": "#fb923c",
    "conv1 grad weight": "#ea580c",
    "conv1 grad bias": "#c2410c",
    "momentum": "#64748b",
}


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iter_groups(base_dir):
    for source_name, rel_path in DEFAULT_SOURCES:
        module = load_module(base_dir / rel_path)
        for group in module.SEGMENT_GROUPS:
            yield source_name, group


def color_for(name):
    if name in PALETTE:
        return PALETTE[name]
    return "#94a3b8"


def format_bytes(value, _pos=None):
    if value >= 1024:
        return f"{value / 1024:.0f}K"
    return str(int(value))


def draw_group(ax, source_name, group, xlim=None, title=None):
    row_height = 0.72
    rows = group["rows"]
    total_size = group["total_size"]
    label_threshold = (xlim or total_size) * 0.035
    y_positions = list(range(len(rows)))[::-1]

    for y, row in zip(y_positions, rows):
        for name, start, end in row["segments"]:
            width = end - start
            if width <= 0:
                raise ValueError(f"bad segment {source_name}/{row['name']}: {name} {start}..{end}")
            ax.broken_barh(
                [(start, width)],
                (y - row_height / 2, row_height),
                facecolors=color_for(name),
                edgecolors="#334155",
                linewidth=0.35,
            )
            if width >= label_threshold:
                ax.text(
                    start + width / 2,
                    y,
                    name,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="#0f172a",
                    clip_on=True,
                )

    ax.set_title(title or f"{source_name}: {group['title']} ({total_size} bytes)", fontsize=11)
    ax.set_xlim(0, xlim or total_size)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row["name"] for row in rows], fontsize=8)
    ax.grid(axis="x", color="#cbd5e1", linewidth=0.6, alpha=0.8)
    ax.set_xlabel("m0_buffer offset / bytes")
    ax.xaxis.set_major_formatter(format_bytes)


def resolve_output(base_dir, output):
    path = Path(output)
    if not path.is_absolute():
        path = base_dir / path
    return path


def save_figure(fig, base_dir, output):
    output = resolve_output(base_dir, output)
    fig.savefig(output, dpi=200)
    print(f"saved {output}")


def load_segment_groups(base_dir):
    modules = {
        name: load_module(base_dir / rel_path) for name, rel_path in DEFAULT_SOURCES
    }
    return modules


def plot_reuse_compare(base_dir, modules, output):
    import matplotlib.pyplot as plt

    inference_reuse = modules["inference"].SEGMENT_GROUPS[0]
    training_forward = modules["training"].SEGMENT_GROUPS[0]
    xlim = max(inference_reuse["total_size"], training_forward["total_size"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(18, 10),
        sharex=True,
        constrained_layout=True,
    )
    draw_group(
        axes[0],
        "training",
        training_forward,
        xlim=xlim,
        title=f"without reuse: forward retained activations ({training_forward['total_size']} bytes)",
    )
    draw_group(
        axes[1],
        "inference",
        inference_reuse,
        xlim=xlim,
        title=f"with reuse: inference forward timeline ({inference_reuse['total_size']} bytes)",
    )
    save_figure(fig, base_dir, output)
    return fig


def plot_training_only(base_dir, modules, output):
    import matplotlib.pyplot as plt

    training_backward = modules["training"].SEGMENT_GROUPS[1]
    height = max(4.0, len(training_backward["rows"]) * 0.38 + 1.0)
    fig, ax = plt.subplots(1, 1, figsize=(18, height), constrained_layout=True)
    draw_group(
        ax,
        "training",
        training_backward,
        title=f"training backward and update timeline ({training_backward['total_size']} bytes)",
    )
    save_figure(fig, base_dir, output)
    return fig


def plot_all(base_dir, groups, output):
    import matplotlib.pyplot as plt

    heights = [max(2.2, len(group["rows"]) * 0.38 + 1.0) for _, group in groups]
    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(18, sum(heights)),
        gridspec_kw={"height_ratios": heights},
        constrained_layout=True,
    )
    if len(groups) == 1:
        axes = [axes]

    for ax, (source_name, group) in zip(axes, groups):
        draw_group(ax, source_name, group)

    save_figure(fig, base_dir, output)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot micro m0_buffer RAM timelines.")
    parser.add_argument(
        "-o",
        "--compare-output",
        default="ram_reuse_compare.png",
        help="reuse comparison image path, relative to micro_impl by default",
    )
    parser.add_argument(
        "--training-output",
        default="training_ram_timeline.png",
        help="training-only image path, relative to micro_impl by default",
    )
    parser.add_argument(
        "--all-output",
        default=None,
        help="optional full timeline image path, relative to micro_impl by default",
    )
    parser.add_argument("--show", action="store_true", help="show the figure window")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    modules = load_segment_groups(base_dir)
    groups = list(iter_groups(base_dir))
    if not groups:
        raise RuntimeError("no RAM segment groups found")

    figures = [
        plot_reuse_compare(base_dir, modules, args.compare_output),
        plot_training_only(base_dir, modules, args.training_output),
    ]
    if args.all_output is not None:
        figures.append(plot_all(base_dir, groups, args.all_output))

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
