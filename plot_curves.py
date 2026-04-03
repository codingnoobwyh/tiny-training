import argparse
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "artifacts" / "runs"
DEFAULT_COMPARE_RUNS = ["fp", "qat", "quant", "qas"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_loss_curves(run_dir: Path, train_result: dict) -> None:
    import matplotlib.pyplot as plt

    epoch_history = train_result["epoch_history"]
    step_history = train_result["step_history"]
    epoch_indices = [row["epoch"] for row in epoch_history]
    epoch_losses = [row["train"]["loss"] for row in epoch_history]
    step_indices = [row["global_step"] for row in step_history]
    step_losses = [row["loss"] for row in step_history]

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)

    axes[0].plot(epoch_indices, epoch_losses, marker="o")
    axes[0].set_title("Epoch Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(step_indices, step_losses, linewidth=1.0, alpha=0.35, label="raw")
    if len(step_losses) >= 5:
        window = min(50, len(step_losses))
        smoothed = []
        for idx in range(len(step_losses)):
            start = max(0, idx - window + 1)
            smoothed.append(sum(step_losses[start : idx + 1]) / (idx - start + 1))
        axes[1].plot(step_indices, smoothed, linewidth=1.5, label=f"moving_avg_{window}")
        axes[1].legend()
    axes[1].set_title("Step Loss")
    axes[1].set_xlabel("global_step")
    axes[1].set_ylabel("loss")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Training Curves: {train_result['run_name']} ({train_result['mode']})")
    output_path = run_dir / "loss_curves.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved loss curves to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training loss curves")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def save_epoch_compare_curves(output_root: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for run_name in DEFAULT_COMPARE_RUNS:
        run_dir = output_root / run_name
        train_result = load_json(run_dir / "train_result.json")
        epoch_history = train_result["epoch_history"]
        epoch_indices = [row["epoch"] for row in epoch_history]
        epoch_losses = [row["train"]["loss"] for row in epoch_history]
        ax.plot(epoch_indices, epoch_losses, marker="o", linewidth=1.5, label=run_name)

    ax.set_title("Epoch Loss Comparison")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = output_root / "epoch_loss_compare.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved epoch comparison curves to {output_path}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if args.run_name is None:
        save_epoch_compare_curves(output_root)
        return

    run_dir = output_root / args.run_name
    train_result = load_json(run_dir / "train_result.json")
    save_loss_curves(run_dir, train_result)


if __name__ == "__main__":
    main()
