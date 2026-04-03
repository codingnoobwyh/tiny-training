import argparse
import torch
from torch import nn

from datasets import build_data_loaders
from pathlib import Path

from baseline.models import build_native_ptq_prepare_model_from_float_state_dict


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_DIR / "artifacts" / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "artifacts" / "runs"


def get_run_dir(output_root: str, run_name: str) -> Path:
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(__import__("json").dumps(payload, indent=2))


def save_checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in data_loader:
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += images.size(0)
    return {"loss": total_loss / total_samples, "top1": 100.0 * total_correct / total_samples}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PTQ baseline from float checkpoint")
    parser.add_argument("--init-from", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--calibration-batches", type=int, default=32)
    return parser.parse_args()


def extract_float_state_dict(checkpoint: dict) -> dict:
    if checkpoint["mode"] != "float":
        raise ValueError(f"PTQ expects a float checkpoint, got mode={checkpoint['mode']}")
    return checkpoint["model_state_dict"]


def calibrate_model(model: nn.Module, data_loader, calibration_batches: int) -> None:
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(data_loader):
            model(images)
            if batch_idx + 1 >= calibration_batches:
                break


def main() -> None:
    args = parse_args()
    run_dir = get_run_dir(args.output_root, args.run_name)

    source_checkpoint = load_checkpoint(args.init_from)
    float_state_dict = extract_float_state_dict(source_checkpoint)

    train_loader, test_loader = build_data_loaders(
        args.data_root,
        args.batch_size,
        args.test_batch_size,
        args.num_workers,
    )
    prepared_model = build_native_ptq_prepare_model_from_float_state_dict(float_state_dict)
    calibrate_model(prepared_model, train_loader, args.calibration_batches)

    quantized_model = torch.ao.quantization.convert(prepared_model.eval(), inplace=False)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(quantized_model, test_loader, criterion)

    ptq_result = {
        "run_name": args.run_name,
        "mode": "ptq",
        "source_checkpoint": args.init_from,
        "calibration_batches": args.calibration_batches,
        "test_top1": metrics["top1"],
        "test_loss": metrics["loss"],
    }
    save_json(run_dir / "eval_result.json", ptq_result)
    save_json(run_dir / "ptq_result.json", ptq_result)
    save_checkpoint(run_dir / "checkpoint.pt", {
        "mode": "ptq",
        "model_state_dict": quantized_model.state_dict(),
        "source_checkpoint": args.init_from,
        "ptq_config": {
            "batch_size": args.batch_size,
            "test_batch_size": args.test_batch_size,
            "num_workers": args.num_workers,
            "calibration_batches": args.calibration_batches,
        },
    })
    print(f"{args.run_name}: mode=ptq test_top1={metrics['top1']:.2f} test_loss={metrics['loss']:.4f}")


if __name__ == "__main__":
    main()
