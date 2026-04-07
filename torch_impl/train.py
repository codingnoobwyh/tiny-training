import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch_impl.datasets import build_data_loaders
from torch_impl.baseline.models import FloatMNISTNet, build_native_qat_model
from torch_impl.quantized import QASSGD, project_quantized_parameters
from torch_impl.quantized.models import QuantizedMNISTNet, initialize_quantized_model_from_ptq_checkpoint


TORCH_IMPL_DIR = Path(__file__).resolve().parent
ROOT_DIR = TORCH_IMPL_DIR.parent
DEFAULT_DATA_ROOT = ROOT_DIR / "data"
DEFAULT_OUTPUT_ROOT = TORCH_IMPL_DIR / "artifacts" / "runs"


def get_run_dir(output_root: str, run_name: str) -> Path:
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def save_checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    desc: str,
    quantized_mode: bool,
    epoch_index: int,
    global_step_start: int,
) -> tuple[dict[str, float], list[dict], int]:
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    global_step = global_step_start
    step_history = []
    progress = tqdm(data_loader, desc=desc, leave=False)
    for step_in_epoch, (images, labels) in enumerate(progress):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if hasattr(optimizer, "pre_step"):
            optimizer.pre_step(model)
        optimizer.step()
        if quantized_mode:
            project_quantized_parameters(model)

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        step_top1 = 100.0 * (logits.argmax(dim=1) == labels).sum().item() / images.size(0)
        step_history.append({
            "epoch": epoch_index,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "loss": loss.item(),
            "top1": step_top1,
        })
        global_step += 1
        progress.set_postfix({"loss": f"{total_loss / total_samples:.4f}", "top1": f"{100.0 * total_correct / total_samples:.2f}"})

    return {"loss": total_loss / total_samples, "top1": 100.0 * total_correct / total_samples}, step_history, global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one route")
    parser.add_argument("--mode", choices=["float", "qat", "quantized", "qas"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--float-lr", type=float, default=0.001)
    parser.add_argument("--qat-lr", type=float, default=0.001)
    parser.add_argument("--quantized-lr", type=float, default=0.001)
    parser.add_argument("--qas-lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def initialize_qat_from_float(model: nn.Module, init_from: str) -> None:
    checkpoint = load_checkpoint(init_from)
    if checkpoint["mode"] != "float":
        raise ValueError(f"qat runner expects float checkpoint, got {checkpoint['mode']}")
    source_state_dict = checkpoint["model_state_dict"]
    target_state_dict = model.state_dict()
    shared_state_dict = {
        key: value
        for key, value in source_state_dict.items()
        if key in target_state_dict and target_state_dict[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(shared_state_dict, strict=False)
    print(
        "Initialized QAT model from float checkpoint: "
        f"loaded={len(shared_state_dict)} missing={len(missing)} unexpected={len(unexpected)}"
    )


def build_model_and_optimizer(args: argparse.Namespace) -> tuple[nn.Module, torch.optim.Optimizer, float, bool]:
    if args.mode == "float":
        model = FloatMNISTNet()
        if args.init_from is not None:
            checkpoint = load_checkpoint(args.init_from)
            if checkpoint["mode"] != "float":
                raise ValueError(f"float runner expects float checkpoint, got {checkpoint['mode']}")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        lr = args.float_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
        return model, optimizer, lr, False

    if args.mode == "qat":
        model = build_native_qat_model()
        if args.init_from is not None:
            initialize_qat_from_float(model, args.init_from)
        lr = args.qat_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
        return model, optimizer, lr, False

    if args.init_from is None:
        raise ValueError(f"{args.mode} runner expects --init-from PTQ checkpoint")
    checkpoint = load_checkpoint(args.init_from)
    if checkpoint["mode"] != "ptq":
        raise ValueError(f"{args.mode} runner expects PTQ checkpoint, got {checkpoint['mode']}")

    model = QuantizedMNISTNet()
    initialize_quantized_model_from_ptq_checkpoint(model, checkpoint)
    if args.mode == "quantized":
        lr = args.quantized_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    else:
        lr = args.qas_lr
        optimizer = QASSGD(model.parameters(), lr=lr, momentum=args.momentum)
    return model, optimizer, lr, True


def build_train_result(
    args: argparse.Namespace,
    epoch_history: list[dict],
    step_history: list[dict],
    lr: float,
    init_from_mode: str | None,
) -> dict:
    return {
        "mode": args.mode,
        "run_name": args.run_name,
        "init_from": args.init_from,
        "init_from_mode": init_from_mode,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "num_workers": args.num_workers,
        "momentum": args.momentum,
        "float_lr": args.float_lr,
        "qat_lr": args.qat_lr,
        "quantized_lr": args.quantized_lr,
        "qas_lr": args.qas_lr,
        "effective_lr": lr,
        "epoch_history": epoch_history,
        "step_history": step_history,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_loader, _ = build_data_loaders(
        args.data_root,
        args.batch_size,
        args.test_batch_size,
        args.num_workers,
    )
    model, optimizer, lr, quantized_mode = build_model_and_optimizer(args)
    criterion = nn.CrossEntropyLoss()
    run_dir = get_run_dir(args.output_root, args.run_name)

    print(f"===== Training mode={args.mode}, lr={lr}, run_dir={run_dir} =====")

    epoch_history = []
    step_history = []
    global_step = 0
    for epoch in range(args.epochs):
        train_metrics, epoch_steps, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            desc=f"{args.mode} train {epoch + 1}/{args.epochs}",
            quantized_mode=quantized_mode,
            epoch_index=epoch,
            global_step_start=global_step,
        )
        epoch_history.append({"epoch": epoch, "train": train_metrics})
        step_history.extend(epoch_steps)
        print(f"{args.mode} epoch={epoch} train_top1={train_metrics['top1']:.2f} train_loss={train_metrics['loss']:.4f}")

    init_from_mode = None
    if args.init_from is not None:
        init_from_mode = load_checkpoint(args.init_from)["mode"]
    train_result = build_train_result(args, epoch_history, step_history, lr, init_from_mode)
    save_json(run_dir / "train_result.json", train_result)
    save_checkpoint(run_dir / "checkpoint.pt", {
        "mode": args.mode,
        "model_state_dict": model.state_dict(),
        "train_config": train_result,
    })
    print(f"Saved training artifacts to {run_dir}")


if __name__ == "__main__":
    main()
