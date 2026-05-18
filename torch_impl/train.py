import argparse
import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MOMENTUM,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PTQ_CHECKPOINT_PATH,
    DEFAULT_QAS_LR,
    DEFAULT_QUANTIZED_LR,
    DEFAULT_SEED,
    DEFAULT_TEST_BATCH_SIZE,
)
from common.dataset import DEFAULT_DATA_ROOT, build_data_loaders
from common.train_utils import build_train_result, get_run_dir, load_checkpoint, save_checkpoint, save_json, torch_train_one_epoch
from torch_impl.checkpoint import init_model_from_ptq
from torch_impl.model import QasMnistNet
from torch_impl.operators import QASSGD, project_quantized_parameters

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train quantized / QAS route")
    parser.add_argument("--mode", choices=["quantized", "qas"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--init-from", default=DEFAULT_PTQ_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


def build_model_and_optimizer(args: argparse.Namespace) -> tuple[nn.Module, torch.optim.Optimizer, float, str]:
    checkpoint = load_checkpoint(args.init_from)
    if checkpoint["mode"] != "ptq":
        raise ValueError(f"{args.mode} runner expects PTQ checkpoint, got {checkpoint['mode']}")

    model = QasMnistNet()
    init_model_from_ptq(model, checkpoint)

    if args.mode == "quantized":
        lr = DEFAULT_QUANTIZED_LR
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=DEFAULT_MOMENTUM)
    else:
        lr = DEFAULT_QAS_LR
        optimizer = QASSGD(model.parameters(), lr=lr, momentum=DEFAULT_MOMENTUM)

    return model, optimizer, lr, checkpoint["mode"]


def main() -> None:
    args = parse_args()
    torch.manual_seed(DEFAULT_SEED)

    train_loader, _ = build_data_loaders(
        DEFAULT_DATA_ROOT,
        DEFAULT_BATCH_SIZE,
        DEFAULT_TEST_BATCH_SIZE,
        DEFAULT_NUM_WORKERS,
    )
    model, optimizer, lr, init_from_mode = build_model_and_optimizer(args)
    criterion = nn.CrossEntropyLoss()
    run_dir = get_run_dir(DEFAULT_OUTPUT_ROOT, args.run_name)

    print(f"===== Training mode={args.mode}, lr={lr}, run_dir={run_dir} =====")

    epoch_history = []
    step_history = []
    global_step = 0
    for epoch in range(args.epochs):
        train_metrics, epoch_steps, global_step = torch_train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            desc=f"{args.mode} train {epoch + 1}/{args.epochs}",
            epoch_index=epoch,
            global_step_start=global_step,
            run_dir=run_dir,
            pre_step=optimizer.pre_step if isinstance(optimizer, QASSGD) else None,
            post_step=project_quantized_parameters,
        )
        epoch_history.append({"epoch": epoch, "train": train_metrics})
        step_history.extend(epoch_steps)
        print(f"{args.mode} epoch={epoch} train_top1={train_metrics['top1']:.2f} train_loss={train_metrics['loss']:.4f}")

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
