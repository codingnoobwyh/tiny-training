import argparse
import sys
from pathlib import Path

import torch
import torch.ao.quantization as tq
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from common.constants import (
    DEFAULT_BASELINE_BATCH_SIZE,
    DEFAULT_BASELINE_EPOCHS,
    DEFAULT_BASELINE_NUM_WORKERS,
    DEFAULT_BASELINE_RUN_NAME,
    DEFAULT_BASELINE_TEST_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_QAT_LR,
    DEFAULT_SEED,
)
from common.baseline.models import (
    FloatMnistNet,
    QuantizedMnistNet,
    build_qat_model,
    initialize_qat_from_float,
)
from common.dataset import DEFAULT_DATA_ROOT, build_data_loaders
from common.train_utils import evaluate_model, get_run_dir, load_checkpoint, save_checkpoint, save_json, torch_train_one_epoch

BASELINE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = BASELINE_DIR / "artifacts"


def _make_train_result(args, mode: str, lr: float, epoch_history: list, step_history: list, init_from: str | None) -> dict:
    return {
        "mode": mode,
        "run_name": args.run_name,
        "init_from": init_from,
        "init_from_mode": "float" if init_from else None,
        "seed": DEFAULT_SEED,
        "epochs": args.epochs,
        "batch_size": DEFAULT_BASELINE_BATCH_SIZE,
        "test_batch_size": DEFAULT_BASELINE_TEST_BATCH_SIZE,
        "num_workers": DEFAULT_BASELINE_NUM_WORKERS,
        "momentum": DEFAULT_MOMENTUM,
        "effective_lr": lr,
        "epoch_history": epoch_history,
        "step_history": step_history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline pipeline: float → PTQ → QAT")
    parser.add_argument("--run-name", default=DEFAULT_BASELINE_RUN_NAME)
    parser.add_argument("--epochs", type=int, default=DEFAULT_BASELINE_EPOCHS)
    parser.add_argument("--calibration-batches", type=int, default=32)
    return parser.parse_args()


def _train_loop(model, optimizer, train_loader, criterion, args, run_dir, mode, lr, init_from):
    epoch_history = []
    step_history = []
    global_step = 0
    for epoch in range(args.epochs):
        train_metrics, epoch_steps, global_step = torch_train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            desc=f"{mode} train {epoch + 1}/{args.epochs}",
            epoch_index=epoch,
            global_step_start=global_step,
        )
        epoch_history.append({"epoch": epoch, "train": train_metrics})
        step_history.extend(epoch_steps)
        print(f"{mode} epoch={epoch} train_top1={train_metrics['top1']:.2f} train_loss={train_metrics['loss']:.4f}")

    train_result = _make_train_result(args, mode, lr, epoch_history, step_history, init_from)
    save_json(run_dir / "train_result.json", train_result)
    save_checkpoint(run_dir / "checkpoint.pt", {
        "mode": mode,
        "model_state_dict": model.state_dict(),
    })
    return train_result


def main() -> None:
    args = parse_args()
    torch.manual_seed(DEFAULT_SEED)

    train_loader, test_loader = build_data_loaders(
        DEFAULT_DATA_ROOT,
        DEFAULT_BASELINE_BATCH_SIZE,
        DEFAULT_BASELINE_TEST_BATCH_SIZE,
        DEFAULT_BASELINE_NUM_WORKERS,
    )
    criterion = nn.CrossEntropyLoss()

    # ———— Phase 1: Float training ————
    float_dir = get_run_dir(DEFAULT_OUTPUT_ROOT, args.run_name)
    print(f"===== Phase 1: Float training, lr={DEFAULT_LR}, run_dir={float_dir} =====")

    float_model = FloatMnistNet()
    float_optimizer = torch.optim.SGD(float_model.parameters(), lr=DEFAULT_LR, momentum=DEFAULT_MOMENTUM)
    _train_loop(float_model, float_optimizer, train_loader, criterion, args, float_dir, "float", DEFAULT_LR, None)

    float_metrics = evaluate_model(float_model, test_loader, criterion)
    save_json(float_dir / "eval_result.json", {
        "run_name": args.run_name,
        "mode": "float",
        "test_top1": float_metrics["top1"],
        "test_loss": float_metrics["loss"],
    })
    print(f"Float eval: top1={float_metrics['top1']:.2f} loss={float_metrics['loss']:.4f}")

    float_ckpt_path = str(float_dir / "checkpoint.pt")

    # ———— Phase 2: PTQ ————
    ptq_dir = get_run_dir(DEFAULT_OUTPUT_ROOT, f"{args.run_name}_ptq")
    print(f"===== Phase 2: PTQ, run_dir={ptq_dir} =====")

    float_state_dict = load_checkpoint(float_ckpt_path)["model_state_dict"]
    ptq_model = QuantizedMnistNet()
    ptq_model.load_state_dict(float_state_dict, strict=True)
    ptq_model.fuse_model()
    ptq_model.qconfig = tq.get_default_qconfig("fbgemm")
    ptq_model = tq.prepare(ptq_model.eval(), inplace=False)
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(train_loader):
            ptq_model(images)
            if batch_idx + 1 >= args.calibration_batches:
                break
    ptq_model = tq.convert(ptq_model.eval(), inplace=False)

    ptq_metrics = evaluate_model(ptq_model, test_loader, criterion)
    ptq_result = {
        "run_name": f"{args.run_name}_ptq",
        "mode": "ptq",
        "source_checkpoint": float_ckpt_path,
        "calibration_batches": args.calibration_batches,
        "test_top1": ptq_metrics["top1"],
        "test_loss": ptq_metrics["loss"],
    }
    save_json(ptq_dir / "eval_result.json", ptq_result)
    save_checkpoint(ptq_dir / "checkpoint.pt", {
        "mode": "ptq",
        "model_state_dict": ptq_model.state_dict(),
        "ptq_config": {
            "batch_size": DEFAULT_BASELINE_BATCH_SIZE,
            "test_batch_size": DEFAULT_BASELINE_TEST_BATCH_SIZE,
            "num_workers": DEFAULT_BASELINE_NUM_WORKERS,
            "calibration_batches": args.calibration_batches,
        },
    })
    print(f"PTQ eval: top1={ptq_metrics['top1']:.2f} loss={ptq_metrics['loss']:.4f}")

    # ———— Phase 3: QAT ————
    qat_dir = get_run_dir(DEFAULT_OUTPUT_ROOT, f"{args.run_name}_qat")
    print(f"===== Phase 3: QAT training, lr={DEFAULT_QAT_LR}, run_dir={qat_dir} =====")

    qat_model = build_qat_model()
    initialize_qat_from_float(qat_model, float_ckpt_path)
    qat_optimizer = torch.optim.SGD(qat_model.parameters(), lr=DEFAULT_QAT_LR, momentum=DEFAULT_MOMENTUM)
    _train_loop(qat_model, qat_optimizer, train_loader, criterion, args, qat_dir, "qat", DEFAULT_QAT_LR, float_ckpt_path)

    qat_metrics = evaluate_model(qat_model, test_loader, criterion)
    save_json(qat_dir / "eval_result.json", {
        "run_name": f"{args.run_name}_qat",
        "mode": "qat",
        "test_top1": qat_metrics["top1"],
        "test_loss": qat_metrics["loss"],
    })
    print(f"QAT eval: top1={qat_metrics['top1']:.2f} loss={qat_metrics['loss']:.4f}")

    print(f"All baselines complete. Float: {float_dir}, PTQ: {ptq_dir}, QAT: {qat_dir}")


if __name__ == "__main__":
    main()
