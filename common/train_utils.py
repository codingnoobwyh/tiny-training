from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from common.constants import (
    DEFAULT_FINETUNE_BATCH_SIZE,
    DEFAULT_FINETUNE_NUM_WORKERS,
    DEFAULT_FINETUNE_TEST_BATCH_SIZE,
    DEFAULT_FLOAT_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_QAS_LR,
    DEFAULT_QAT_LR,
    DEFAULT_QUANTIZED_LR,
    DEFAULT_SEED,
)


def get_run_dir(output_root: str, run_name: str) -> Path:
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def build_train_result(
    args,
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
        "seed": DEFAULT_SEED,
        "epochs": args.epochs,
        "batch_size": DEFAULT_FINETUNE_BATCH_SIZE,
        "test_batch_size": DEFAULT_FINETUNE_TEST_BATCH_SIZE,
        "num_workers": DEFAULT_FINETUNE_NUM_WORKERS,
        "momentum": DEFAULT_MOMENTUM,
        "float_lr": DEFAULT_FLOAT_LR,
        "qat_lr": DEFAULT_QAT_LR,
        "quantized_lr": DEFAULT_QUANTIZED_LR,
        "qas_lr": DEFAULT_QAS_LR,
        "effective_lr": lr,
        "epoch_history": epoch_history,
        "step_history": step_history,
    }


def save_checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def evaluate_model(model, data_loader, criterion) -> dict[str, float]:
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


def torch_train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion,
    optimizer,
    desc: str,
    epoch_index: int,
    global_step_start: int,
    pre_step=None,
    post_step=None,
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

        if pre_step is not None:
            pre_step(model)
        optimizer.step()
        if post_step is not None:
            post_step(model)

        predictions = logits.argmax(dim=1)
        batch_size = images.size(0)
        batch_correct = (predictions == labels).sum().item()
        total_loss += loss.item() * batch_size
        total_correct += batch_correct
        total_samples += batch_size

        step_history.append({
            "epoch": epoch_index,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "loss": loss.item(),
            "top1": 100.0 * batch_correct / batch_size,
        })
        global_step += 1
        progress.set_postfix({
            "loss": f"{total_loss / total_samples:.4f}",
            "top1": f"{100.0 * total_correct / total_samples:.2f}",
        })

    return {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_correct / total_samples,
    }, step_history, global_step
