from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_FLOAT_LR,
    DEFAULT_MOMENTUM,
    DEFAULT_NUM_WORKERS,
    DEFAULT_QAS_LR,
    DEFAULT_QAT_LR,
    DEFAULT_QUANTIZED_LR,
    DEFAULT_SEED,
    DEFAULT_TEST_BATCH_SIZE,
)


def get_run_dir(output_root: str, run_name: str) -> Path:
    run_dir = Path(output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _format_scalar(value: float, scientific: bool = False) -> str:
    if scientific:
        return f"{float(value):.6e}"
    return f"{value:.6f}"


def _format_kv(key: str, value: str, width: int = 20) -> str:
    return f"{key:<{width}}: {value}"


def _tensor_preview(array, limit: int = 10, *, scientific: bool = False) -> str:
    if array is None:
        return "None"
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    values = np.asarray(array).reshape(-1)[:limit].tolist()

    def _format_value(value) -> str:
        w = 12
        if isinstance(value, (bool, np.bool_)):
            return f"{int(value):>{w}d}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):>{w}d}"
        value = float(value)
        if scientific:
            return f"{value:>{w}.4e}"
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value)):>{w}d}"
        return f"{value:>{w}.4f}"

    return "[" + ", ".join(_format_value(v) for v in values) + "]"


def save_step_txt_dump(
    run_dir: Path,
    epoch_index: int,
    step_in_epoch: int,
    global_step: int,
    images,
    labels,
    logits,
    loss,
    weights_before: dict,
    gradients: dict,
    weights_after: dict,
) -> Path:
    dump_dir = run_dir / "step_txt_dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_path = dump_dir / f"epoch_{epoch_index:04d}.txt"

    if hasattr(loss, "detach"):
        loss_val = float(loss.detach().cpu().item())
    else:
        loss_val = float(loss)

    if hasattr(logits, "detach"):
        preds = logits.argmax(dim=1)
    else:
        preds = np.argmax(logits, axis=1)

    lines = [
        "=" * 80,
        _format_kv("epoch", str(epoch_index)),
        _format_kv("step_in_epoch", str(step_in_epoch)),
        _format_kv("global_step", str(global_step)),
        _format_kv("loss", _format_scalar(loss_val)),
        _format_kv("labels", _tensor_preview(labels, 10)),
        _format_kv("logits", _tensor_preview(logits, 10)),
        _format_kv("predictions", _tensor_preview(preds, 10)),
        _format_kv("images", _tensor_preview(images, 10)),
        "",
    ]

    for name in weights_before.keys():
        lines.extend([
            f"[{name}]",
            _format_kv("weight_before", _tensor_preview(weights_before[name], 10)),
            _format_kv("grad", _tensor_preview(gradients.get(name), 10, scientific=True)),
            _format_kv("weight_after", _tensor_preview(weights_after[name], 10)),
            "",
        ])

    content = "\n".join(lines) + "\n"
    if step_in_epoch == 0:
        dump_path.write_text(content)
    else:
        with dump_path.open("a") as handle:
            handle.write("\n" + content)
    return dump_path


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
        "batch_size": DEFAULT_BATCH_SIZE,
        "test_batch_size": DEFAULT_TEST_BATCH_SIZE,
        "num_workers": DEFAULT_NUM_WORKERS,
        "momentum": DEFAULT_MOMENTUM,
        "float_lr": DEFAULT_FLOAT_LR,
        "qat_lr": DEFAULT_QAT_LR,
        "quantized_lr": DEFAULT_QUANTIZED_LR,
        "qas_lr": DEFAULT_QAS_LR,
        "effective_lr": lr,
        "epoch_history": epoch_history,
        "step_history": step_history,
    }


# —————————————————————————— torch helpers ——————————————————————————


def _named_parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def _named_gradient_snapshot(model: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if param.grad is None else param.grad.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def _write_torch_step_dump(
    run_dir: Path,
    epoch_index: int,
    step_in_epoch: int,
    global_step: int,
    images,
    labels,
    logits,
    loss,
    weights_before,
    gradients,
    weights_after,
) -> str:
    dump_path = save_step_txt_dump(
        run_dir=run_dir,
        epoch_index=epoch_index,
        step_in_epoch=step_in_epoch,
        global_step=global_step,
        images=images,
        labels=labels,
        logits=logits,
        loss=loss,
        weights_before=weights_before,
        gradients=gradients,
        weights_after=weights_after,
    )
    return str(dump_path)


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
    run_dir: Path,
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
        weights_before = _named_parameter_snapshot(model)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        if pre_step is not None:
            pre_step(model)
        gradients = _named_gradient_snapshot(model)
        optimizer.step()
        if post_step is not None:
            post_step(model)

        weights_after = _named_parameter_snapshot(model)
        dump_path = _write_torch_step_dump(
            run_dir=run_dir,
            epoch_index=epoch_index,
            step_in_epoch=step_in_epoch,
            global_step=global_step,
            images=images,
            labels=labels,
            logits=logits,
            loss=loss,
            weights_before=weights_before,
            gradients=gradients,
            weights_after=weights_after,
        )

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
            "step_txt_dump": dump_path,
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
