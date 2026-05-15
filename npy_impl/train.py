"""
NumPy 版量化训练入口.

入口参数和训练产物尽量对齐 torch_impl/train.py 的 quantized/qas 通路.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from common.constants import TRAINABLE_LAYERS
from common.train_utils import build_train_result, get_run_dir, load_checkpoint, save_json, save_step_txt_dump
from common.dataset import DEFAULT_DATA_ROOT, build_data_loaders
from npy_impl import (
    QuantizedMNISTNet,
    cross_entropy_loss,
    load_quantized_model_params,
    make_trainable_params,
    project_quantized_parameters,
    qas_pre_step,
    sgd_step,
)

NPY_IMPL_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = NPY_IMPL_DIR / "artifacts" / "runs"



def save_numpy_checkpoint(path: Path, mode: str, params: dict[str, dict[str, np.ndarray]], train_config: dict) -> None:
    payload = {
        "mode": np.array(mode),
        "train_config_json": np.array(json.dumps(train_config)),
    }
    for layer_name, layer_params in params.items():
        for key, value in layer_params.items():
            if value is not None:
                payload[f"{layer_name}.{key}"] = value
    np.savez(path, **payload)


def _to_numpy_batch(images: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return (
        images.detach().cpu().numpy().astype(np.float32, copy=False),
        labels.detach().cpu().numpy().astype(np.int64, copy=False),
    )


def _parameter_snapshot(params: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    snapshot = {}
    for layer_name in TRAINABLE_LAYERS:
        snapshot[f"{layer_name}.weight"] = params[layer_name]["weight"].copy()
        if params[layer_name]["bias"] is not None:
            snapshot[f"{layer_name}.bias"] = params[layer_name]["bias"].copy()
    return snapshot


def _gradient_snapshot(grads: dict[str, dict[str, np.ndarray] | np.ndarray]) -> dict[str, np.ndarray | None]:
    snapshot = {}
    for layer_name in TRAINABLE_LAYERS:
        snapshot[f"{layer_name}.weight"] = grads[layer_name]["weight"].copy()
        snapshot[f"{layer_name}.bias"] = None if grads[layer_name]["bias"] is None else grads[layer_name]["bias"].copy()
    return snapshot


def train_one_epoch(
    model: QuantizedMNISTNet,
    data_loader,
    lr: float,
    momentum: float,
    use_qas: bool,
    epoch_index: int,
    global_step_start: int,
    momentum_buffers: dict[str, dict[str, np.ndarray]],
    run_dir: Path,
    dump_step_txt: bool,
) -> tuple[dict[str, float], list[dict], int]:
    from tqdm import tqdm

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    global_step = global_step_start
    step_history = []

    progress = tqdm(data_loader, desc=f"npy {epoch_index + 1}", leave=False)
    for step_in_epoch, (images_t, labels_t) in enumerate(progress):
        images, labels = _to_numpy_batch(images_t, labels_t)

        weights_before = _parameter_snapshot(model.params) if dump_step_txt else None
        logits = model.forward(images)
        loss = cross_entropy_loss(logits, labels)
        grads = model.backward_from_labels(labels)
        gradients = _gradient_snapshot(grads) if dump_step_txt else None
        if use_qas:
            grads = qas_pre_step(model.params, grads)
        sgd_step(
            model.params,
            grads,
            lr=lr,
            momentum=momentum,
            momentum_buffers=momentum_buffers,
        )
        project_quantized_parameters(model.params)
        weights_after = _parameter_snapshot(model.params) if dump_step_txt else None

        dump_path = None
        if dump_step_txt:
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

        predictions = np.argmax(logits, axis=1)
        correct = int(np.sum(predictions == labels))
        batch_size = int(images.shape[0])
        step_top1 = 100.0 * correct / batch_size

        total_loss += loss * batch_size
        total_correct += correct
        total_samples += batch_size
        running_loss = total_loss / total_samples
        running_top1 = 100.0 * total_correct / total_samples
        step_history.append({
            "epoch": epoch_index,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "loss": loss,
            "top1": step_top1,
            "step_txt_dump": None if dump_path is None else str(dump_path),
        })
        global_step += 1
        progress.set_postfix({"loss": f"{running_loss:.4f}", "top1": f"{running_top1:.2f}"})

    return {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_correct / total_samples,
    }, step_history, global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one NumPy quantized route")
    parser.add_argument("--mode", choices=["quantized", "qas"], required=True)
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
    parser.add_argument("--dump-step-txt", action="store_true")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> tuple[QuantizedMNISTNet, float, str | None]:
    if args.init_from is None:
        raise ValueError(f"{args.mode} runner expects --init-from PTQ checkpoint")

    checkpoint = load_checkpoint(args.init_from)
    init_from_mode = checkpoint.get("mode")
    if init_from_mode != "ptq":
        raise ValueError(f"{args.mode} runner expects PTQ checkpoint, got {init_from_mode}")

    base_params = load_quantized_model_params(args.init_from)
    trainable_params = make_trainable_params(base_params)
    model = QuantizedMNISTNet(trainable_params)
    lr = args.quantized_lr if args.mode == "quantized" else args.qas_lr
    return model, lr, init_from_mode


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader, _ = build_data_loaders(
        args.data_root,
        args.batch_size,
        args.test_batch_size,
        args.num_workers,
    )
    model, lr, init_from_mode = build_model(args)
    run_dir = get_run_dir(args.output_root, args.run_name)

    print(f"===== NumPy training mode={args.mode}, lr={lr}, run_dir={run_dir} =====")

    epoch_history = []
    step_history = []
    momentum_buffers: dict[str, dict[str, np.ndarray]] = {}
    global_step = 0
    use_qas = args.mode == "qas"

    for epoch in range(args.epochs):
        train_metrics, epoch_steps, global_step = train_one_epoch(
            model,
            train_loader,
            lr=lr,
            momentum=args.momentum,
            use_qas=use_qas,
            epoch_index=epoch,
            global_step_start=global_step,
            momentum_buffers=momentum_buffers,
            run_dir=run_dir,
            dump_step_txt=args.dump_step_txt,
        )
        epoch_history.append({"epoch": epoch, "train": train_metrics})
        step_history.extend(epoch_steps)
        print(f"{args.mode} epoch={epoch} train_top1={train_metrics['top1']:.2f} train_loss={train_metrics['loss']:.4f}")

    train_result = build_train_result(args, epoch_history, step_history, lr, init_from_mode)
    save_json(run_dir / "train_result.json", train_result)
    save_numpy_checkpoint(run_dir / "checkpoint.npz", args.mode, model.params, train_result)
    print(f"Saved NumPy training artifacts to {run_dir}")


if __name__ == "__main__":
    main()
