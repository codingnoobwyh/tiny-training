"""
统一对拍入口：
1. 前向对拍
2. 单步训练对拍（quantized / qas）
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from mnist_data import load_mnist_float
from npy_impl import (
    QuantizedMNISTNet as NumpyQuantizedMNISTNet,
    cross_entropy_loss,
    load_quantized_model_params,
    make_trainable_params,
    project_quantized_parameters as numpy_project_quantized_parameters,
    qas_pre_step,
    sgd_step,
)
from torch_impl.quantized import QASSGD, project_quantized_parameters as torch_project_quantized_parameters
from torch_impl.quantized.models import (
    QuantizedMNISTNet as TorchQuantizedMNISTNet,
    initialize_quantized_model_from_ptq_checkpoint,
)
from torch_impl.datasets import build_data_loaders


PTQ_CHECKPOINT_PATH = REPO_ROOT / "torch_impl" / "artifacts" / "runs" / "ptq" / "checkpoint.pt"
OUTPUT_DIR = REPO_ROOT / "npy_impl" / "artifacts"
TRAINABLE_NAMES = ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")
TRAINABLE_LAYERS = ("conv1", "conv2", "fc1", "fc2")


def _tensor_preview(array, limit: int = 10, *, scientific: bool = False) -> str:
    arr = np.asarray(array).reshape(-1)
    values = arr[:limit].tolist()

    def _format_value(value) -> str:
        width = 12
        if isinstance(value, (bool, np.bool_)):
            return f"{int(value):>{width}d}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):>{width}d}"
        value = float(value)
        if scientific:
            return f"{value:>{width}.4e}"
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value)):>{width}d}"
        return f"{value:>{width}.4f}"

    return "[" + ", ".join(_format_value(v) for v in values) + "]"


def _format_kv(key: str, value: str) -> str:
    return f"{key:<24}: {value}"


def _format_float(value: float) -> str:
    return f"{float(value):.6e}"


def _save_report(filename: str, lines: list[str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = OUTPUT_DIR / filename
    dump_path.write_text("\n".join(lines))
    return dump_path


def _append_or_write_report(path: Path, lines: list[str], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines) + "\n"
    if append:
        with path.open("a") as handle:
            handle.write("\n" + content)
    else:
        path.write_text(content)


def _load_batch(sample_index: int, batch_size: int, train: bool) -> tuple[np.ndarray, np.ndarray]:
    images, labels = load_mnist_float(str(REPO_ROOT / "data"), train=train)
    return (
        images[sample_index : sample_index + batch_size],
        labels[sample_index : sample_index + batch_size],
    )


def _build_torch_quantized_model() -> TorchQuantizedMNISTNet:
    checkpoint = torch.load(PTQ_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = TorchQuantizedMNISTNet().train()
    initialize_quantized_model_from_ptq_checkpoint(model, checkpoint)
    return model


def _build_numpy_quantized_model() -> NumpyQuantizedMNISTNet:
    params = make_trainable_params(load_quantized_model_params())
    return NumpyQuantizedMNISTNet(params)


def _numpy_named_weights(model: NumpyQuantizedMNISTNet) -> dict[str, np.ndarray]:
    weights = {}
    for layer in TRAINABLE_LAYERS:
        weights[f"{layer}.weight"] = model.params[layer]["weight"].copy()
        if model.params[layer]["bias"] is not None:
            weights[f"{layer}.bias"] = model.params[layer]["bias"].copy()
    return weights


def _numpy_named_grads(grads: dict) -> dict[str, np.ndarray | None]:
    named = {}
    for layer in TRAINABLE_LAYERS:
        named[f"{layer}.weight"] = grads[layer]["weight"].copy()
        named[f"{layer}.bias"] = None if grads[layer]["bias"] is None else grads[layer]["bias"].copy()
    return named


def _torch_named_weights(model: TorchQuantizedMNISTNet) -> dict[str, np.ndarray]:
    return {
        name: param.detach().cpu().clone().numpy()
        for name, param in model.named_parameters()
    }


def _torch_named_grads(model: TorchQuantizedMNISTNet) -> dict[str, np.ndarray | None]:
    return {
        name: None if param.grad is None else param.grad.detach().cpu().clone().numpy()
        for name, param in model.named_parameters()
    }


def run_forward_compare(sample_index: int = 0, train: bool = False) -> Path:
    torch_model = _build_torch_quantized_model().eval()
    numpy_model = _build_numpy_quantized_model()

    image_batch, label_batch = _load_batch(sample_index=sample_index, batch_size=1, train=train)
    label = int(label_batch[0])

    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(image_batch)).cpu().numpy()[0]
    numpy_logits = numpy_model.forward(image_batch)[0]

    max_abs_diff = float(np.max(np.abs(torch_logits - numpy_logits)))
    lines = [
        _format_kv("compare_type", "forward"),
        _format_kv("sample_index", str(sample_index)),
        _format_kv("label", str(label)),
        _format_kv("torch_pred", str(int(np.argmax(torch_logits)))),
        _format_kv("numpy_pred", str(int(np.argmax(numpy_logits)))),
        _format_kv("max_abs_diff", f"{max_abs_diff:.6f}"),
        _format_kv("image_first10", _tensor_preview(image_batch, 10)),
        _format_kv("torch_logits", _tensor_preview(torch_logits, 10)),
        _format_kv("numpy_logits", _tensor_preview(numpy_logits, 10)),
    ]
    dump_path = _save_report(f"compare_forward_{sample_index:05d}.txt", lines)
    print("\n".join(lines))
    print(_format_kv("dump_path", str(dump_path)))
    return dump_path


def _torch_step_compare(mode: str, image_batch: np.ndarray, label_batch: np.ndarray, lr: float) -> dict:
    model = _build_torch_quantized_model().train()
    if mode == "qas":
        optimizer = QASSGD(model.parameters(), lr=lr, momentum=0.0)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    optimizer.zero_grad()
    weight_before = {
        name: param.detach().cpu().clone().numpy()
        for name, param in model.named_parameters()
    }

    logits_before = model(torch.from_numpy(image_batch))
    loss_before = F.cross_entropy(logits_before, torch.tensor(label_batch, dtype=torch.long))
    loss_before.backward()

    grads = {
        name: (None if param.grad is None else param.grad.detach().cpu().clone().numpy())
        for name, param in model.named_parameters()
    }

    if mode == "qas":
        optimizer.pre_step(model)
        grads_after_pre_step = {
            name: (None if param.grad is None else param.grad.detach().cpu().clone().numpy())
            for name, param in model.named_parameters()
        }
    else:
        grads_after_pre_step = grads

    optimizer.step()
    torch_project_quantized_parameters(model)

    weight_after = {
        name: param.detach().cpu().clone().numpy()
        for name, param in model.named_parameters()
    }
    logits_after = model(torch.from_numpy(image_batch)).detach().cpu().numpy()
    loss_after = float(F.cross_entropy(torch.from_numpy(logits_after), torch.tensor(label_batch, dtype=torch.long)).item())

    return {
        "loss_before": float(loss_before.detach().cpu().item()),
        "loss_after": loss_after,
        "logits_before": logits_before.detach().cpu().numpy(),
        "logits_after": logits_after,
        "weight_before": weight_before,
        "grads": grads,
        "grads_after_pre_step": grads_after_pre_step,
        "weight_after": weight_after,
    }


def _numpy_step_compare(mode: str, image_batch: np.ndarray, label_batch: np.ndarray, lr: float) -> dict:
    model = _build_numpy_quantized_model()

    weight_before = {
        f"{layer}.weight": model.params[layer]["weight"].copy()
        for layer in ("conv1", "conv2", "fc1", "fc2")
    }
    weight_before.update({
        f"{layer}.bias": model.params[layer]["bias"].copy()
        for layer in ("conv1", "conv2", "fc1", "fc2")
        if model.params[layer]["bias"] is not None
    })

    logits_before = model.forward(image_batch)
    loss_before = cross_entropy_loss(logits_before, label_batch)
    grads = model.backward_from_labels(label_batch)

    named_grads = {}
    for layer in ("conv1", "conv2", "fc1", "fc2"):
        named_grads[f"{layer}.weight"] = grads[layer]["weight"].copy()
        if grads[layer]["bias"] is not None:
            named_grads[f"{layer}.bias"] = grads[layer]["bias"].copy()

    if mode == "qas":
        grads = qas_pre_step(model.params, grads)
        named_grads_after_pre = {}
        for layer in ("conv1", "conv2", "fc1", "fc2"):
            named_grads_after_pre[f"{layer}.weight"] = grads[layer]["weight"].copy()
            if grads[layer]["bias"] is not None:
                named_grads_after_pre[f"{layer}.bias"] = grads[layer]["bias"].copy()
    else:
        named_grads_after_pre = named_grads

    sgd_step(model.params, grads, lr=lr)
    numpy_project_quantized_parameters(model.params)

    weight_after = {
        f"{layer}.weight": model.params[layer]["weight"].copy()
        for layer in ("conv1", "conv2", "fc1", "fc2")
    }
    weight_after.update({
        f"{layer}.bias": model.params[layer]["bias"].copy()
        for layer in ("conv1", "conv2", "fc1", "fc2")
        if model.params[layer]["bias"] is not None
    })

    logits_after = model.forward(image_batch)
    loss_after = cross_entropy_loss(logits_after, label_batch)

    return {
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "logits_before": logits_before,
        "logits_after": logits_after,
        "weight_before": weight_before,
        "grads": named_grads,
        "grads_after_pre_step": named_grads_after_pre,
        "weight_after": weight_after,
    }


def run_step_compare(
    mode: str = "quantized",
    sample_index: int = 0,
    batch_size: int = 8,
    train: bool = True,
    lr: float = 1e-3,
) -> Path:
    image_batch, label_batch = _load_batch(sample_index=sample_index, batch_size=batch_size, train=train)

    torch_result = _torch_step_compare(mode=mode, image_batch=image_batch, label_batch=label_batch, lr=lr)
    numpy_result = _numpy_step_compare(mode=mode, image_batch=image_batch, label_batch=label_batch, lr=lr)

    lines = [
        _format_kv("compare_type", "step"),
        _format_kv("mode", mode),
        _format_kv("sample_index", str(sample_index)),
        _format_kv("batch_size", str(batch_size)),
        _format_kv("lr", f"{lr:.6f}"),
        _format_kv("torch_loss_before", _format_float(torch_result["loss_before"])),
        _format_kv("numpy_loss_before", _format_float(numpy_result["loss_before"])),
        _format_kv("loss_before_diff", f"{abs(torch_result['loss_before'] - numpy_result['loss_before']):.6e}"),
        _format_kv("torch_loss_after", _format_float(torch_result["loss_after"])),
        _format_kv("numpy_loss_after", _format_float(numpy_result["loss_after"])),
        _format_kv("loss_after_diff", f"{abs(torch_result['loss_after'] - numpy_result['loss_after']):.6e}"),
        _format_kv(
            "logits_before_diff",
            f"{float(np.max(np.abs(torch_result['logits_before'] - numpy_result['logits_before']))):.6e}",
        ),
        _format_kv(
            "logits_after_diff",
            f"{float(np.max(np.abs(torch_result['logits_after'] - numpy_result['logits_after']))):.6e}",
        ),
        "",
    ]

    for name in ("conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"):
        torch_grad = torch_result["grads"].get(name)
        numpy_grad = numpy_result["grads"].get(name)
        torch_grad_after_pre = torch_result["grads_after_pre_step"].get(name)
        numpy_grad_after_pre = numpy_result["grads_after_pre_step"].get(name)
        torch_weight_after = torch_result["weight_after"].get(name)
        numpy_weight_after = numpy_result["weight_after"].get(name)

        lines.extend([
            f"[{name}]",
            _format_kv(
                "grad_diff",
                "None" if torch_grad is None or numpy_grad is None else f"{float(np.max(np.abs(torch_grad - numpy_grad))):.6e}",
            ),
            _format_kv(
                "grad_after_pre_diff",
                "None"
                if torch_grad_after_pre is None or numpy_grad_after_pre is None
                else f"{float(np.max(np.abs(torch_grad_after_pre - numpy_grad_after_pre))):.6e}",
            ),
            _format_kv(
                "weight_after_diff",
                "None"
                if torch_weight_after is None or numpy_weight_after is None
                else f"{float(np.max(np.abs(torch_weight_after - numpy_weight_after))):.6e}",
            ),
            _format_kv("torch_grad", "None" if torch_grad is None else _tensor_preview(torch_grad, 10, scientific=True)),
            _format_kv("numpy_grad", "None" if numpy_grad is None else _tensor_preview(numpy_grad, 10, scientific=True)),
            _format_kv(
                "torch_grad_after_pre",
                "None" if torch_grad_after_pre is None else _tensor_preview(torch_grad_after_pre, 10, scientific=True),
            ),
            _format_kv(
                "numpy_grad_after_pre",
                "None" if numpy_grad_after_pre is None else _tensor_preview(numpy_grad_after_pre, 10, scientific=True),
            ),
            _format_kv(
                "torch_weight_after",
                "None" if torch_weight_after is None else _tensor_preview(torch_weight_after, 10),
            ),
            _format_kv(
                "numpy_weight_after",
                "None" if numpy_weight_after is None else _tensor_preview(numpy_weight_after, 10),
            ),
            "",
        ])

    dump_path = _save_report(f"compare_step_{mode}.txt", lines)
    print("\n".join(lines))
    print(_format_kv("dump_path", str(dump_path)))
    return dump_path


def _compare_array_max_abs(a: np.ndarray | None, b: np.ndarray | None) -> str:
    if a is None or b is None:
        return "None"
    return f"{float(np.max(np.abs(a - b))):.6e}"


def _extend_train_step_dump(
    lines: list[str],
    name: str,
    torch_before: dict[str, np.ndarray],
    torch_grad: dict[str, np.ndarray | None],
    torch_grad_after_pre: dict[str, np.ndarray | None],
    torch_after: dict[str, np.ndarray],
    numpy_before: dict[str, np.ndarray],
    numpy_grad: dict[str, np.ndarray | None],
    numpy_grad_after_pre: dict[str, np.ndarray | None],
    numpy_after: dict[str, np.ndarray],
) -> None:
    lines.extend([
        f"[{name}]",
        _format_kv("before_diff", _compare_array_max_abs(torch_before.get(name), numpy_before.get(name))),
        _format_kv("grad_diff", _compare_array_max_abs(torch_grad.get(name), numpy_grad.get(name))),
        _format_kv("grad_after_pre_diff", _compare_array_max_abs(torch_grad_after_pre.get(name), numpy_grad_after_pre.get(name))),
        _format_kv("after_diff", _compare_array_max_abs(torch_after.get(name), numpy_after.get(name))),
        "",
    ])


def run_train_compare(
    mode: str = "quantized",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 1e-3,
    momentum: float = 0.0,
    num_workers: int = 0,
    max_steps: int | None = None,
) -> Path:
    torch.manual_seed(0)
    np.random.seed(0)
    train_loader, _ = build_data_loaders(
        str(REPO_ROOT / "data"),
        batch_size=batch_size,
        test_batch_size=batch_size,
        num_workers=num_workers,
    )

    torch_model = _build_torch_quantized_model().train()
    torch_optimizer = QASSGD(torch_model.parameters(), lr=lr, momentum=momentum) if mode == "qas" else torch.optim.SGD(torch_model.parameters(), lr=lr, momentum=momentum)

    numpy_model = _build_numpy_quantized_model()
    numpy_momentum_buffers: dict[str, dict[str, np.ndarray]] = {}

    output_dir = OUTPUT_DIR / f"train_compare_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    last_dump_path = output_dir / "epoch_0000.txt"
    for epoch in range(epochs):
        dump_path = output_dir / f"epoch_{epoch:04d}.txt"
        last_dump_path = dump_path
        for step_in_epoch, (images_t, labels_t) in enumerate(train_loader):
            if max_steps is not None and step_in_epoch >= max_steps:
                break

            image_batch = images_t.detach().cpu().numpy().astype(np.float32, copy=False)
            label_batch = labels_t.detach().cpu().numpy().astype(np.int64, copy=False)

            torch_optimizer.zero_grad()
            torch_weight_before = _torch_named_weights(torch_model)
            torch_logits_before = torch_model(images_t)
            torch_loss_before = F.cross_entropy(torch_logits_before, labels_t)
            torch_loss_before.backward()
            torch_grad = _torch_named_grads(torch_model)
            if mode == "qas":
                torch_optimizer.pre_step(torch_model)
                torch_grad_after_pre = _torch_named_grads(torch_model)
            else:
                torch_grad_after_pre = torch_grad
            torch_optimizer.step()
            torch_project_quantized_parameters(torch_model)
            torch_weight_after = _torch_named_weights(torch_model)
            torch_logits_after = torch_model(images_t).detach().cpu().numpy()

            numpy_weight_before = _numpy_named_weights(numpy_model)
            numpy_logits_before = numpy_model.forward(image_batch)
            numpy_loss_before = cross_entropy_loss(numpy_logits_before, label_batch)
            numpy_grads = numpy_model.backward_from_labels(label_batch)
            numpy_grad = _numpy_named_grads(numpy_grads)
            if mode == "qas":
                numpy_grads = qas_pre_step(numpy_model.params, numpy_grads)
                numpy_grad_after_pre = _numpy_named_grads(numpy_grads)
            else:
                numpy_grad_after_pre = numpy_grad
            sgd_step(
                numpy_model.params,
                numpy_grads,
                lr=lr,
                momentum=momentum,
                momentum_buffers=numpy_momentum_buffers,
            )
            numpy_project_quantized_parameters(numpy_model.params)
            numpy_weight_after = _numpy_named_weights(numpy_model)
            numpy_logits_after = numpy_model.forward(image_batch)

            lines = [
                "=" * 96,
                _format_kv("compare_type", "train_step"),
                _format_kv("mode", mode),
                _format_kv("epoch", str(epoch)),
                _format_kv("step_in_epoch", str(step_in_epoch)),
                _format_kv("global_step", str(global_step)),
                _format_kv("batch_size", str(batch_size)),
                _format_kv("lr", f"{lr:.6f}"),
                _format_kv("momentum", f"{momentum:.6f}"),
                _format_kv("torch_loss_before", _format_float(float(torch_loss_before.detach().cpu().item()))),
                _format_kv("numpy_loss_before", _format_float(numpy_loss_before)),
                _format_kv("loss_before_diff", f"{abs(float(torch_loss_before.detach().cpu().item()) - numpy_loss_before):.6e}"),
                _format_kv("logits_before_diff", f"{float(np.max(np.abs(torch_logits_before.detach().cpu().numpy() - numpy_logits_before))):.6e}"),
                _format_kv("logits_after_diff", f"{float(np.max(np.abs(torch_logits_after - numpy_logits_after))):.6e}"),
                "",
            ]

            for name in TRAINABLE_NAMES:
                _extend_train_step_dump(
                    lines,
                    name,
                    torch_before=torch_weight_before,
                    torch_grad=torch_grad,
                    torch_grad_after_pre=torch_grad_after_pre,
                    torch_after=torch_weight_after,
                    numpy_before=numpy_weight_before,
                    numpy_grad=numpy_grad,
                    numpy_grad_after_pre=numpy_grad_after_pre,
                    numpy_after=numpy_weight_after,
                )

            _append_or_write_report(dump_path, lines, append=step_in_epoch != 0)
            global_step += 1

    print(_format_kv("dump_path", str(last_dump_path)))
    return last_dump_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["forward", "step", "train"], default="forward")
    parser.add_argument("--mode", choices=["quantized", "qas"], default="quantized")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.kind == "forward":
        run_forward_compare(sample_index=args.sample_index, train=args.train)
    elif args.kind == "step":
        run_step_compare(
            mode=args.mode,
            sample_index=args.sample_index,
            batch_size=args.batch_size,
            train=args.train,
            lr=args.lr,
        )
    else:
        run_train_compare(
            mode=args.mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            num_workers=args.num_workers,
            max_steps=args.max_steps,
        )
