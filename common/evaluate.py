import argparse
import json
import sys
from pathlib import Path

import numpy as np
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.constants import (
    DEFAULT_FINETUNE_NUM_WORKERS,
    DEFAULT_FINETUNE_TEST_BATCH_SIZE,
    DEFAULT_NPY_QAS_RUN_ROOT,
    DEFAULT_QAS_RUN_NAME,
    DEFAULT_TORCH_QAS_RUN_ROOT,
)
from common.dataset import DEFAULT_DATA_ROOT, build_data_loaders
from common.train_utils import evaluate_model, load_checkpoint, save_json
from npy_impl.loss import cross_entropy_loss
from npy_impl.model import QasMnistNet as NpyQasMnistNet
from torch_impl.model import QasMnistNet as TorchQasMnistNet

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TORCH_RUN_ROOT = ROOT_DIR / DEFAULT_TORCH_QAS_RUN_ROOT
DEFAULT_NPY_RUN_ROOT = ROOT_DIR / DEFAULT_NPY_QAS_RUN_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate torch/npy QAS runs")
    parser.add_argument("--impl", choices=["torch", "npy"], required=True)
    parser.add_argument("--run-name", default=DEFAULT_QAS_RUN_NAME)
    parser.add_argument("--run-root", default=None)
    return parser.parse_args()


def _run_dir(impl: str, run_name: str, run_root: str | None) -> Path:
    if run_root is not None:
        return Path(run_root) / run_name
    if impl == "torch":
        return DEFAULT_TORCH_RUN_ROOT / run_name
    return DEFAULT_NPY_RUN_ROOT / run_name


def _load_torch_qas_model(run_dir: Path) -> tuple[nn.Module, dict]:
    checkpoint = load_checkpoint(run_dir / "checkpoint.pt")
    if checkpoint.get("mode") != "qas":
        raise ValueError(f"Expected torch qas checkpoint, got mode={checkpoint.get('mode')}")

    model = TorchQasMnistNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, _load_train_result(run_dir)


def _load_numpy_qas_model(run_dir: Path) -> tuple[NpyQasMnistNet, dict]:
    checkpoint = np.load(run_dir / "checkpoint.npz", allow_pickle=False)
    mode = str(checkpoint["mode"])
    if mode != "qas":
        raise ValueError(f"Expected npy qas checkpoint, got mode={mode}")

    params: dict[str, dict[str, np.ndarray]] = {}
    for key in checkpoint.files:
        if key in {"mode", "train_config_json"}:
            continue
        layer_name, param_name = key.split(".", 1)
        params.setdefault(layer_name, {})[param_name] = checkpoint[key]
    return NpyQasMnistNet(params), _load_train_result(run_dir)


def _load_train_result(run_dir: Path) -> dict:
    return json.loads((run_dir / "train_result.json").read_text())


def _evaluate_numpy_model(model: NpyQasMnistNet, test_loader) -> dict[str, float]:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images_t, labels_t in test_loader:
        images = images_t.detach().cpu().numpy().astype(np.float32, copy=False)
        labels = labels_t.detach().cpu().numpy().astype(np.int64, copy=False)
        logits = model.forward(images)
        batch_size = int(labels.shape[0])

        total_loss += cross_entropy_loss(logits, labels) * batch_size
        total_correct += int(np.sum(np.argmax(logits, axis=1) == labels))
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": 100.0 * total_correct / total_samples,
    }


def evaluate_run(impl: str, run_name: str = DEFAULT_QAS_RUN_NAME, run_root: str | None = None) -> dict:
    run_dir = _run_dir(impl, run_name, run_root)
    if impl == "torch":
        model, train_config = _load_torch_qas_model(run_dir)
    else:
        model, train_config = _load_numpy_qas_model(run_dir)

    _, test_loader = build_data_loaders(
        DEFAULT_DATA_ROOT,
        train_config["batch_size"],
        train_config.get("test_batch_size", DEFAULT_FINETUNE_TEST_BATCH_SIZE),
        train_config.get("num_workers", DEFAULT_FINETUNE_NUM_WORKERS),
    )

    metrics = (
        evaluate_model(model, test_loader, nn.CrossEntropyLoss())
        if impl == "torch"
        else _evaluate_numpy_model(model, test_loader)
    )
    result = {
        "impl": impl,
        "run_name": run_name,
        "mode": "qas",
        "test_top1": metrics["top1"],
        "test_loss": metrics["loss"],
    }
    save_json(run_dir / "eval_result.json", result)
    return result


def main() -> None:
    args = parse_args()
    result = evaluate_run(args.impl, args.run_name, args.run_root)
    print(
        f"{result['impl']}:{result['run_name']} "
        f"top1={result['test_top1']:.2f} loss={result['test_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
