"""
对拍 torch_impl 与 npy_impl 的量化前向结果，并将结果保存为文本文件。
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from mnist_data import load_mnist_float
from npy_impl import QuantizedMNISTNet as NumpyQuantizedMNISTNet, load_quantized_model_params
from torch_impl.quantized.models import (
    QuantizedMNISTNet as TorchQuantizedMNISTNet,
    initialize_quantized_model_from_ptq_checkpoint,
)


PTQ_CHECKPOINT_PATH = REPO_ROOT / "torch_impl" / "artifacts" / "runs" / "ptq" / "checkpoint.pt"
OUTPUT_DIR = REPO_ROOT / "npy_impl" / "artifacts"


def _tensor_preview(array, limit: int = 10) -> str:
    arr = np.asarray(array).reshape(-1)
    values = arr[:limit].tolist()

    def _format_value(value) -> str:
        width = 12
        if isinstance(value, (bool, np.bool_)):
            return f"{int(value):>{width}d}"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):>{width}d}"
        value = float(value)
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value)):>{width}d}"
        return f"{value:>{width}.4f}"

    return "[" + ", ".join(_format_value(v) for v in values) + "]"


def _format_kv(key: str, value: str) -> str:
    return f"{key:<20}: {value}"


def save_compare_txt(
    sample_index: int,
    label: int,
    image: np.ndarray,
    torch_logits: np.ndarray,
    numpy_logits: np.ndarray,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = OUTPUT_DIR / f"compare_sample_{sample_index:05d}.txt"

    torch_pred = int(np.argmax(torch_logits))
    numpy_pred = int(np.argmax(numpy_logits))
    max_abs_diff = float(np.max(np.abs(torch_logits - numpy_logits)))

    lines = [
        _format_kv("sample_index", str(sample_index)),
        _format_kv("label", str(label)),
        _format_kv("torch_pred", str(torch_pred)),
        _format_kv("numpy_pred", str(numpy_pred)),
        _format_kv("max_abs_diff", f"{max_abs_diff:.6f}"),
        _format_kv("image_first10", _tensor_preview(image, 10)),
        _format_kv("torch_logits", _tensor_preview(torch_logits, 10)),
        _format_kv("numpy_logits", _tensor_preview(numpy_logits, 10)),
    ]
    dump_path.write_text("\n".join(lines))
    return dump_path


def run_forward_compare(sample_index: int = 0, train: bool = False) -> Path:
    checkpoint = torch.load(PTQ_CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    torch_quantized_model = TorchQuantizedMNISTNet().eval()
    initialize_quantized_model_from_ptq_checkpoint(torch_quantized_model, checkpoint)

    numpy_quantized_model = NumpyQuantizedMNISTNet(load_quantized_model_params())

    images, labels = load_mnist_float(str(REPO_ROOT / "data"), train=train)
    image = images[sample_index : sample_index + 1]
    label = int(labels[sample_index])

    with torch.no_grad():
        torch_logits = torch_quantized_model(torch.from_numpy(image)).cpu().numpy()[0]
    numpy_logits = numpy_quantized_model.forward(image)[0]

    dump_path = save_compare_txt(
        sample_index=sample_index,
        label=label,
        image=image,
        torch_logits=torch_logits,
        numpy_logits=numpy_logits,
    )

    print(_format_kv("sample_index", str(sample_index)))
    print(_format_kv("label", str(label)))
    print(_format_kv("torch_pred", str(int(np.argmax(torch_logits)))))
    print(_format_kv("numpy_pred", str(int(np.argmax(numpy_logits)))))
    print(_format_kv("max_abs_diff", f"{float(np.max(np.abs(torch_logits - numpy_logits))):.6f}"))
    print(_format_kv("dump_path", str(dump_path)))
    return dump_path


if __name__ == "__main__":
    run_forward_compare(sample_index=0, train=False)
