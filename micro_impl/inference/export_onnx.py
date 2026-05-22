import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if __package__ in {None, ""}:
    sys.path.insert(0, str(REPO_ROOT))

from common.baseline.models import FloatMnistNet
from common.dataset import DEFAULT_DATA_ROOT, export_mnist_fp32_payload
from common.train_utils import load_checkpoint


DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "common/baseline/artifacts/baseline/checkpoint.pt"
DEFAULT_MICRO_DIR = SCRIPT_DIR / "artifacts"
DEFAULT_ONNX_PATH = DEFAULT_MICRO_DIR / "model.onnx"
DEFAULT_DATA_DIR = DEFAULT_MICRO_DIR / "data"
DEFAULT_GOLDEN_PATH = DEFAULT_MICRO_DIR / "golden.txt"
DEFAULT_CONFIG_PATH = DEFAULT_MICRO_DIR / "micro.cfg"
DEFAULT_QUANT_CONFIG_PATH = DEFAULT_MICRO_DIR / "micro_quant.cfg"
DEFAULT_DATA_COUNT = 50
DEFAULT_MICRO_CONFIG = """[micro_param]
enable_micro=true
target=RISCV
support_parallel=false
"""


def ensure_micro_config(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    if config_path.exists():
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_MICRO_CONFIG)


def write_quant_config(config_path: Path = DEFAULT_QUANT_CONFIG_PATH, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        DEFAULT_MICRO_CONFIG
        + f"""
[common_quant_param]
quant_type=FULL_QUANT
bit_num=8

[data_preprocess_param]
calibrate_path=input:{data_dir}
calibrate_size={DEFAULT_DATA_COUNT}
input_type=BIN

[full_quant_param]
activation_quant_method=MAX_MIN
bias_correction=true
enable_all_ops=true
"""
    )


def export_float_onnx_model(checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH, output_path: Path = DEFAULT_ONNX_PATH) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["mode"] != "float":
        raise ValueError(f"onnx export expects float checkpoint, got {checkpoint['mode']}")

    model = FloatMnistNet()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
        opset_version=17,
    )


def export_onnx_golden(
    onnx_path: Path = DEFAULT_ONNX_PATH,
    input_path: Path = DEFAULT_DATA_DIR / "test_00000.bin",
    output_path: Path = DEFAULT_GOLDEN_PATH,
) -> None:
    image = np.fromfile(input_path, dtype=np.float32).reshape(1, 1, 28, 28)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: image})[0].astype(np.float32, copy=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shape = " ".join(str(dim) for dim in output.shape)
    values = " ".join(f"{value:.9g}" for value in output.reshape(-1))
    output_path.write_text(f"logits {output.ndim} {shape}\n{values}\n")


def export_micro_artifacts() -> None:
    ensure_micro_config()
    export_float_onnx_model()
    export_mnist_fp32_payload(DEFAULT_DATA_ROOT, DEFAULT_DATA_DIR, train=False, count=DEFAULT_DATA_COUNT)
    write_quant_config()
    export_onnx_golden()


def main() -> None:
    export_micro_artifacts()
    print(f"ONNX exported to {DEFAULT_ONNX_PATH}")
    print(f"Input payloads exported to {DEFAULT_DATA_DIR}")
    print(f"Golden output exported to {DEFAULT_GOLDEN_PATH}")
    print(f"Quant config exported to {DEFAULT_QUANT_CONFIG_PATH}")


if __name__ == "__main__":
    main()
