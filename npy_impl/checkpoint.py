"""
PTQ checkpoint 到 NumPy 参数结构的转换入口.
"""

from pathlib import Path

import numpy as np
import torch

from common.constants import DEFAULT_PTQ_CHECKPOINT_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PTQ_CHECKPOINT_PATH = PROJECT_ROOT / DEFAULT_PTQ_CHECKPOINT_PATH


def _to_numpy(value: torch.Tensor, dtype) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype)


def _bias_to_int_domain(bias_fp: torch.Tensor, x_scale: torch.Tensor, w_scale: np.ndarray) -> np.ndarray:
    bias = _to_numpy(bias_fp, np.float32)
    x_scale_np = _to_numpy(x_scale, np.float32)
    return np.round(bias / (x_scale_np * w_scale)).astype(np.int32)


def _layer_params(
    weight_q: torch.Tensor,
    bias_fp: torch.Tensor,
    x_scale: torch.Tensor,
    x_zero: torch.Tensor,
    y_scale: torch.Tensor,
    y_zero: torch.Tensor,
) -> dict[str, np.ndarray]:
    w_scale = _to_numpy(weight_q.q_per_channel_scales(), np.float32)
    y_scale_np = _to_numpy(y_scale, np.float32)
    x_scale_np = _to_numpy(x_scale, np.float32)

    return {
        "weight": weight_q.int_repr().cpu().numpy().astype(np.int8),
        "bias": _bias_to_int_domain(bias_fp, x_scale, w_scale),
        "x_scale": x_scale_np,
        "y_scale": y_scale_np,
        "zero_x": _to_numpy(x_zero, np.int32),
        "zero_y": _to_numpy(y_zero, np.int32),
        "w_scale": w_scale,
        "effective_scale": (x_scale_np * w_scale / y_scale_np).astype(np.float32),
    }


def extract_quantization_params(state: dict) -> dict[str, dict[str, np.ndarray]]:
    input_scale = state["quant.scale"]
    input_zero = state["quant.zero_point"]
    fc1_weight_q, fc1_bias_fp = state["fc1._packed_params._packed_params"]
    fc2_weight_q, fc2_bias_fp = state["fc2._packed_params._packed_params"]

    return {
        "conv1": _layer_params(
            weight_q=state["conv1.weight"],
            bias_fp=state["conv1.bias"],
            x_scale=input_scale,
            x_zero=input_zero,
            y_scale=state["conv1.scale"],
            y_zero=state["conv1.zero_point"],
        ),
        "conv2": _layer_params(
            weight_q=state["conv2.weight"],
            bias_fp=state["conv2.bias"],
            x_scale=state["conv1.scale"],
            x_zero=state["conv1.zero_point"],
            y_scale=state["conv2.scale"],
            y_zero=state["conv2.zero_point"],
        ),
        "fc1": _layer_params(
            weight_q=fc1_weight_q,
            bias_fp=fc1_bias_fp,
            x_scale=state["conv2.scale"],
            x_zero=state["conv2.zero_point"],
            y_scale=state["fc1.scale"],
            y_zero=state["fc1.zero_point"],
        ),
        "fc2": _layer_params(
            weight_q=fc2_weight_q,
            bias_fp=fc2_bias_fp,
            x_scale=state["fc1.scale"],
            x_zero=state["fc1.zero_point"],
            y_scale=state["fc2.scale"],
            y_zero=state["fc2.zero_point"],
        ),
    }


def load_quantized_model_params(checkpoint_path: str | Path = PTQ_CHECKPOINT_PATH) -> dict[str, dict[str, np.ndarray]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("mode") != "ptq":
        raise ValueError(f"Expected mode='ptq', got mode='{checkpoint.get('mode')}'")
    return extract_quantization_params(checkpoint["model_state_dict"])
