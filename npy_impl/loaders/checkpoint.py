"""
从现有的 PTQ checkpoint 中, 把 numpy_impl 前向真正要用的量化参数提取出来.
"""

from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PTQ_CHECKPOINT_PATH = PROJECT_ROOT / "torch_impl" / "artifacts" / "runs" / "ptq" / "checkpoint.pt"


def extract_quantization_params(state_dict) -> dict[str, dict[str, np.ndarray]]:
    """
    从 PyTorch state_dict 中提取量化参数

    返回格式：
    {
        "conv1": {"weight": int8, "bias": int32, "x_scale": float32, "y_scale": float32,
                  "zero_x": int32, "zero_y": int32, "w_scale": float32, "effective_scale": float32},
        "conv2": {...},
        "fc1": {...},
        "fc2": {...}
    }
    """
    params = {}

    # -----------------------------
    # conv1
    # -----------------------------
    conv1_w_q = state_dict["conv1.weight"]
    conv1_w_scale = conv1_w_q.q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
    conv1_w_int = conv1_w_q.int_repr().cpu().numpy().astype(np.int8)
    conv1_bias_fp = state_dict["conv1.bias"].detach().cpu().numpy().astype(np.float32)

    # bias 在 PTQ checkpoint 里是 float32, 转换为整数累加域
    # bias_int = round(bias_fp / (x_scale * w_scale))
    conv1_bias_int = np.round(conv1_bias_fp / (state_dict["quant.scale"].detach().cpu().numpy().astype(np.float32) * conv1_w_scale)).astype(np.int32)

    params["conv1"] = {
        "weight": conv1_w_int,
        "bias": conv1_bias_int,
        "x_scale": state_dict["quant.scale"].detach().cpu().numpy().astype(np.float32),  # 输入量化尺度
        "y_scale": state_dict["conv1.scale"].detach().cpu().numpy().astype(np.float32),  # 输出量化尺度
        "zero_x": state_dict["quant.zero_point"].detach().cpu().numpy().astype(np.int32),  # 输入零点
        "zero_y": state_dict["conv1.zero_point"].detach().cpu().numpy().astype(np.int32),  # 输出零点
        "w_scale": conv1_w_scale,  # 权重量化尺度
        "effective_scale": (state_dict["quant.scale"].detach().cpu().numpy().astype(np.float32) * conv1_w_scale /
                            state_dict["conv1.scale"].detach().cpu().numpy().astype(np.float32)),  # 有效尺度
    }

    # -----------------------------
    # conv2
    # -----------------------------
    # conv2 的输入来自 conv1 的输出, 因此：
    # - conv2.x_scale 来自 conv1.scale
    # - conv2.zero_x 来自 conv1.zero_point
    conv2_w_q = state_dict["conv2.weight"]
    conv2_w_scale = conv2_w_q.q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
    conv2_w_int = conv2_w_q.int_repr().cpu().numpy().astype(np.int8)
    conv2_bias_fp = state_dict["conv2.bias"].detach().cpu().numpy().astype(np.float32)
    conv2_x_scale = state_dict["conv1.scale"].detach().cpu().numpy().astype(np.float32)
    conv2_x_zero = state_dict["conv1.zero_point"].detach().cpu().numpy().astype(np.int32)
    conv2_bias_int = np.round(conv2_bias_fp / (conv2_x_scale * conv2_w_scale)).astype(np.int32)

    params["conv2"] = {
        "weight": conv2_w_int,
        "bias": conv2_bias_int,
        "x_scale": conv2_x_scale,
        "y_scale": state_dict["conv2.scale"].detach().cpu().numpy().astype(np.float32),
        "zero_x": conv2_x_zero,
        "zero_y": state_dict["conv2.zero_point"].detach().cpu().numpy().astype(np.int32),
        "w_scale": conv2_w_scale,
        "effective_scale": (conv2_x_scale * conv2_w_scale /
                            state_dict["conv2.scale"].detach().cpu().numpy().astype(np.float32)),
    }

    # -----------------------------
    # fc1
    # -----------------------------
    # Linear 层在 PTQ checkpoint 里不是直接平铺出来的,
    # 而是打包在 _packed_params._packed_params 中.
    fc1_w_q, fc1_bias_fp = state_dict["fc1._packed_params._packed_params"]
    fc1_w_scale = fc1_w_q.q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
    fc1_w_int = fc1_w_q.int_repr().cpu().numpy().astype(np.int8)
    fc1_x_scale = state_dict["conv2.scale"].detach().cpu().numpy().astype(np.float32)
    fc1_x_zero = state_dict["conv2.zero_point"].detach().cpu().numpy().astype(np.int32)
    fc1_bias_int = np.round(fc1_bias_fp.detach().cpu().numpy().astype(np.float32) /
                           (fc1_x_scale * fc1_w_scale)).astype(np.int32)

    params["fc1"] = {
        "weight": fc1_w_int,
        "bias": fc1_bias_int,
        "x_scale": fc1_x_scale,
        "y_scale": state_dict["fc1.scale"].detach().cpu().numpy().astype(np.float32),
        "zero_x": fc1_x_zero,
        "zero_y": state_dict["fc1.zero_point"].detach().cpu().numpy().astype(np.int32),
        "w_scale": fc1_w_scale,
        "effective_scale": (fc1_x_scale * fc1_w_scale /
                            state_dict["fc1.scale"].detach().cpu().numpy().astype(np.float32)),
    }

    # -----------------------------
    # fc2
    # -----------------------------
    fc2_w_q, fc2_bias_fp = state_dict["fc2._packed_params._packed_params"]
    fc2_w_scale = fc2_w_q.q_per_channel_scales().detach().cpu().numpy().astype(np.float32)
    fc2_w_int = fc2_w_q.int_repr().cpu().numpy().astype(np.int8)
    fc2_x_scale = state_dict["fc1.scale"].detach().cpu().numpy().astype(np.float32)
    fc2_x_zero = state_dict["fc1.zero_point"].detach().cpu().numpy().astype(np.int32)
    fc2_bias_int = np.round(fc2_bias_fp.detach().cpu().numpy().astype(np.float32) /
                           (fc2_x_scale * fc2_w_scale)).astype(np.int32)

    params["fc2"] = {
        "weight": fc2_w_int,
        "bias": fc2_bias_int,
        "x_scale": fc2_x_scale,
        "y_scale": state_dict["fc2.scale"].detach().cpu().numpy().astype(np.float32),
        "zero_x": fc2_x_zero,
        "zero_y": state_dict["fc2.zero_point"].detach().cpu().numpy().astype(np.int32),
        "w_scale": fc2_w_scale,
        "effective_scale": (fc2_x_scale * fc2_w_scale /
                            state_dict["fc2.scale"].detach().cpu().numpy().astype(np.float32)),
    }

    return params


def load_quantized_model_params(checkpoint_path: str | Path = PTQ_CHECKPOINT_PATH) -> dict[str, dict[str, np.ndarray]]:
    """
    加载量化模型参数

    Args:
        checkpoint_path: PTQ checkpoint 文件路径

    Returns:
        量化参数字典
    """
    # 这是 numpy_impl 读取量化起点的唯一入口
    # 后续前向实现应当直接消费这个函数的输出, 而不是自己再解析 PyTorch checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PTQ checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if checkpoint.get("mode") != "ptq":
        raise ValueError(f"Expected mode='ptq', got mode='{checkpoint.get('mode')}'")

    # 真正给 numpy_impl 使用的是提取后的分层参数结构, 而不是原始 state_dict
    return extract_quantization_params(checkpoint["model_state_dict"])
