"""
NumPy 版最小参数更新.
"""

import numpy as np

from common.constants import INT8_QMAX, INT8_QMIN, INT32_QMAX, INT32_QMIN, TRAINABLE_LAYERS


def make_trainable_params(params: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """
    将 PTQ 提取出的参数转换成可训练参数.

    约定：
    - weight / bias 使用 float32 存储, 便于梯度更新
    - 其它量化元信息保持原始 dtype
    """
    trainable_params: dict[str, dict[str, np.ndarray]] = {}
    for layer_name, layer_params in params.items():
        copied_layer: dict[str, np.ndarray] = {}
        for key, value in layer_params.items():
            copied_value = value.copy() if isinstance(value, np.ndarray) else np.array(value, copy=True)
            if key in ("weight", "bias") and copied_value is not None:
                copied_value = copied_value.astype(np.float32, copy=False)
            copied_layer[key] = copied_value
        trainable_params[layer_name] = copied_layer
    return trainable_params


def sgd_step(
    params: dict[str, dict[str, np.ndarray]],
    grads: dict[str, dict[str, np.ndarray] | np.ndarray],
    lr: float,
    momentum: float = 0.0,
    momentum_buffers: dict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    """
    最小 SGD 更新.
    """
    lr = float(lr)
    momentum = float(momentum)
    for layer_name in TRAINABLE_LAYERS:
        layer_grads = grads[layer_name]

        weight_grad = layer_grads["weight"].astype(np.float32)
        if momentum_buffers is not None and momentum != 0.0:
            weight_buffer = momentum_buffers.setdefault(layer_name, {}).setdefault(
                "weight",
                np.zeros_like(params[layer_name]["weight"], dtype=np.float32),
            )
            weight_buffer *= momentum
            weight_buffer += weight_grad
            weight_grad = weight_buffer
        params[layer_name]["weight"] -= lr * weight_grad

        if params[layer_name]["bias"] is not None and layer_grads["bias"] is not None:
            bias_grad = layer_grads["bias"].astype(np.float32)
            if momentum_buffers is not None and momentum != 0.0:
                bias_buffer = momentum_buffers.setdefault(layer_name, {}).setdefault(
                    "bias",
                    np.zeros_like(params[layer_name]["bias"], dtype=np.float32),
                )
                bias_buffer *= momentum
                bias_buffer += bias_grad
                bias_grad = bias_buffer
            params[layer_name]["bias"] -= lr * bias_grad


def qas_pre_step(
    params: dict[str, dict[str, np.ndarray]],
    grads: dict[str, dict[str, np.ndarray] | np.ndarray],
) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    """
    QAS 梯度重标定.

    沿用 torch_impl 的同一公式：
    - effective_scale = x_scale * w_scale / y_scale
    - w_scale = effective_scale * y_scale / x_scale
    - grad_w <- grad_w / (w_scale^2)
    - grad_b <- grad_b / ((effective_scale * y_scale)^2)
    """
    scaled_grads: dict[str, dict[str, np.ndarray] | np.ndarray] = {
        "grad_input_q": grads["grad_input_q"],
    }

    for layer_name in TRAINABLE_LAYERS:
        layer_params = params[layer_name]
        layer_grads = grads[layer_name]

        x_scale = np.asarray(layer_params["x_scale"], dtype=np.float32)
        y_scale = np.asarray(layer_params["y_scale"], dtype=np.float32)
        effective_scale = np.asarray(layer_params["effective_scale"], dtype=np.float32)
        w_scale = (effective_scale * y_scale / x_scale).astype(np.float32)

        weight_grad = layer_grads["weight"].astype(np.float32, copy=True)
        view_shape = (w_scale.shape[0],) + (1,) * (weight_grad.ndim - 1)
        weight_grad /= (w_scale.reshape(view_shape) ** 2)

        bias_grad = None
        if layer_grads["bias"] is not None:
            bias_grad = layer_grads["bias"].astype(np.float32, copy=True)
            bias_grad /= ((effective_scale * y_scale) ** 2)

        scaled_grads[layer_name] = {
            "weight": weight_grad,
            "bias": bias_grad,
        }

    return scaled_grads


def project_quantized_parameters(params: dict[str, dict[str, np.ndarray]]) -> None:
    """
    将训练后的参数投影回合法量化域.

    约定：
    - weight: round + clamp 到 int8 范围, 但仍以 float32 存储
    - bias: round + clamp 到 int32 范围, 但仍以 float32 存储
    """
    for layer_name in TRAINABLE_LAYERS:
        weight = params[layer_name]["weight"]
        params[layer_name]["weight"] = np.clip(
            np.round(weight),
            INT8_QMIN,
            INT8_QMAX,
        ).astype(np.float32)

        bias = params[layer_name]["bias"]
        if bias is not None:
            params[layer_name]["bias"] = np.clip(
                np.round(bias),
                INT32_QMIN,
                INT32_QMAX,
            ).astype(np.float32)
