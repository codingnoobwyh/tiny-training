"""
简化量化前向算子 - 仅支持Mnist网络结构
"""

import numpy as np
from typing import Optional

from common.constants import INT8_QMIN, INT8_QMAX, INT32_QMIN, INT32_QMAX, TRAINABLE_LAYERS


def round_clip_int8(x: np.ndarray | float | int) -> np.ndarray:
    """四舍五入并裁剪到 int8 范围. """
    x_rounded = np.round(x)
    return x_rounded.clip(INT8_QMIN, INT8_QMAX).astype(np.int8)


def round_clip_int32(x: np.ndarray | float | int) -> np.ndarray:
    """四舍五入并裁剪到 int32 范围. """
    x_rounded = np.round(x)
    return x_rounded.clip(INT32_QMIN, INT32_QMAX).astype(np.int32)


def conv2d_3x3_int(
    x_q: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    zero_x: np.ndarray,
    zero_y: np.ndarray,
    effective_scale: np.ndarray,
) -> np.ndarray:
    """
    固定参数: padding=1, stride=1, dilation=1, groups=1

    Args:
        x_q: 输入整数码值, shape (N, C_in, H, W), dtype=int8
        weight: 权重, shape (C_out, C_in, 3, 3), dtype=int8
        bias: 偏置, shape (C_out,), dtype=int32, 可为 None
        zero_x: 输入零点, dtype=int32
        zero_y: 输出零点, dtype=int32
        effective_scale: 有效尺度, dtype=float32

    Returns:
        输出整数码值, shape (N, C_out, H, W), dtype=int8
    """
    N, C_in, H_in, W_in = x_q.shape
    C_out, _, _, _ = weight.shape

    # 输出尺寸（padding=1时保持尺寸不变）
    H_out = H_in
    W_out = W_in

    # 当前 demo 下 zero_x/zero_y 都是标量；effective_scale 仍按输出通道处理.
    effective_scale_b = effective_scale.reshape(1, -1, 1, 1)

    # 减去输入零点
    x_centered = x_q - zero_x

    # 卷积计算
    output = np.zeros((N, C_out, H_out, W_out), dtype=np.int32)

    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    # 计算卷积和
                    conv_sum = 0
                    for c_in in range(C_in):
                        for kh in range(3):
                            for kw in range(3):
                                # padding=1, 所以索引从0开始
                                h_in = h - 1 + kh
                                w_in = w - 1 + kw

                                if 0 <= h_in < H_in and 0 <= w_in < W_in:
                                    x_value = np.int32(x_centered[n, c_in, h_in, w_in])
                                    w_value = np.int32(weight[c_out, c_in, kh, kw])
                                    conv_sum += x_value * w_value

                    # 加偏置（npy_impl 使用 int32 累加无需 round；torch_impl 使用 float32 累加，
                    # 因此在加偏置前有额外 round 以消除浮点漂移。此处语义等价。）
                    if bias is not None:
                        conv_sum += bias[c_out]

                    # 乘有效尺度
                    output[n, c_out, h, w] = round_clip_int32(
                        conv_sum * effective_scale_b[0, c_out, 0, 0]
                    )

    # 加输出零点并裁剪到 int8
    output = output.astype(np.float32) + zero_y
    return round_clip_int8(output)


def conv2d_3x3_int_backward(
    grad_output: np.ndarray,
    x_q: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    zero_x: np.ndarray,
    effective_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    3x3 same padding 卷积的反向传播.

    当前实现对齐 torch_impl 的 backward 语义：
    - 忽略 round 的导数
    - zero_y 不参与梯度
    - 主链梯度先乘 effective_scale
    """
    N, C_in, H_in, W_in = x_q.shape
    C_out, _, _, _ = weight.shape

    grad_conv_out = grad_output.astype(np.float32) * effective_scale.reshape(1, -1, 1, 1).astype(np.float32)
    x_centered = x_q.astype(np.float32) - np.float32(zero_x)
    weight_float = weight.astype(np.float32)

    grad_input = np.zeros_like(x_centered, dtype=np.float32)
    grad_weight = np.zeros_like(weight_float, dtype=np.float32)
    grad_bias = grad_conv_out.sum(axis=(0, 2, 3)).astype(np.float32) if bias is not None else None

    for n in range(N):
        for c_out in range(C_out):
            for h_out in range(H_in):
                for w_out in range(W_in):
                    grad_value = grad_conv_out[n, c_out, h_out, w_out]
                    for c_in in range(C_in):
                        for kh in range(3):
                            for kw in range(3):
                                h_in_idx = h_out - 1 + kh
                                w_in_idx = w_out - 1 + kw
                                if 0 <= h_in_idx < H_in and 0 <= w_in_idx < W_in:
                                    grad_weight[c_out, c_in, kh, kw] += grad_value * x_centered[n, c_in, h_in_idx, w_in_idx]
                                    grad_input[n, c_in, h_in_idx, w_in_idx] += grad_value * weight_float[c_out, c_in, kh, kw]

    return grad_input, grad_weight, grad_bias


def relu_int(
    x_q: np.ndarray,
    zero_y: np.ndarray,
) -> np.ndarray:
    """
    ReLU激活函数

    Args:
        x_q: 输入整数码值, shape 任意, dtype=int8
        zero_y: 输出零点, dtype=int32

    Returns:
        输出整数码值, shape 与输入相同, dtype=int8
    """
    # ReLU: clamp(x, zero_y, INT8_QMAX)
    output = np.clip(x_q, zero_y, INT8_QMAX)
    return output.astype(np.int8)


def maxpool2d_2x2_int(x_q: np.ndarray) -> np.ndarray:
    """
    2x2最大池化

    Args:
        x_q: 输入整数码值, shape (N, C, H, W), dtype=int8

    Returns:
        输出整数码值, shape (N, C, H_out, W_out), dtype=int8
    """
    N, C, H_in, W_in = x_q.shape

    # 输出尺寸
    H_out = (H_in - 2) // 2 + 1
    W_out = (W_in - 2) // 2 + 1

    # 最大池化
    output = np.zeros((N, C, H_out, W_out), dtype=np.int8)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    # 池化窗口
                    h_start = h_out * 2
                    w_start = w_out * 2

                    # 2x2窗口内元素取最大
                    window_max = max(
                        x_q[n, c, h_start, w_start],
                        x_q[n, c, h_start, w_start + 1],
                        x_q[n, c, h_start + 1, w_start],
                        x_q[n, c, h_start + 1, w_start + 1],
                    )

                    output[n, c, h_out, w_out] = window_max

    return output


def maxpool2d_2x2_int_backward(
    grad_output: np.ndarray,
    x_q: np.ndarray,
) -> np.ndarray:
    """
    2x2 最大池化的反向传播.
    """
    N, C, H_in, W_in = x_q.shape
    _, _, H_out, W_out = grad_output.shape
    grad_input = np.zeros_like(x_q, dtype=np.float32)

    for n in range(N):
        for c in range(C):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    h_start = h_out * 2
                    w_start = w_out * 2

                    window = np.array([
                        x_q[n, c, h_start, w_start],
                        x_q[n, c, h_start, w_start + 1],
                        x_q[n, c, h_start + 1, w_start],
                        x_q[n, c, h_start + 1, w_start + 1],
                    ])
                    max_index = int(np.argmax(window))

                    if max_index == 0:
                        grad_input[n, c, h_start, w_start] += grad_output[n, c, h_out, w_out]
                    elif max_index == 1:
                        grad_input[n, c, h_start, w_start + 1] += grad_output[n, c, h_out, w_out]
                    elif max_index == 2:
                        grad_input[n, c, h_start + 1, w_start] += grad_output[n, c, h_out, w_out]
                    else:
                        grad_input[n, c, h_start + 1, w_start + 1] += grad_output[n, c, h_out, w_out]

    return grad_input


def linear_int(
    x_q: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    zero_x: np.ndarray,
    zero_y: np.ndarray,
    effective_scale: np.ndarray,
) -> np.ndarray:
    """
    全连接层

    Args:
        x_q: 输入整数码值, shape (N, C_in), dtype=int8
        weight: 权重, shape (C_out, C_in), dtype=int8
        bias: 偏置, shape (C_out,), dtype=int32, 可为 None
        zero_x: 输入零点, dtype=int32
        zero_y: 输出零点, dtype=int32
        effective_scale: 有效尺度, dtype=float32

    Returns:
        输出整数码值, shape (N, C_out), dtype=int8
    """
    C_out = weight.shape[0]

    # 当前 demo 下 zero_x/zero_y 都是标量；effective_scale 仍按输出特征处理.
    effective_scale_b = effective_scale.reshape(1, -1)

    # 减去输入零点
    x_centered = x_q.astype(np.int32) - zero_x

    # 线性变换
    output = np.dot(x_centered, weight.astype(np.int32).T)

    # 加偏置（int32 累加无浮点漂移，无需 torch_impl 的额外 round）
    if bias is not None:
        output = output + bias.reshape(1, -1)

    # 乘有效尺度
    output = output.astype(np.float32) * effective_scale_b

    # 加输出零点并裁剪
    output = output + zero_y
    return round_clip_int8(output)


def linear_int_backward(
    grad_output: np.ndarray,
    x_q: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    zero_x: np.ndarray,
    effective_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    全连接层反向传播.
    """
    grad_linear_out = grad_output.astype(np.float32) * effective_scale.reshape(1, -1).astype(np.float32)
    x_centered = x_q.astype(np.float32) - np.float32(zero_x)
    weight_float = weight.astype(np.float32)

    grad_input = grad_linear_out @ weight_float
    grad_weight = grad_linear_out.T @ x_centered
    grad_bias = grad_linear_out.sum(axis=0) if bias is not None else None
    return grad_input.astype(np.float32), grad_weight.astype(np.float32), None if grad_bias is None else grad_bias.astype(np.float32)


def flatten_int(x_q: np.ndarray) -> np.ndarray:
    """
    扁平化

    Args:
        x_q: 输入整数码值, shape (N, C, H, W), dtype=int8

    Returns:
        输出整数码值, shape (N, C*H*W), dtype=int8
    """
    # 扁平化为 (N, C*H*W)
    return x_q.reshape(x_q.shape[0], -1)


def flatten_int_backward(
    grad_output: np.ndarray,
    input_shape: tuple[int, ...],
) -> np.ndarray:
    """
    扁平化反向传播.
    """
    return grad_output.reshape(input_shape)


def relu_int_backward(
    grad_output: np.ndarray,
    x_q: np.ndarray,
    zero_y: np.ndarray,
) -> np.ndarray:
    """
    ReLU 反向传播.
    """
    mask = (x_q >= zero_y) & (x_q <= INT8_QMAX)
    return (grad_output * mask.astype(np.float32)).astype(np.float32)


def make_trainable_params(params: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, np.ndarray]]:
    """
    将 PTQ 提取出的参数转换成可训练参数.

    weight / bias 用 float32 存储, 其它量化元信息保持原 dtype.
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
    对齐 torch.optim.SGD 的最小更新路径.
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
    QAS 梯度重标定, 对齐 torch_impl.operators.QASSGD.pre_step.
    """
    scaled_grads: dict[str, dict[str, np.ndarray] | np.ndarray] = {
        "grad_input_q": grads["grad_input_q"],
    }

    for layer_name in TRAINABLE_LAYERS:
        layer_params = params[layer_name]
        layer_grads = grads[layer_name]

        w_scale = np.asarray(layer_params["w_scale"], dtype=np.float32)
        x_scale = np.asarray(layer_params["x_scale"], dtype=np.float32)

        weight_grad = layer_grads["weight"].astype(np.float32, copy=True)
        view_shape = (w_scale.shape[0],) + (1,) * (weight_grad.ndim - 1)
        weight_grad /= (w_scale.reshape(view_shape) ** 2)

        bias_grad = layer_grads["bias"].astype(np.float32, copy=True)
        bias_grad /= ((x_scale * w_scale) ** 2)

        scaled_grads[layer_name] = {
            "weight": weight_grad,
            "bias": bias_grad,
        }

    return scaled_grads


def project_quantized_parameters(params: dict[str, dict[str, np.ndarray]]) -> None:
    """
    将训练后的参数投影回合法量化域.
    """
    for layer_name in TRAINABLE_LAYERS:
        params[layer_name]["weight"] = np.clip(
            np.round(params[layer_name]["weight"]),
            INT8_QMIN,
            INT8_QMAX,
        ).astype(np.float32)

        params[layer_name]["bias"] = np.clip(
            np.round(params[layer_name]["bias"]),
            INT32_QMIN,
            INT32_QMAX,
        ).astype(np.float32)
