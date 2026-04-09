"""
简化量化前向算子 - 仅支持MNIST网络结构
"""

import numpy as np
from typing import Optional


INT8_QMIN = -128
INT8_QMAX = 127
INT32_QMIN = -2147483648
INT32_QMAX = 2147483647


def round_clip_int8(x: np.ndarray | float | int) -> np.ndarray:
    """四舍五入并裁剪到 int8 范围。"""
    x_rounded = np.round(x)
    return x_rounded.clip(INT8_QMIN, INT8_QMAX).astype(np.int8)


def round_clip_int32(x: np.ndarray | float | int) -> np.ndarray:
    """四舍五入并裁剪到 int32 范围。"""
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

    # 当前 demo 下 zero_x/zero_y 都是标量；effective_scale 仍按输出通道处理。
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

                    # 加偏置
                    if bias is not None:
                        conv_sum += bias[c_out]

                    # 乘有效尺度
                    output[n, c_out, h, w] = round_clip_int32(
                        conv_sum * effective_scale_b[0, c_out, 0, 0]
                    )

    # 加输出零点并裁剪到 int8
    output = output.astype(np.float32) + zero_y
    return round_clip_int8(output)


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
    # ReLU: max(x, zero_y)
    output = np.maximum(x_q, zero_y)
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

    # 当前 demo 下 zero_x/zero_y 都是标量；effective_scale 仍按输出特征处理。
    effective_scale_b = effective_scale.reshape(1, -1)

    # 减去输入零点
    x_centered = x_q.astype(np.int32) - zero_x

    # 线性变换
    output = np.dot(x_centered, weight.astype(np.int32).T)

    # 加偏置
    if bias is not None:
        output = output + bias.reshape(1, -1)

    # 乘有效尺度
    output = output.astype(np.float32) * effective_scale_b

    # 加输出零点并裁剪
    output = output + zero_y
    return round_clip_int8(output)


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
