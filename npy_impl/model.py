"""
MNIST量化前向推理 - 使用简化算子

这个实现完全按照torch_impl/QuantizedMNISTNet的结构,
但使用简化后的算子, 便于调试和理解。
"""

import numpy as np
from .ops import (
    conv2d_3x3_int,
    relu_int,
    maxpool2d_2x2_int,
    linear_int,
    flatten_int,
)


class QuantizedMNISTNet:
    """
    网络结构：
    - 输入量化
    - conv1 (1->16) + relu + pool (28->14)
    - conv2 (16->32) + relu + pool (14->7)
    - flatten (32*7*7 -> 1568)
    - fc1 (1568->128) + relu
    - fc2 (128->10)
    - 输出反量化
    """

    def __init__(self, params):
        """
        初始化模型参数

        Args:
            params: 从PTQ checkpoint加载的量化参数
        """
        # 存储所有量化参数
        self.params = params

        # 输入量化参数从 conv1 的 x_scale 和 zero_x 获取
        self.input_scale = params['conv1']['x_scale']
        self.input_zero_point = params['conv1']['zero_x']

    def quantize_input(self, x_fp: np.ndarray) -> np.ndarray:
        """输入量化: 使用PTQ确定的量化参数"""
        x_q = np.round(x_fp / self.input_scale) + self.input_zero_point
        return x_q.clip(-128, 127).astype(np.int8)

    def dequantize_output(self, x_q: np.ndarray) -> np.ndarray:
        """输出反量化: 使用最后一层的输出量化参数"""
        x_fp = (x_q.astype(np.float32) - self.params['fc2']['zero_y']) * self.params['fc2']['y_scale']
        return x_fp

    def forward(self, x_fp: np.ndarray) -> np.ndarray:
        """
        完整的前向推理流程

        Args:
            x_fp: 浮点输入, shape (N, 1, 28, 28), dtype=float32, range [0, 1]

        Returns:
            logits: 浮点输出, shape (N, 10), dtype=float32
        """
        # 1. 输入量化
        x = self.quantize_input(x_fp)

        # 2. conv1 + relu + pool
        conv1_params = self.params['conv1']
        x = conv2d_3x3_int(
            x,
            conv1_params['weight'],
            conv1_params['bias'],
            conv1_params['zero_x'],
            conv1_params['zero_y'],
            conv1_params['effective_scale']
        )

        x = relu_int(x, conv1_params['zero_y'])
        x = maxpool2d_2x2_int(x)

        # 3. conv2 + relu + pool
        conv2_params = self.params['conv2']
        x = conv2d_3x3_int(
            x,
            conv2_params['weight'],
            conv2_params['bias'],
            conv2_params['zero_x'],
            conv2_params['zero_y'],
            conv2_params['effective_scale']
        )

        x = relu_int(x, conv2_params['zero_y'])
        x = maxpool2d_2x2_int(x)

        # 4. flatten
        x = flatten_int(x)

        # 5. fc1 + relu
        fc1_params = self.params['fc1']
        x = linear_int(
            x,
            fc1_params['weight'],
            fc1_params['bias'],
            fc1_params['zero_x'],
            fc1_params['zero_y'],
            fc1_params['effective_scale']
        )

        x = relu_int(x, fc1_params['zero_y'])

        # 6. fc2
        fc2_params = self.params['fc2']
        x = linear_int(
            x,
            fc2_params['weight'],
            fc2_params['bias'],
            fc2_params['zero_x'],
            fc2_params['zero_y'],
            fc2_params['effective_scale']
        )

        # 7. 输出反量化
        logits = self.dequantize_output(x)
        return logits

    def predict(self, x_fp: np.ndarray) -> np.ndarray:
        """预测类别"""
        logits = self.forward(x_fp)
        return np.argmax(logits, axis=1)
