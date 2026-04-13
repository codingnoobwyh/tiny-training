"""
MNIST量化前向推理 - 使用简化算子

这个实现完全按照torch_impl/QuantizedMNISTNet的结构,
但使用简化后的算子, 便于调试和理解。
"""

import numpy as np
from .loss import cross_entropy_backward
from .ops import (
    conv2d_3x3_int,
    conv2d_3x3_int_backward,
    relu_int,
    relu_int_backward,
    maxpool2d_2x2_int,
    maxpool2d_2x2_int_backward,
    linear_int,
    linear_int_backward,
    flatten_int,
    flatten_int_backward,
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
        self.cache: dict[str, np.ndarray | tuple[int, ...]] = {}

    def quantize_input(self, x_fp: np.ndarray) -> np.ndarray:
        """输入量化: 使用PTQ确定的量化参数"""
        x_q = np.round(x_fp / self.input_scale) + self.input_zero_point
        return x_q.clip(-128, 127).astype(np.int8)

    def dequantize_output(self, x_q: np.ndarray) -> np.ndarray:
        """输出反量化: 使用最后一层的输出量化参数"""
        x_fp = (x_q.astype(np.float32) - self.params['fc2']['zero_y']) * self.params['fc2']['y_scale']
        return x_fp

    def dequantize_output_backward(self, grad_logits: np.ndarray) -> np.ndarray:
        """输出反量化的反向传播。"""
        return grad_logits.astype(np.float32) * self.params['fc2']['y_scale'].astype(np.float32)

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
        self.cache = {
            "x_fp": x_fp.astype(np.float32, copy=False),
            "input_q": x.copy(),
        }

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
        self.cache["conv1_out"] = x.copy()

        x = relu_int(x, conv1_params['zero_y'])
        self.cache["relu1_out"] = x.copy()
        x = maxpool2d_2x2_int(x)
        self.cache["pool1_out"] = x.copy()

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
        self.cache["conv2_out"] = x.copy()

        x = relu_int(x, conv2_params['zero_y'])
        self.cache["relu2_out"] = x.copy()
        x = maxpool2d_2x2_int(x)
        self.cache["pool2_out"] = x.copy()

        # 4. flatten
        self.cache["flatten_input_shape"] = x.shape
        x = flatten_int(x)
        self.cache["flatten_out"] = x.copy()

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
        self.cache["fc1_out"] = x.copy()

        x = relu_int(x, fc1_params['zero_y'])
        self.cache["fc1_relu_out"] = x.copy()

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
        self.cache["fc2_out_q"] = x.copy()

        # 7. 输出反量化
        logits = self.dequantize_output(x)
        self.cache["logits"] = logits.copy()
        return logits

    def backward(self, grad_logits: np.ndarray) -> dict[str, np.ndarray | dict[str, np.ndarray | None]]:
        """
        整网反向传播。
        """
        grad_fc2_out_q = self.dequantize_output_backward(grad_logits)

        grad_fc1_relu_out, grad_fc2_weight, grad_fc2_bias = linear_int_backward(
            grad_fc2_out_q,
            self.cache["fc1_relu_out"],
            self.params["fc2"]["weight"],
            self.params["fc2"]["bias"],
            self.params["fc2"]["zero_x"],
            self.params["fc2"]["effective_scale"],
        )

        grad_fc1_out = relu_int_backward(
            grad_fc1_relu_out,
            self.cache["fc1_out"],
            self.params["fc1"]["zero_y"],
        )

        grad_flatten_out, grad_fc1_weight, grad_fc1_bias = linear_int_backward(
            grad_fc1_out,
            self.cache["flatten_out"],
            self.params["fc1"]["weight"],
            self.params["fc1"]["bias"],
            self.params["fc1"]["zero_x"],
            self.params["fc1"]["effective_scale"],
        )

        grad_pool2_out = flatten_int_backward(
            grad_flatten_out,
            self.cache["flatten_input_shape"],
        )

        grad_relu2_out = maxpool2d_2x2_int_backward(
            grad_pool2_out,
            self.cache["relu2_out"],
        )

        grad_conv2_out = relu_int_backward(
            grad_relu2_out,
            self.cache["conv2_out"],
            self.params["conv2"]["zero_y"],
        )

        grad_pool1_out, grad_conv2_weight, grad_conv2_bias = conv2d_3x3_int_backward(
            grad_conv2_out,
            self.cache["pool1_out"],
            self.params["conv2"]["weight"],
            self.params["conv2"]["bias"],
            self.params["conv2"]["zero_x"],
            self.params["conv2"]["effective_scale"],
        )

        grad_relu1_out = maxpool2d_2x2_int_backward(
            grad_pool1_out,
            self.cache["relu1_out"],
        )

        grad_conv1_out = relu_int_backward(
            grad_relu1_out,
            self.cache["conv1_out"],
            self.params["conv1"]["zero_y"],
        )

        grad_input_q, grad_conv1_weight, grad_conv1_bias = conv2d_3x3_int_backward(
            grad_conv1_out,
            self.cache["input_q"],
            self.params["conv1"]["weight"],
            self.params["conv1"]["bias"],
            self.params["conv1"]["zero_x"],
            self.params["conv1"]["effective_scale"],
        )

        return {
            "grad_input_q": grad_input_q.astype(np.float32),
            "conv1": {"weight": grad_conv1_weight, "bias": grad_conv1_bias},
            "conv2": {"weight": grad_conv2_weight, "bias": grad_conv2_bias},
            "fc1": {"weight": grad_fc1_weight, "bias": grad_fc1_bias},
            "fc2": {"weight": grad_fc2_weight, "bias": grad_fc2_bias},
        }

    def backward_from_labels(self, labels: np.ndarray) -> dict[str, np.ndarray | dict[str, np.ndarray | None]]:
        """
        从当前 forward 的 logits 和标签直接回传。
        """
        grad_logits = cross_entropy_backward(self.cache["logits"], labels)
        return self.backward(grad_logits)

    def predict(self, x_fp: np.ndarray) -> np.ndarray:
        """预测类别"""
        logits = self.forward(x_fp)
        return np.argmax(logits, axis=1)
