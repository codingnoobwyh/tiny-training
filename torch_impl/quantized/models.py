import torch
import torch.nn as nn

from .qas_ops import QASConvReLU2d, QASLinear, QASLinearReLU


class QuantizedMNISTNet(nn.Module):
    # 真实量化训练主线模型。
    # 这里负责的只有两件事：
    # 1. 用量化算子拼出网络
    # 2. 保存 PTQ -> quantized/QAS 训练真正要用的量化状态
    def __init__(self):
        super().__init__()
        self.register_buffer("input_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("input_zero_point", torch.tensor(0.0, dtype=torch.float32))
        self.conv1 = QASConvReLU2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = QASConvReLU2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = QASLinearReLU(32 * 7 * 7, 128)
        self.fc2 = QASLinear(128, 10)

    def quantize_input(self, x: torch.Tensor) -> torch.Tensor:
        # 第一层输入必须先按 PTQ 确定好的输入量化参数编码成码值。
        # 如果这里直接 round 原始 [0,1] 图像，整个真实量化前向都会从第一层开始失真。
        x_q = torch.round(x / self.input_scale) + self.input_zero_point
        return x_q.clamp(-128, 127)

    def dequantize_output(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.fc2.zero_y) * self.fc2.y_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quantize_input(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dequantize_output(x)
        return x

def _quantize_bias_to_int_domain(bias_fp: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    denom = x_scale.to(torch.float32) * w_scale.to(torch.float32)
    return torch.round(bias_fp.to(torch.float32) / denom)


def _copy_buffer_value(dst: torch.Tensor, src: torch.Tensor) -> None:
    dst.copy_(src.to(dst.dtype).reshape(dst.shape))


def initialize_quantized_model_from_ptq_checkpoint(model: QuantizedMNISTNet, checkpoint: dict) -> None:
    state = checkpoint["model_state_dict"]

    quant_scale = state["quant.scale"].to(torch.float32)
    quant_zero = state["quant.zero_point"].to(torch.float32)
    _copy_buffer_value(model.input_scale, quant_scale)
    _copy_buffer_value(model.input_zero_point, quant_zero)

    conv1_w_q = state["conv1.weight"]
    conv1_w_scale = conv1_w_q.q_per_channel_scales().to(torch.float32)
    model.conv1.weight.data = conv1_w_q.int_repr().to(torch.float32)
    model.conv1.bias.data = _quantize_bias_to_int_domain(state["conv1.bias"].detach(), quant_scale, conv1_w_scale)
    _copy_buffer_value(model.conv1.zero_x, quant_zero)
    _copy_buffer_value(model.conv1.zero_y, state["conv1.zero_point"].to(torch.float32))
    _copy_buffer_value(model.conv1.x_scale, quant_scale)
    _copy_buffer_value(model.conv1.y_scale, state["conv1.scale"].to(torch.float32))
    model.conv1.effective_scale.copy_(quant_scale * conv1_w_scale / state["conv1.scale"].to(torch.float32))

    conv2_x_scale = state["conv1.scale"].to(torch.float32)
    conv2_x_zero = state["conv1.zero_point"].to(torch.float32)
    conv2_w_q = state["conv2.weight"]
    conv2_w_scale = conv2_w_q.q_per_channel_scales().to(torch.float32)
    model.conv2.weight.data = conv2_w_q.int_repr().to(torch.float32)
    model.conv2.bias.data = _quantize_bias_to_int_domain(state["conv2.bias"].detach(), conv2_x_scale, conv2_w_scale)
    _copy_buffer_value(model.conv2.zero_x, conv2_x_zero)
    _copy_buffer_value(model.conv2.zero_y, state["conv2.zero_point"].to(torch.float32))
    _copy_buffer_value(model.conv2.x_scale, conv2_x_scale)
    _copy_buffer_value(model.conv2.y_scale, state["conv2.scale"].to(torch.float32))
    model.conv2.effective_scale.copy_(conv2_x_scale * conv2_w_scale / state["conv2.scale"].to(torch.float32))

    fc1_w_q, fc1_bias_fp = state["fc1._packed_params._packed_params"]
    fc1_x_scale = state["conv2.scale"].to(torch.float32)
    fc1_x_zero = state["conv2.zero_point"].to(torch.float32)
    fc1_w_scale = fc1_w_q.q_per_channel_scales().to(torch.float32)
    model.fc1.weight.data = fc1_w_q.int_repr().to(torch.float32)
    model.fc1.bias.data = _quantize_bias_to_int_domain(fc1_bias_fp.detach(), fc1_x_scale, fc1_w_scale)
    _copy_buffer_value(model.fc1.zero_x, fc1_x_zero)
    _copy_buffer_value(model.fc1.zero_y, state["fc1.zero_point"].to(torch.float32))
    _copy_buffer_value(model.fc1.x_scale, fc1_x_scale)
    _copy_buffer_value(model.fc1.y_scale, state["fc1.scale"].to(torch.float32))
    model.fc1.effective_scale.copy_(fc1_x_scale * fc1_w_scale / state["fc1.scale"].to(torch.float32))

    fc2_w_q, fc2_bias_fp = state["fc2._packed_params._packed_params"]
    fc2_x_scale = state["fc1.scale"].to(torch.float32)
    fc2_x_zero = state["fc1.zero_point"].to(torch.float32)
    fc2_w_scale = fc2_w_q.q_per_channel_scales().to(torch.float32)
    model.fc2.weight.data = fc2_w_q.int_repr().to(torch.float32)
    model.fc2.bias.data = _quantize_bias_to_int_domain(fc2_bias_fp.detach(), fc2_x_scale, fc2_w_scale)
    _copy_buffer_value(model.fc2.zero_x, fc2_x_zero)
    _copy_buffer_value(model.fc2.zero_y, state["fc2.zero_point"].to(torch.float32))
    _copy_buffer_value(model.fc2.x_scale, fc2_x_scale)
    _copy_buffer_value(model.fc2.y_scale, state["fc2.scale"].to(torch.float32))
    model.fc2.effective_scale.copy_(fc2_x_scale * fc2_w_scale / state["fc2.scale"].to(torch.float32))
