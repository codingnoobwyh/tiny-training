import copy
import torch
import torch.nn as nn
import torch.ao.quantization as tq

from .qas_ops import QASConvReLU2d, QASLinear, QASLinearReLU


class FloatMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 这是浮点训练的标杆模型，也是原生 QAT 的母体结构。
        # 原生 eager-mode QAT 需要先对 Conv/ReLU、Linear/ReLU 做模块融合，
        # 所以这里把每个 ReLU 都拆成独立成员，方便后面 fuse。
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一个卷积块把 28x28 的单通道图像编码成更丰富的 16 通道特征。
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # 第二个卷积块进一步提取局部模式，同时把空间尺寸压到 7x7。
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # 卷积特征图先展平，再接两层全连接分类头。
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)

        # 最后一层直接输出 logits，不额外做 softmax。
        # 因为训练时 CrossEntropyLoss 内部已经处理了这部分。
        x = self.fc2(x)
        return x


class QuantizableMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 这是原生 PyTorch eager-mode quantization 可复用的母体结构。
        # 1. 模型主体仍然是普通 Conv/Linear
        # 2. 额外插入 QuantStub / DeQuantStub
        # 3. 后续既可以走 prepare_qat，也可以走 prepare 做 PTQ 校准
        self.quant = tq.QuantStub()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.dequant = tq.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # prepare_qat / prepare 之后，quant/dequant 之间的算子会自动带上 fake quant 或 observer。
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        # 原生 eager-mode QAT 需要先融合可融合模块。
        tq.fuse_modules(
            self,
            [["conv1", "relu1"], ["conv2", "relu2"], ["fc1", "relu3"]],
            inplace=True,
        )


class QuantizedMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 这条模型线服务于“real quantized training”。
        # 卷积和第一层全连接都融合了量化版 ReLU，最后分类层保持线性输出 logits。
        self.conv1 = QASConvReLU2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = QASConvReLU2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = QASLinearReLU(32 * 7 * 7, 128)
        self.fc2 = QASLinear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def build_native_qat_model() -> nn.Module:
    # 构建“训练态”的原生 QAT 模型，而不是最终 convert 后的量化推理模型。
    model = QuantizableMNISTNet()
    model.fuse_model()
    model.qconfig = tq.get_default_qat_qconfig("fbgemm")
    return tq.prepare_qat(model.train(), inplace=False)


def convert_qat_model_for_inference(model: nn.Module) -> nn.Module:
    # 如需评估真正量化后的模型，需要先把训练态 QAT 模型 convert 成推理模型。
    converted = copy.deepcopy(model).eval()
    return tq.convert(converted, inplace=False)


def build_native_ptq_prepare_model() -> nn.Module:
    # 构建“校准态”的原生 PTQ 模型。
    # 它会插入 observer，但不会插入 QAT 用的 fake quant 训练逻辑。
    model = QuantizableMNISTNet()
    model.fuse_model()
    model.qconfig = tq.get_default_qconfig("fbgemm")
    return tq.prepare(model.eval(), inplace=False)


def build_native_ptq_prepare_model_from_float_state_dict(float_state_dict: dict[str, torch.Tensor]) -> nn.Module:
    # PTQ 必须从一份已经训练好的浮点参数出发。
    # 这里不能先 prepare 再用 strict=False 去加载，
    # 因为 fuse 之后模块键名会从 conv1.weight 变成 conv1.0.weight，
    # 那样会把真正的卷积/线性权重大量漏掉，最后量化模型接近随机初始化。
    model = QuantizableMNISTNet()
    model.load_state_dict(float_state_dict, strict=True)
    model.fuse_model()
    model.qconfig = tq.get_default_qconfig("fbgemm")
    return tq.prepare(model.eval(), inplace=False)


def build_native_ptq_converted_model() -> nn.Module:
    # 构建一个已经 convert 过的 PTQ 推理模型骨架，供后续加载量化 state_dict 使用。
    prepared = build_native_ptq_prepare_model()
    return tq.convert(prepared, inplace=False)


def _quantize_bias_to_int_domain(bias_fp: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    # 自定义真实量化前向在 bias 分支里使用的是“整数累加域 bias”语义。
    # 因此这里需要把 PTQ 里保存的浮点 bias 重新映射到 int32 累加域，
    # 只是仍用 float tensor 承载这些整数值，方便继续训练。
    denom = x_scale.to(torch.float32) * w_scale.to(torch.float32)
    return torch.round(bias_fp.to(torch.float32) / denom)


def _copy_buffer_value(dst: torch.Tensor, src: torch.Tensor) -> None:
    # 原生 PTQ checkpoint 里的 scale / zero_point 有时是 shape=[1]，
    # 我们自定义层里则更倾向于把标量保存成 0-d tensor。
    # 这里统一按目标 buffer 的形状重排后再 copy，避免广播报错。
    dst.copy_(src.to(dst.dtype).reshape(dst.shape))


def initialize_quantized_model_from_ptq_checkpoint(model: nn.Module, checkpoint: dict) -> None:
    # 这里把 PyTorch 原生 PTQ checkpoint 中的量化结果，映射到我们自定义的真实量化训练模型里。
    # 目标不是“直接加载原生量化模块”，而是提取：
    # 1. int8 权重码值
    # 2. 浮点 bias 对应的整数累加域 bias
    # 3. x_scale / y_scale / zero_x / zero_y
    # 4. effective_scale = x_scale * w_scale / y_scale
    state = checkpoint["model_state_dict"]

    quant_scale = state["quant.scale"].to(torch.float32)
    quant_zero = state["quant.zero_point"].to(torch.float32)

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


def build_model(mode: str) -> nn.Module:
    if mode == "float":
        return FloatMNISTNet()
    if mode == "qat":
        return build_native_qat_model()
    if mode == "quantized":
        return QuantizedMNISTNet()
    raise ValueError(f"Unsupported mode: {mode}")
