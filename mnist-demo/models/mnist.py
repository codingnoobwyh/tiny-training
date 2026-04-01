import copy
import torch
import torch.nn as nn
import torch.ao.quantization as tq


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


def build_model(mode: str) -> nn.Module:
    if mode == "float":
        return FloatMNISTNet()
    if mode == "qat":
        return build_native_qat_model()
    raise ValueError(f"Unsupported mode: {mode}")
