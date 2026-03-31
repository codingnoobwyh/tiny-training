import torch
import torch.nn as nn

from .ops_quant import QuantizedLinear


class FloatMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 这是浮点训练的标杆模型。
        # 使用经典 MNIST 卷积结构：
        # conv -> relu -> maxpool -> conv -> relu -> maxpool -> fc -> relu -> fc
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一个卷积块把 28x28 的单通道图像编码成更丰富的 16 通道特征。
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 第二个卷积块进一步提取局部模式，同时把空间尺寸压到 7x7。
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 卷积特征图先展平，再接两层全连接分类头。
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        # 最后一层直接输出 logits，不额外做 softmax。
        # 因为训练时 CrossEntropyLoss 内部已经处理了这部分。
        x = self.fc2(x)
        return x


class QuantMNISTNet(nn.Module):
    def __init__(self, hidden_dims: tuple[int, int] = (256, 128), w_bits: int = 4, a_bits: int = 4):
        super().__init__()
        # 当前量化模型先保留最小 MLP 结构。
        # 这不是最终形态，而是为了先把“量化训练 + QAS”主链路跑通。
        # 后面如果继续演进，再把卷积也替换成量化卷积层。
        self.flatten = nn.Flatten()
        self.fc1 = QuantizedLinear(28 * 28, hidden_dims[0], w_bits=w_bits, a_bits=a_bits)
        self.fc2 = QuantizedLinear(hidden_dims[0], hidden_dims[1], w_bits=w_bits, a_bits=a_bits)
        self.fc3 = QuantizedLinear(hidden_dims[1], 10, w_bits=w_bits, a_bits=a_bits, quantize_output=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def build_model(mode: str, w_bits: int = 4, a_bits: int = 4) -> nn.Module:
    if mode == "float":
        return FloatMNISTNet()
    if mode in {"quant", "quant_qas"}:
        return QuantMNISTNet(w_bits=w_bits, a_bits=a_bits)
    raise ValueError(f"Unsupported mode: {mode}")
