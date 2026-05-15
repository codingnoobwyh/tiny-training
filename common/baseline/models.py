import torch
import torch.ao.quantization as tq
from torch import nn

from common.train_utils import load_checkpoint


class FloatMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
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
        return x


class QuantizableMNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
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
        tq.fuse_modules(
            self,
            [["conv1", "relu1"], ["conv2", "relu2"], ["fc1", "relu3"]],
            inplace=True,
        )


def build_qat_model() -> nn.Module:
    model = QuantizableMNISTNet()
    model.fuse_model()
    model.qconfig = tq.get_default_qat_qconfig("fbgemm")
    return tq.prepare_qat(model.train(), inplace=False)


def build_ptq_model() -> nn.Module:
    model = QuantizableMNISTNet()
    model.fuse_model()
    model.qconfig = tq.get_default_qconfig("fbgemm")
    prepared = tq.prepare(model.eval(), inplace=False)
    return tq.convert(prepared, inplace=False)


def initialize_qat_from_float(model: nn.Module, init_from: str) -> None:
    checkpoint = load_checkpoint(init_from)
    if checkpoint["mode"] != "float":
        raise ValueError(f"qat runner expects float checkpoint, got {checkpoint['mode']}")
    source_state_dict = checkpoint["model_state_dict"]
    target_state_dict = model.state_dict()
    shared_state_dict = {
        key: value
        for key, value in source_state_dict.items()
        if key in target_state_dict and target_state_dict[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(shared_state_dict, strict=False)
    print(
        "Initialized QAT model from float checkpoint: "
        f"loaded={len(shared_state_dict)} missing={len(missing)} unexpected={len(unexpected)}"
    )
