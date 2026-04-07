import torch
import torch.nn as nn
import torch.ao.quantization as tq


class FloatMNISTNet(nn.Module):
    # 浮点基线模型。
    # 这里只放最直接的模型定义，不混入 QAT/PTQ/QAS 训练入口逻辑。
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
    # PyTorch 原生 QAT/PTQ baseline 的母体网络。
    # 这个模型只服务 baseline，不代表 QAS 方法本体。
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
def build_native_qat_model() -> nn.Module:
    model = QuantizableMNISTNet()
    model.fuse_model()
    model.qconfig = tq.get_default_qat_qconfig("fbgemm")
    return tq.prepare_qat(model.train(), inplace=False)


def build_native_ptq_prepare_model() -> nn.Module:
    model = QuantizableMNISTNet()
    model.fuse_model()
    model.qconfig = tq.get_default_qconfig("fbgemm")
    return tq.prepare(model.eval(), inplace=False)


def build_native_ptq_prepare_model_from_float_state_dict(float_state_dict: dict[str, torch.Tensor]) -> nn.Module:
    model = QuantizableMNISTNet()
    model.load_state_dict(float_state_dict, strict=True)
    model.fuse_model()
    model.qconfig = tq.get_default_qconfig("fbgemm")
    return tq.prepare(model.eval(), inplace=False)


def build_native_ptq_converted_model() -> nn.Module:
    import warnings

    prepared = build_native_ptq_prepare_model()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*must run observer before calling calculate_qparams.*")
        return tq.convert(prepared, inplace=False)
