import torch
import torch.nn as nn

from torch_impl.operators import QASConv2d, QASLinear, QuantizedReLU


class QasMnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("input_scale", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("input_zero_point", torch.tensor(0.0, dtype=torch.float32))
        self.conv1 = QASConv2d(1, 12,
                               zero_x=0.0, zero_y=0.0,
                               x_scale=1.0, w_scale=torch.ones(12), y_scale=1.0)
        self.relu1 = QuantizedReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = QASConv2d(12, 12,
                               zero_x=0.0, zero_y=0.0,
                               x_scale=1.0, w_scale=torch.ones(12), y_scale=1.0)
        self.relu2 = QuantizedReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = QASLinear(12 * 5 * 5, 128,
                             zero_x=0.0, zero_y=0.0,
                             x_scale=1.0, w_scale=torch.ones(128), y_scale=1.0)
        self.relu3 = QuantizedReLU()
        self.fc2 = QASLinear(128, 10,
                             zero_x=0.0, zero_y=0.0,
                             x_scale=1.0, w_scale=torch.ones(10), y_scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.round(x / self.input_scale) + self.input_zero_point
        x = x.clamp(-128, 127)
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
        x = (x - self.fc2.zero_y) * self.fc2.y_scale
        return x
