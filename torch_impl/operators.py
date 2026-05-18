import torch
import torch.nn as nn
import torch.nn.functional as F

from common.constants import INT8_QMIN, INT8_QMAX, INT32_QMIN, INT32_QMAX


def _to_buffer_tensor(value, dtype=torch.float32) -> torch.Tensor:
    # 量化参数既可能是 Python 标量, 也可能已经是 tensor
    # 这里统一转成 float tensor, 方便后面 register_buffer
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


class _QuantizedReLURange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, zero_y: torch.Tensor, a_bit: int) -> torch.Tensor:
        qmax = 2 ** (a_bit - 1) - 1
        lower = zero_y
        binary_mask = (lower <= x) & (x <= qmax)
        ctx.save_for_backward(binary_mask)
        upper = torch.full_like(x, qmax)
        return torch.minimum(torch.maximum(x, lower), upper)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (binary_mask,) = ctx.saved_tensors
        return grad_output * binary_mask, None, None


class _QASConv2dFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        zero_x: torch.Tensor,
        zero_y: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        y_scale: torch.Tensor,
        stride,
        padding,
        dilation,
        groups: int,
    ) -> torch.Tensor:
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_size = x.shape
        ctx.weight_size = weight.shape

        weight_int = weight.round().clamp(INT8_QMIN, INT8_QMAX)
        x_int = torch.round(x)
        x_centered = x_int - zero_x

        effective_scale = x_scale * w_scale / y_scale
        ctx.save_for_backward(weight_int, effective_scale, x_centered)

        conv_out = F.conv2d(
            input=x_centered,
            weight=weight_int,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        conv_out = torch.round(conv_out)
        conv_out = conv_out + bias.view(1, -1, 1, 1)

        out = torch.round(conv_out * effective_scale.view(1, -1, 1, 1))
        out = out + zero_y
        out = out.clamp(INT8_QMIN, INT8_QMAX)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight_int, effective_scale, x_centered = ctx.saved_tensors

        grad_conv_out = grad_output * effective_scale.view(1, -1, 1, 1)
        grad_bias = grad_conv_out.sum([0, 2, 3])

        grad_conv_in = torch.nn.grad.conv2d_input(
            input_size=ctx.input_size,
            weight=weight_int,
            grad_output=grad_conv_out,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )
        grad_x = grad_conv_in

        grad_w = torch.nn.grad.conv2d_weight(
            input=x_centered,
            weight_size=ctx.weight_size,
            grad_output=grad_conv_out,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )

        return grad_x, grad_w, grad_bias, None, None, None, None, None, None, None, None, None


class _QASLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        zero_x: torch.Tensor,
        zero_y: torch.Tensor,
        x_scale: torch.Tensor,
        w_scale: torch.Tensor,
        y_scale: torch.Tensor,
    ) -> torch.Tensor:
        weight_int = weight.round().clamp(INT8_QMIN, INT8_QMAX)
        x_int = torch.round(x)
        x_centered = x_int - zero_x

        effective_scale = x_scale * w_scale / y_scale
        ctx.save_for_backward(weight_int, effective_scale, x_centered)

        out = F.linear(x_centered, weight_int, None)
        out = torch.round(out)
        out = out + bias.view(1, -1)
        out = torch.round(out * effective_scale.view(1, -1))
        out = out + zero_y
        out = out.clamp(INT8_QMIN, INT8_QMAX)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight_int, effective_scale, x_centered = ctx.saved_tensors

        grad_linear_out = grad_output * effective_scale.view(1, -1)
        grad_bias = grad_linear_out.sum(dim=0)
        grad_x = grad_linear_out @ weight_int
        grad_w = grad_linear_out.transpose(0, 1) @ x_centered

        return grad_x, grad_w, grad_bias, None, None, None, None, None


class QASConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        zero_x,
        zero_y,
        x_scale,
        w_scale,
        y_scale,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.register_buffer("zero_x", _to_buffer_tensor(zero_x))
        self.register_buffer("zero_y", _to_buffer_tensor(zero_y))
        self.register_buffer("x_scale", _to_buffer_tensor(x_scale))
        self.register_buffer("w_scale", _to_buffer_tensor(w_scale))
        self.register_buffer("y_scale", _to_buffer_tensor(y_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _QASConv2dFunc.apply(
            x,
            self.weight,
            self.bias,
            self.zero_x,
            self.zero_y,
            self.x_scale,
            self.w_scale,
            self.y_scale,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantizedReLU(nn.Module):
    def __init__(self, zero_y=0.0, a_bit: int = 8):
        super().__init__()
        self.register_buffer("zero_y", _to_buffer_tensor(zero_y))
        self.a_bit = a_bit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _QuantizedReLURange.apply(x, self.zero_y, self.a_bit)


class QASLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        zero_x,
        zero_y,
        x_scale,
        w_scale,
        y_scale,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=True)

        self.register_buffer("zero_x", _to_buffer_tensor(zero_x))
        self.register_buffer("zero_y", _to_buffer_tensor(zero_y))
        self.register_buffer("x_scale", _to_buffer_tensor(x_scale))
        self.register_buffer("w_scale", _to_buffer_tensor(w_scale))
        self.register_buffer("y_scale", _to_buffer_tensor(y_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _QASLinearFunc.apply(
            x,
            self.weight,
            self.bias,
            self.zero_x,
            self.zero_y,
            self.x_scale,
            self.w_scale,
            self.y_scale,
        )


class QASSGD(torch.optim.SGD):
    def pre_step(self, model: torch.nn.Module) -> None:
        for module in model.modules():
            if not isinstance(module, (QASConv2d, QASLinear)):
                continue

            w_scale = module.w_scale.detach().to(torch.float32)
            x_scale = module.x_scale.detach().to(torch.float32)
            weight_grad = module.weight.grad
            bias_grad = module.bias.grad

            # Conv: [C_out, 1, 1, 1]; Linear: [C_out, 1].
            broadcast_shape = [w_scale.shape[0]] + [1] * (weight_grad.dim() - 1)
            weight_scale = w_scale.view(*broadcast_shape)
            bias_scale = x_scale * w_scale

            with torch.no_grad():
                weight_grad.div_(weight_scale.square())
                bias_grad.div_(bias_scale.square())


def project_quantized_parameters(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, (QASConv2d, QASLinear)):
                continue

            module.weight.data.round_().clamp_(INT8_QMIN, INT8_QMAX)
            module.bias.data.round_().clamp_(INT32_QMIN, INT32_QMAX)
