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


def _get_per_channel_scale(x: torch.Tensor, n_bit: int = 8, eps: float = 1e-6) -> torch.Tensor:
    # 按第 0 维逐通道取最大绝对值, 再映射到 int8 网格
    #
    # 这里一定要和 "前向量化 scale" 区分开:
    # 1. 前向真正使用的量化参数是层上的 x_scale / y_scale / effective_scale,
    #    它们应该来自 PTQ, 或者来自后续更正式的量化初始化流程
    # 2. 这个函数算出来的 scale 不是前向量化参数,
    #    而是为了把 backward 产生的浮点梯度重新投影回离散量化网格而临时构造的 scale
    #
    # 所以这里的 scale 只服务于 "梯度重投影" , 不是卷积前向公式的一部分
    x = x.reshape(x.shape[0], -1)
    max_abs = x.abs().max(dim=1)[0].clamp_min(eps)
    qmax = 2 ** (n_bit - 1) - 1
    return max_abs / qmax


def _project_gradient_per_channel(grad: torch.Tensor, n_bit: int = 8) -> torch.Tensor:
    # 这里把 "当前一步 backward 得到的浮点梯度" 重新投影到量化网格上
    # 这里用的 per-channel scale 不是 PTQ 已经确定好的前向 scale,
    # 而是从当前 grad 本身统计出来、专门给梯度离散化使用的 scale
    # 注意:梯度仍然是 float tensor, 只是它的取值被限制在某个离散集合里
    scales = _get_per_channel_scale(grad, n_bit=n_bit)
    view_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
    return (grad / scales.view(*view_shape)).round() * scales.view(*view_shape)


def _project_activation_gradient_per_channel(grad: torch.Tensor, n_bit: int = 8) -> torch.Tensor:
    # 对输入梯度做重投影时, 我们希望按通道维 C 来量化
    # 对 NCHW 张量来说, C 在 dim=1, 因此先转成 CNHW, 再复用同一个 per-channel 量化逻辑
    # 这里同样是在量化 "梯度" , 不是在重新估计前向激活的 x_scale
    grad_t = grad.transpose(0, 1)
    scales = _get_per_channel_scale(grad_t, n_bit=n_bit)
    return (grad / scales.view(1, -1, 1, 1)).round() * scales.view(1, -1, 1, 1)


def _project_feature_gradient(grad: torch.Tensor, n_bit: int = 8) -> torch.Tensor:
    # 对线性层输入梯度做重投影时, 希望按特征维度处理.
    # 对 [N, C] 张量来说, 先转成 [C, N], 再按 "第 0 维是通道" 复用同一套逻辑.
    grad_t = grad.transpose(0, 1)
    scales = _get_per_channel_scale(grad_t, n_bit=n_bit)
    return (grad / scales.view(1, -1)).round() * scales.view(1, -1)


class _QuantizedReLURange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, zero_y: torch.Tensor, a_bit: int) -> torch.Tensor:
        # zero_y 对应 "实数 0" 在输出量化域里的码值, 量化 ReLU 在码值域里的效果就是 clamp(min=zero_y, max=qmax)
        qmax = 2 ** (a_bit - 1) - 1
        lower = zero_y
        # bool 变量, 记录输入 x 哪些位置被截断, shape 和 x 一样.
        binary_mask = (lower <= x) & (x <= qmax)
        ctx.save_for_backward(binary_mask)
        # 创建一个和 x 形状一样的新张量, 所有元素都等于 qmax.
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
            input_size  = ctx.input_size,
            weight      = weight_int,
            grad_output = grad_conv_out,
            stride      = ctx.stride,
            padding     = ctx.padding,
            dilation    = ctx.dilation,
            groups      = ctx.groups,
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

        grad_w = _project_gradient_per_channel(grad_w, n_bit=8)
        grad_x = _project_activation_gradient_per_channel(grad_x, n_bit=8)

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

        grad_w = _project_gradient_per_channel(grad_w, n_bit=8)
        grad_x = _project_feature_gradient(grad_x, n_bit=8)

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
        w_bit: int = 8,
        a_bit: int = 8,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
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

        self.w_bit = w_bit
        self.a_bit = a_bit

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
        w_bit: int = 8,
        a_bit: int = 8,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=True)

        self.register_buffer("zero_x", _to_buffer_tensor(zero_x))
        self.register_buffer("zero_y", _to_buffer_tensor(zero_y))
        self.register_buffer("x_scale", _to_buffer_tensor(x_scale))
        self.register_buffer("w_scale", _to_buffer_tensor(w_scale))
        self.register_buffer("y_scale", _to_buffer_tensor(y_scale))

        self.w_bit = w_bit
        self.a_bit = a_bit

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
            x_scale = module.x_scale.detach().to(torch.float32)
            w_scale = module.w_scale.detach().to(torch.float32)

            if module.weight.grad is not None:
                view_shape = [w_scale.shape[0]] + [1] * (module.weight.grad.dim() - 1)
                module.weight.grad.data.div_(w_scale.view(*view_shape) ** 2)

            if module.bias is not None and module.bias.grad is not None:
                module.bias.grad.data.div_((x_scale * w_scale) ** 2)


def project_quantized_parameters(model: torch.nn.Module) -> None:
    # 参数投影放在 optimizer.step() 之后, 而不是 backward 里.
    # 这样职责清楚：
    # 1. backward 只负责算梯度
    # 2. optimizer.step() 负责更新浮点参数副本
    # 3. 这里再把参数拉回 "合法量化码值域"
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, (QASConv2d, QASLinear)):
                continue

            module.weight.data.copy_(module.weight.data.round().clamp(INT8_QMIN, INT8_QMAX))
            if module.bias is not None:
                module.bias.data.copy_(module.bias.data.round().clamp(INT32_QMIN, INT32_QMAX))
