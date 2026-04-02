import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_buffer_tensor(value, dtype=torch.float32) -> torch.Tensor:
    # 量化参数既可能是 Python 标量, 也可能已经是 tensor
    # 这里统一转成 float tensor, 方便后面 register_buffer
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=dtype)
    return torch.tensor(value, dtype=dtype)


def _round_tensor(x: torch.Tensor) -> torch.Tensor:
    # 真实量化训练里, 很多中间量在“语义上”已经是整数
    # 但为了保留 autograd 路径, 这里仍用 float tensor 承载整数值
    return x.round()


def _as_channel_bias(x: torch.Tensor) -> torch.Tensor:
    # 零点可能是标量, 也可能按通道存成一维向量。
    # 卷积前向统一扩成 [1, C, 1, 1], 方便和 NCHW 张量广播。
    if x.ndim == 0:
        return x
    if x.ndim == 1:
        return x.view(1, -1, 1, 1)
    return x


def _get_per_channel_scale(x: torch.Tensor, n_bit: int = 8, eps: float = 1e-6) -> torch.Tensor:
    # 按第 0 维逐通道取最大绝对值, 再映射到 int8 网格
    #
    # 这里一定要和“前向量化 scale”区分开:
    # 1. 前向真正使用的量化参数是层上的 x_scale / y_scale / effective_scale, 
    #    它们应该来自 PTQ, 或者来自后续更正式的量化初始化流程
    # 2. 这个函数算出来的 scale 不是前向量化参数, 
    #    而是为了把 backward 产生的浮点梯度重新投影回离散量化网格而临时构造的 scale
    #
    # 所以这里的 scale 只服务于“梯度重投影”, 不是卷积前向公式的一部分
    x = x.reshape(x.shape[0], -1)
    max_abs = x.abs().max(dim=1)[0].clamp_min(eps)
    qmax = 2 ** (n_bit - 1) - 1
    return max_abs / qmax


def _project_gradient_per_channel(grad: torch.Tensor, n_bit: int = 8) -> torch.Tensor:
    # 这里把“当前一步 backward 得到的浮点梯度”重新投影到量化网格上
    # 这里用的 per-channel scale 不是 PTQ 已经确定好的前向 scale, 
    # 而是从当前 grad 本身统计出来、专门给梯度离散化使用的 scale
    # 注意:梯度仍然是 float tensor, 只是它的取值被限制在某个离散集合里
    scales = _get_per_channel_scale(grad, n_bit=n_bit)
    view_shape = [grad.shape[0]] + [1] * (grad.dim() - 1)
    return (grad / scales.view(*view_shape)).round() * scales.view(*view_shape)


def _project_activation_gradient_per_channel(grad: torch.Tensor, n_bit: int = 8) -> torch.Tensor:
    # 对输入梯度做重投影时, 我们希望按通道维 C 来量化
    # 对 NCHW 张量来说, C 在 dim=1, 因此先转成 CNHW, 再复用同一个 per-channel 量化逻辑
    # 这里同样是在量化“梯度”, 不是在重新估计前向激活的 x_scale
    grad_t = grad.transpose(0, 1)
    scales = _get_per_channel_scale(grad_t, n_bit=n_bit)
    return (grad / scales.view(1, -1, 1, 1)).round() * scales.view(1, -1, 1, 1)


def _project_feature_gradient(grad: torch.Tensor, n_bit: int = 8) -> torch.Tensor:
    # 对线性层输入梯度做重投影时，希望按特征维度处理。
    # 对 [N, C] 张量来说，先转成 [C, N]，再按“第 0 维是通道”复用同一套逻辑。
    grad_t = grad.transpose(0, 1)
    scales = _get_per_channel_scale(grad_t, n_bit=n_bit)
    return (grad / scales.view(1, -1)).round() * scales.view(1, -1)


class _QuantizedReLURange(torch.autograd.Function):
    '''
    1. 下界由 ReLU 语义决定, 直接截到 zero_y
    2. 上界由激活量化位宽决定, 限制到当前码值范围上限
    '''
    @staticmethod
    def forward(ctx, x: torch.Tensor, zero_y: torch.Tensor, a_bit: int) -> torch.Tensor:
        qmax = 2 ** (a_bit - 1) - 1
        # zero_y 对应“实数 0”在输出量化域里的码值, 因此量化版 ReLU 在码值域里的效果就是 clamp(min=zero_y, max=qmax)
        lower = _as_channel_bias(zero_y)
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
        bias: torch.Tensor | None,
        zero_x: torch.Tensor,
        zero_y: torch.Tensor,
        effective_scale: torch.Tensor,
        stride,
        padding,
        dilation,
        groups: int,
    ) -> torch.Tensor:
        # 1. 输入和权重先 round 成“整数语义”
        # 2. 输入减去输入零点
        # 3. 做整数卷积累加
        # 4. 加量化 bias
        # 5. 乘 effective_scale 映射回输出量化域
        # 6. 再加输出零点
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.input_size = x.shape
        ctx.weight_size = weight.shape

        weight_int = _round_tensor(weight)
        x_int = _round_tensor(x)
        x_centered = x_int - _as_channel_bias(zero_x)

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
        conv_out = _round_tensor(conv_out)

        if bias is not None:
            conv_out = conv_out + bias.view(1, -1, 1, 1)

        # effective_scale = scale_x * scale_w / scale_y, 一次性把输入/输出/权重三者
        # 的scale关系一次性折合到一起, 前向公式更紧凑
        out = _round_tensor(conv_out * effective_scale.view(1, -1, 1, 1))
        out = out + _as_channel_bias(zero_y)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # 前向里对应的中间变量可以写成:
        # y_conv = conv(...)
        # y_bias = round(y_conv) + bias
        # y_scaled = round(y_bias * effective_scale)
        # y_out = y_scaled + zero_y
        #
        # backward 传进来的 grad_output 是 dL/d(y_out)
        # 忽略 round 的不可导性,  仅保留乘 effective_scale 这条主链的梯度
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
        
        grad_zero_x = -grad_conv_in.sum([0, 2, 3])
        grad_zero_y = grad_output.sum([0, 2, 3])

        grad_w = torch.nn.grad.conv2d_weight(
            input=x_centered,
            weight_size=ctx.weight_size,
            grad_output=grad_conv_out,
            stride=ctx.stride,
            padding=ctx.padding,
            dilation=ctx.dilation,
            groups=ctx.groups,
        )

        # 把梯度重投影到 int8 网格
        grad_w = _project_gradient_per_channel(grad_w, n_bit=8)
        grad_x = _project_activation_gradient_per_channel(grad_x, n_bit=8)

        return grad_x, grad_w, grad_bias, grad_zero_x, grad_zero_y, None, None, None, None, None


class _QASLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        zero_x: torch.Tensor,
        zero_y: torch.Tensor,
        effective_scale: torch.Tensor,
    ) -> torch.Tensor:
        # 线性层沿用和卷积一样的整数语义：
        # 1. x / w 先 round 到整数码值
        # 2. 输入减 zero_x
        # 3. 做整数域线性变换
        # 4. 加量化 bias
        # 5. 乘 effective_scale 映射回输出量化域
        # 6. 加 zero_y
        weight_int = _round_tensor(weight)
        x_int = _round_tensor(x)
        x_centered = x_int - zero_x.view(1, -1) if zero_x.ndim == 1 else x_int - zero_x

        ctx.save_for_backward(weight_int, effective_scale, x_centered)

        out = F.linear(x_centered, weight_int, None)
        out = _round_tensor(out)
        if bias is not None:
            out = out + bias.view(1, -1)
        out = _round_tensor(out * effective_scale.view(1, -1))
        out = out + (zero_y.view(1, -1) if zero_y.ndim == 1 else zero_y)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight_int, effective_scale, x_centered = ctx.saved_tensors

        grad_linear_out = grad_output * effective_scale.view(1, -1)
        grad_bias = grad_linear_out.sum(dim=0)
        grad_x = grad_linear_out @ weight_int
        grad_zero_x = -grad_x.sum(dim=0)
        grad_zero_y = grad_output.sum(dim=0)
        grad_w = grad_linear_out.transpose(0, 1) @ x_centered

        # 和卷积保持一致：默认把梯度重新投影到 int8 网格。
        grad_w = _project_gradient_per_channel(grad_w, n_bit=8)
        grad_x = _project_feature_gradient(grad_x, n_bit=8)

        return grad_x, grad_w, grad_bias, grad_zero_x, grad_zero_y, None


class QASConvReLU2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        zero_x=0.0,
        zero_y=0.0,
        effective_scale=None,
        x_scale=1.0,
        y_scale=1.0,
        w_bit: int = 8,
        a_bit: int = 8,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # zero point 和 scale 先按原论文默认思路固定为 buffer
        # 这样这个最小 demo 里真正被优化器更新的仍然是 weight / bias
        self.register_buffer("zero_x", _to_buffer_tensor(zero_x))
        self.register_buffer("zero_y", _to_buffer_tensor(zero_y))
        if effective_scale is None:
            effective_scale = torch.ones(out_channels, dtype=torch.float32)
        self.register_buffer("effective_scale", _to_buffer_tensor(effective_scale))
        self.register_buffer("x_scale", _to_buffer_tensor(x_scale))
        self.register_buffer("y_scale", _to_buffer_tensor(y_scale))

        self.w_bit = w_bit
        self.a_bit = a_bit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        融合版量化卷积 + ReLU。
        1. 先执行真实量化语义卷积
        2. 再立刻在量化码值域里做 ReLU 和上界截断
        '''
        out = _QASConv2dFunc.apply(
            x,
            self.weight,
            self.bias,
            self.zero_x,
            self.zero_y,
            self.effective_scale,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return _QuantizedReLURange.apply(out, self.zero_y, self.a_bit)


class QASLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        zero_x=0.0,
        zero_y=0.0,
        effective_scale=None,
        x_scale=1.0,
        y_scale=1.0,
        w_bit: int = 8,
        a_bit: int = 8,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.register_buffer("zero_x", _to_buffer_tensor(zero_x))
        self.register_buffer("zero_y", _to_buffer_tensor(zero_y))
        if effective_scale is None:
            effective_scale = torch.ones(out_features, dtype=torch.float32)
        self.register_buffer("effective_scale", _to_buffer_tensor(effective_scale))
        self.register_buffer("x_scale", _to_buffer_tensor(x_scale))
        self.register_buffer("y_scale", _to_buffer_tensor(y_scale))

        self.w_bit = w_bit
        self.a_bit = a_bit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FC 后面默认不融合 ReLU，因为最后分类头通常直接输出 logits。
        # 如果中间隐藏层要接 ReLU，可以在网络结构里显式接一个量化 ReLU 模块。
        return _QASLinearFunc.apply(
            x,
            self.weight,
            self.bias,
            self.zero_x,
            self.zero_y,
            self.effective_scale,
        )
