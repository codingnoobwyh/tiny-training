import torch
import torch.nn as nn
import torch.nn.functional as F


def _symmetric_scale(x: torch.Tensor, n_bits: int, eps: float = 1e-8) -> torch.Tensor:
    # 对称量化方案: scale = max(abs(x)) / qmax
    qmax = 2 ** (n_bits - 1) - 1
    max_abs = x.detach().abs().amax()
    scale = max_abs / qmax
    return scale.clamp_min(eps)


class FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor, n_bits: int) -> torch.Tensor:
        # forward 做“假量化”：
        # 1. 除以 scale，把浮点值映射到整数网格
        # 2. round 到最近的离散点
        # 3. clamp 到位宽允许的范围
        # 4. 再乘回 scale，返回一个 float tensor
        #
        # 返回值仍然是浮点张量，所以后续 PyTorch 计算图可以继续跑；
        # 但它的数值已经被限制到了量化网格上。
        qmax = 2 ** (n_bits - 1) - 1
        qmin = -2 ** (n_bits - 1)
        scaled = x / scale
        quantized = scaled.round().clamp(qmin, qmax)
        ctx.save_for_backward(scaled)
        ctx.qmin = qmin
        ctx.qmax = qmax
        return quantized * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scaled, = ctx.saved_tensors
        # backward 使用 STE 近似。
        # round 理论上不可导，所以这里不是严格求导，
        # 而是让处在量化范围内的值把梯度直通回去。
        mask = (scaled >= ctx.qmin) & (scaled <= ctx.qmax)
        grad_x = grad_output * mask
        return grad_x, None, None


def fake_quantize_tensor(x: torch.Tensor, n_bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 除了返回假量化后的张量，还把当前使用的 scale 一并返回。
    # 后面 QAS 会读取这些 scale 来修正梯度。
    scale = _symmetric_scale(x, n_bits)
    x_q = FakeQuantizeSTE.apply(x, scale, n_bits)
    return x_q, scale.detach()


class QuantizedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bits: int = 4,
        a_bits: int = 4,
        quantize_output: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        # w_bits 对应权重量化位宽，a_bits 对应激活量化位宽。
        # 最后一层通常不继续量化输出，所以留一个 quantize_output 开关。
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.quantize_output = quantize_output

        # 下面三个 buffer 记录“最近一次前向传播”看到的量化尺度。
        # QAS 不在 forward 里改参数，而是在 backward 之后读取这些值修正 grad。
        self.register_buffer("last_input_scale", torch.tensor(1.0))
        self.register_buffer("last_weight_scale", torch.tensor(1.0))
        self.register_buffer("last_output_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 量化层的前向传播分三步：
        # 1. 输入激活假量化
        # 2. 权重假量化
        # 3. 用量化后的张量做线性层计算
        x_q, x_scale = fake_quantize_tensor(x, self.a_bits)
        w_q, w_scale = fake_quantize_tensor(self.weight, self.w_bits)
        out = F.linear(x_q, w_q, self.bias)

        # 把当前 batch 的尺度缓存下来，供 QAS 在优化器里读取。
        self.last_input_scale.copy_(x_scale)
        self.last_weight_scale.copy_(w_scale)

        if self.quantize_output:
            # 隐藏层输出也继续量化，确保下一层拿到的仍然是量化语义下的激活。
            out_q, out_scale = fake_quantize_tensor(out, self.a_bits)
            self.last_output_scale.copy_(out_scale)
            return out_q

        # 最后一层如果不量化输出，就把 output scale 置回 1。
        self.last_output_scale.fill_(1.0)
        return out


class QASSGD(torch.optim.SGD):
    @staticmethod
    def pre_step(model: nn.Module) -> None:
        # 这里是 demo 里唯一真正和 QAS 强绑定的逻辑。
        #
        # 普通 quant 模式：
        #   backward 后直接 step
        #
        # quant_qas 模式：
        #   backward 后先执行 pre_step
        #   读取每层记录下来的 input/weight/output scale
        #   再用这些 scale 去重标定 weight.grad 和 bias.grad
        for module in model.modules():
            if not isinstance(module, QuantizedLinear):
                continue

            input_scale = module.last_input_scale.detach().clamp_min(1e-8)
            weight_scale = module.last_weight_scale.detach().clamp_min(1e-8)
            output_scale = module.last_output_scale.detach().clamp_min(1e-8)

            if module.weight.grad is not None:
                # 量化会让不同层参数的数值尺度不一致。
                # 这里用 output_scale / weight_scale 做一个最小版的权重梯度修正。
                correction = (output_scale / weight_scale).clamp(0.25, 4.0)
                module.weight.grad.mul_(correction)

            if module.bias is not None and module.bias.grad is not None:
                # bias 同时受到输入尺度和权重尺度影响，所以单独修正。
                correction = (output_scale / (input_scale * weight_scale)).clamp(0.25, 4.0)
                module.bias.grad.mul_(correction)
