import torch

from torch_impl.model import QasMnistNet

# PTQ bias 是浮点域, QAS forward 里 bias 加在整数累加域
def _bias_to_int_domain(bias_fp: torch.Tensor, x_scale: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return torch.round(bias_fp.to(torch.float32) / (x_scale.to(torch.float32) * w_scale.to(torch.float32)))


def _load_layer_from_ptq(
    layer,
    relu,
    weight_q: torch.Tensor,
    bias_fp: torch.Tensor,
    x_scale: torch.Tensor,
    x_zero: torch.Tensor,
    y_scale: torch.Tensor,
    y_zero: torch.Tensor,
) -> None:
    w_scale = weight_q.q_per_channel_scales().to(torch.float32)

    # PTQ 权重是 quantized tensor, QAS 模型里存 int code 的 float tensor
    layer.weight.copy_(weight_q.int_repr().to(torch.float32))
    layer.bias.copy_(_bias_to_int_domain(bias_fp, x_scale, w_scale))

    layer.zero_x.copy_(x_zero.to(torch.float32).reshape(layer.zero_x.shape))
    layer.zero_y.copy_(y_zero.to(torch.float32).reshape(layer.zero_y.shape))
    layer.x_scale.copy_(x_scale.to(torch.float32).reshape(layer.x_scale.shape))
    layer.w_scale.copy_(w_scale.reshape(layer.w_scale.shape))
    layer.y_scale.copy_(y_scale.to(torch.float32).reshape(layer.y_scale.shape))

    if relu is not None:
        relu.zero_y.copy_(y_zero.to(torch.float32).reshape(relu.zero_y.shape))


def init_model_from_ptq(model: QasMnistNet, checkpoint: dict) -> None:
    state = checkpoint["model_state_dict"]

    with torch.no_grad():
        input_scale = state["quant.scale"].to(torch.float32)
        input_zero = state["quant.zero_point"].to(torch.float32)
        model.input_scale.copy_(input_scale.reshape(model.input_scale.shape))
        model.input_zero_point.copy_(input_zero.reshape(model.input_zero_point.shape))

        _load_layer_from_ptq(
            layer=model.conv1,
            relu=model.relu1,
            weight_q=state["conv1.weight"],
            bias_fp=state["conv1.bias"],
            x_scale=input_scale,
            x_zero=input_zero,
            y_scale=state["conv1.scale"],
            y_zero=state["conv1.zero_point"],
        )

        _load_layer_from_ptq(
            layer=model.conv2,
            relu=model.relu2,
            weight_q=state["conv2.weight"],
            bias_fp=state["conv2.bias"],
            x_scale=state["conv1.scale"],
            x_zero=state["conv1.zero_point"],
            y_scale=state["conv2.scale"],
            y_zero=state["conv2.zero_point"],
        )

        fc1_weight_q, fc1_bias_fp = state["fc1._packed_params._packed_params"]
        _load_layer_from_ptq(
            layer=model.fc1,
            relu=model.relu3,
            weight_q=fc1_weight_q,
            bias_fp=fc1_bias_fp,
            x_scale=state["conv2.scale"],
            x_zero=state["conv2.zero_point"],
            y_scale=state["fc1.scale"],
            y_zero=state["fc1.zero_point"],
        )

        fc2_weight_q, fc2_bias_fp = state["fc2._packed_params._packed_params"]
        _load_layer_from_ptq(
            layer=model.fc2,
            relu=None,
            weight_q=fc2_weight_q,
            bias_fp=fc2_bias_fp,
            x_scale=state["fc1.scale"],
            x_zero=state["fc1.zero_point"],
            y_scale=state["fc2.scale"],
            y_zero=state["fc2.zero_point"],
        )
