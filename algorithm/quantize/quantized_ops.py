"""
In this file, we implement quantized ops to simulate the device-side inference of quantized models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

sys.path.append("..")

USE_FP_SCALE = True


#############################
########### utils ###########
#############################

def to_np(x):  # cast x to np array if it is a number
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return np.array([x])


def to_pt(x):
    if isinstance(x, torch.Tensor):
        if x.dtype == torch.float64:
            return x.float()
        else:
            return x.float()  # NOTE: for backward, we need everything float
    elif isinstance(x, np.ndarray):
        org_dtype = x.dtype
        x = torch.from_numpy(x)
        if org_dtype in [np.int64, np.int32, np.int8]:
            return x.float()  # NOTE: for backward, we need everything float
        elif org_dtype in [np.float32, np.float64]:
            return x.float()
        else:
            raise NotImplementedError(org_dtype)
    else:
        return to_pt(np.array(x))


#############################
####### quantized ops #######
#############################
'''
量化推理实现, 模拟 MCU 上真实执行结果
'''


# 量化平均池化层
class QuantizedAvgPool(nn.Module):
    def __init__(self):
        super(QuantizedAvgPool, self).__init__()

    def forward(self, x):
        x = x.float()
        # 对 H、W 维度做全局平均池化
        x = x.mean([-1, -2], keepdim=True).round().int()
        return x


class QuantizedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 zero_x=0, zero_w=0, zero_y=0,
                 effective_scale=None,
                 significand=1, channel_shift=0,
                 w_bit=8, a_bit=None,
                 ):
        super(QuantizedConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                              padding, dilation, groups, bias, padding_mode)
        # 输入量化零点
        self.register_buffer('zero_x', to_pt(zero_x))
        # 权重量化零点
        self.register_buffer('zero_w', to_pt(zero_w))
        # 输出量化零点
        self.register_buffer('zero_y', to_pt(zero_y))
        # 缩放因子, 把卷积累加结果重新映射到输出量化空间
        self.register_buffer('effective_scale', effective_scale)
        # 把 effective_scale 近似为 significand / 2 ^ channel_shift
        self.significand = significand
        self.channel_shift = channel_shift
        # 权重量化位宽
        self.w_bit = w_bit
        # 激活量化位宽
        self.a_bit = a_bit if a_bit is not None else w_bit

    def forward(self, x):
        # assume x and weight are both in int8
        weight = self.weight.int()  # - self.zero_w.view(-1, 1, 1, 1)
        x = x.int() - self.zero_x
        # pytorch 的 conv2d 不支持 int8 运算，所以先转换成 float 进行卷积，然后再转换回 int
        out = F.conv2d(x.float(),
                       weight.float(),
                       None,
                       self.stride,
                       self.padding,
                       self.dilation,
                       self.groups).round().int()
        # 加偏置, view(1, -1, 1, 1) 是为了广播到每个输出通道 NCHW -> 1C11
        out = out + self.bias.int().view(1, -1, 1, 1)

        if self.effective_scale is not None:
            # 卷积累加结果是高精度整数, 乘一个缩放因子把它映射到输出量化空间
            out = (out.double() * self.effective_scale.view(1, -1, 1, 1)).round().int()
        else:
            # effective_scale为空, 走定点整数方案
            out = out.type(torch.int64)
            # 乘整数系数 significand
            out = out * self.significand.view(1, -1, 1, 1)
            # nudge 操作, 因为要除 2^31, 加减除数的一半实现四舍五入
            out[out >= 0] += (1 << 30)
            out[out < 0] += (1 - (1 << 30))

            out = out // (1 << 31)
            out = out.cpu().numpy()

            shift = (-self.channel_shift).view(1, -1, 1, 1)
            shift = shift.cpu().numpy()
            # 带舍入的右移
            mask = ((1 << shift) - 1)
            remainder = mask & out
            threshold = mask >> 1
            remainder[out < 0] -= 1
            out = out >> shift
            out[remainder > threshold] += 1

            out = torch.from_numpy(out)
        # 输出零点
        out = out + self.zero_y
        # 截断到激活量化位宽的范围
        return out.clamp(- 2 ** (self.a_bit - 1), 2 ** (self.a_bit - 1) - 1)


class QuantizedElementwise(nn.Module):
    def __init__(self, operator, zero_x1, zero_x2, zero_y, scale_x1, scale_x2, scale_y):
        super().__init__()
        self.operator = operator
        assert operator in ['add', 'mult']
        self.register_buffer('zero_x1', to_pt(zero_x1))
        self.register_buffer('zero_x2', to_pt(zero_x2))
        self.register_buffer('zero_y', to_pt(zero_y))
        self.register_buffer('scale_x1', to_pt(scale_x1))
        self.register_buffer('scale_x2', to_pt(scale_x2))
        self.register_buffer('scale_y', to_pt(scale_y))

    def forward(self, x1, x2):
        # 反量化
        x1 = (x1.int() - self.zero_x1) * self.scale_x1
        x2 = (x2.int() - self.zero_x2) * self.scale_x2

        if self.operator == 'add':
            out = x1 + x2
        elif self.operator == 'mult':
            out = x1 * x2
        else:
            raise NotImplementedError
        # 量化回整数
        out = (out.double() / self.scale_y).round().int()
        out = out + self.zero_y
        return out

# 从输入特征里提取“每个通道的重要性”，再把这个重要性乘回原输入，对通道做加权
class QuantizedSE(nn.Module):
    def __init__(self, fc, q_mult, a_bit=8):
        super().__init__()
        self.fc = fc
        self.avg_pool = QuantizedAvgPool()
        self.q_mult = q_mult

        self.a_bit = a_bit

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc(out)
        x = self.q_mult(x, out)
        return x.clamp(- 2 ** (self.a_bit - 1), 2 ** (self.a_bit - 1) - 1)


class QuantizedMbBlock(nn.Module):
    def __init__(self, conv, q_add=None, residual_conv=None, a_bit=8):
        super().__init__()
        self.conv = conv
        self.q_add = q_add
        self.residual_conv = residual_conv

        self.a_bit = a_bit

    def forward(self, x):
        out = self.conv(x)
        if self.q_add is not None:
            if self.residual_conv is not None:
                x = self.residual_conv(x)
            out = self.q_add(x, out)
            return out.clamp(- 2 ** (self.a_bit - 1), 2 ** (self.a_bit - 1) - 1)
        else:
            return out

