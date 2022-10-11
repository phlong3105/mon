#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standard and Experimental Layers.

Layer's arguments:
    b : bias
    d : dilation
    g : groups
    k : kernel_size
    ic: in_channels
    oc: out_channels
    s : stride
    p : padding

Forward pass:
    input/output
"""

from __future__ import annotations

import math
from functools import partial
from typing import Type

import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.nn import *
from torchvision.ops import Permute
from torchvision.ops import StochasticDepth

from one.core import *


# H1: - STANDARD ---------------------------------------------------------------
# Standard and well-known layers (in torch.nn) and combinations of them.

# H2: - Activation -------------------------------------------------------------

def hard_sigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return input.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(input + 3.0) / 6.0


class ArgMax(Module):
    """
    Find the indices of the maximum value of all elements in the input
    image.

    Args:
        dim (int | None): Dimension to find the indices of the maximum value.
            Defaults to None.
    """
    
    def __init__(
        self,
        dim: int | None = None,
        *args, **kwargs
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, input: Tensor) -> Tensor:
        return torch.argmax(input, dim=self.dim)


class Clamp(Module):
    """
    Clamp the feature value within [min, max]. More details can be found in
    `torch.clamp()`.

    Args:
        min (float): Lower-bound of the range to be clamped to. Defaults to -1.0
        max (float): Upper-bound of the range to be clamped to. Defaults to -1.0
    """
    
    def __init__(
        self,
        min: float = -1.0,
        max: float =  1.0,
        *args, **kwargs
    ):
        super().__init__()
        self.min = min
        self.max = max
        
    def forward(self, input: Tensor) -> Tensor:
        return torch.clamp(input, min=self.min, max=self.max)


class FReLU(Module):
    
    def __init__(
        self,
        c1: int,
        k : Ints = 3,
        *args, **kwargs
    ):
        super().__init__()
        k         = to_2tuple(k)
        self.conv = Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.act  = BatchNorm2d(c1)
        
    def forward(self, input: Tensor) -> Tensor:
        return torch.max(input, self.act(self.conv(input)))


CELU               = nn.CELU
Clip               = Clamp
ELU                = nn.ELU
GELU               = nn.GELU
GLU                = nn.GLU
Hardshrink         = nn.Hardshrink
Hardsigmoid        = nn.Hardsigmoid
Hardswish          = nn.Hardswish
Hardtanh           = nn.Hardtanh
LeakyReLU          = nn.LeakyReLU
LogSigmoid         = nn.LogSigmoid
LogSoftmax         = nn.LogSoftmax
Mish               = nn.Mish
MultiheadAttention = nn.MultiheadAttention
PReLU              = nn.PReLU
ReLU               = nn.ReLU
ReLU6              = nn.ReLU6
RReLU              = nn.RReLU
SELU               = nn.SELU
Sigmoid            = nn.Sigmoid
SiLU               = nn.SiLU
Softmax            = nn.Softmax
Softmax2d          = nn.Softmax2d
Softmin            = nn.Softmin
Softplus           = nn.Softplus
Softshrink         = nn.Softshrink
Softsign           = nn.Softsign
Tanh               = nn.Tanh
Tanhshrink         = nn.Tanhshrink
Threshold          = nn.Threshold


def to_act_layer(
    act    : Callable | None = ReLU(),
    inplace: bool            = True,
    *args, **kwargs
) -> Module:
    """
    Create activation layer.
    """
    # if isinstance(act, str):
    #     act = LAYERS.build(name=act)
    if isinstance(act, types.FunctionType):
        kwargs |= {"inplace": inplace}
        act     = act(*args, **kwargs)
    elif act is None:
        act     = Identity()
    return act


# H2: - Attention --------------------------------------------------------------

class ChannelAttention(Module):
    """
    Channel Attention adopted from the paper:
        "CBAM: Convolutional Block Attention Module"
    
    Args:
        channels (int): Number of input and output channels.
        reduction (int): Reduction factor.
    
    References:
        https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """
    
    def __init__(
        self,
        channels    : int,
        reduction   : int,
        kernel_size : Ints,
        stride      : Ints       = 1,
        padding     : Ints | str = 0,
        dilation    : Ints       = 1,
        groups      : int        = 1,
        bias        : bool       = False,
        padding_mode: str        = "zeros",
        device      : Any        = None,
        dtype       : Any        = None,
        avg_pool    : bool       = True,
        max_pool    : bool       = True,
        *args, **kwargs
    ):
        super().__init__()
        # Global average pooling: feature -> point
        self.avg_pool = AdaptiveAvgPool2d(1) if avg_pool else None
        self.max_pool = AdaptiveMaxPool2d(1) if max_pool else None
        # Feature channel downscale and upscale -> channel weight
        self.fc       = Sequential(
            Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            ReLU(inplace=True),
            Conv2d(
                in_channels  = channels // reduction,
                out_channels = channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
        )
        self.act = Sigmoid()
        
    def forward(self, input: Tensor) -> Tensor:
        avg_out = None
        max_out = None
        if self.avg_pool is not None:
            avg_out = self.fc(self.avg_pool(input))
        if self.max_pool is not None:
            max_out = self.fc(self.max_pool(input))
        if avg_out is not None and max_out is not None:
            x = avg_out + max_out
        else:
            x = avg_out or max_out
        x = self.act(x)
        return x


class PixelAttention(Module):
    """
    Pixel Attention Layer.
    
    Args:
        reduction (int): Reduction factor.
    """
    
    def __init__(
        self,
        channels    : int,
        reduction   : int,
        kernel_size : Ints,
        stride      : Ints       = 1,
        padding     : Ints | str = 0,
        dilation    : Ints       = 1,
        groups      : int        = 1,
        bias        : bool       = True,
        padding_mode: str        = "zeros",
        device      : Any        = None,
        dtype       : Any        = None,
        *args, **kwargs
    ):
        super().__init__()
        self.fc = Sequential(
            Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            ReLU(inplace=True),
            Conv2d(
                in_channels  = channels // reduction,
                out_channels = 1,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
        )
        self.act = Sigmoid()
        
    def forward(self, input: Tensor) -> Tensor:
        x  = input
        x  = self.fc(x)
        x  = self.act(x)
        x  = torch.mul(input, x)
        return x


class SimAM(Module):
    """
    A Simple, Parameter-Free Attention Module for Convolutional Neural Network
    adopted from paper:
        "https://github.com/ZjjConan/SimAM"
    """
    
    def __init__(
        self,
        e_lambda: float = 1e-4,
        *args, **kwargs
    ):
        super().__init__()
        self.e_lambda = e_lambda
        self.act      = Sigmoid()
        
    def forward(self, input: Tensor) -> Tensor:
        x  = input
        # Spatial size
        b, c, h, w = x.size()
        n          = w * h - 1
        # Square of (t - u)
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # d.sum() / n is channel variance
        v = d.sum(dim=[2, 3], keepdim=True) / n
        # E_inv groups all important of x
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        # Attended features
        y = x * self.act(e_inv)
        return y


class SpatialAttention(Module):
    """
    Spatial Attention adopted from the paper:
        "CBAM: Convolutional Block Attention Module"
    
    Args:
        channels (int): Number of input and output channels.
        reduction (int): Reduction factor.
        
    References:
        https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """
    
    def __init__(
        self,
        kernel_size : Ints,
        stride      : Ints = 1,
        dilation    : Ints = 1,
        groups      : int  = 1,
        bias        : bool = False,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels  = 2,
            out_channels = 1,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = kernel_size // 2,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = Sigmoid()
        
    def forward(self, input: Tensor) -> Tensor:
        x          = input
        avg_out    = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max( x, dim=1, keepdim=True)
        x          = torch.cat([avg_out, max_out], dim=1)
        x          = self.conv(x)
        x          = self.act(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels     : int,
        se_ratio        : float           = 0.25,
        reduced_base_chs: int      | None = None,
        act             : Callable | None = ReLU,
        gate_fn         : Callable | None = hard_sigmoid,
        divisor         : int             = 4,
        *args, **kwargs
    ):
        super().__init__()
        self.gate_fn = gate_fn
        reduced_channels = self.make_divisible(
            v       = (reduced_base_chs or in_channels) * se_ratio,
            divisor = divisor
        )
        self.avg_pool    = AdaptiveAvgPool2d(1)
        self.conv_reduce = Conv2d(
            in_channels  = in_channels,
            out_channels = reduced_channels,
            kernel_size  = 1,
            bias         = True
        )
        self.act1        = to_act_layer(act=act, inplace=True)
        self.conv_expand = Conv2d(
            in_channels  = reduced_channels,
            out_channels = in_channels,
            kernel_size  = 1,
            bias         = True
        )
    
    def make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    
    def forward(self, input: Tensor) -> Tensor:
        x    = input
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x    = x * self.gate_fn(x_se)
        return x


class SupervisedAttentionModule(Module):
    """
    Supervised Attention Module.
    """
    
    def __init__(
        self,
        channels    : int,
        kernel_size : Ints,
        stride      : int  = 1,
        dilation    : Ints = 1,
        groups      : int  = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None,
        **_
    ):
        super().__init__()
        padding = kernel_size[0] // 2 \
            if isinstance(kernel_size, Sequence) \
            else kernel_size // 2

        self.conv1 = Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv2  = Conv2d(
            in_channels  = channels,
            out_channels = 3,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv3  = Conv2d(
            in_channels  = 3,
            out_channels = channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = Sigmoid()
        
    def forward(self, input: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Run forward pass.

        Args:
            input (Sequence[Tensor]): A list of 2 tensors. The first tensor is
                the output from previous layer. The second tensor is the
                current step input.
            
        Returns:
            Supervised attention features.
            Output feature for the next layer.
        """
        assert_sequence_of_length(input, 2)
        fy    = input[0]
        x     = input[1]

        x1    = self.conv1(fy)
        img   = self.conv2(fy) + x
        x2    = self.act(self.conv3(img))
        x1   *= x2
        x1   += fy
        pred  = x1
        return pred, img


# H2: - Bottleneck -------------------------------------------------------------

class GhostBottleneck(Module):
    """
    Ghost Bottleneck with optional SE.
    
    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    """

    def __init__(
        self,
        in_channels : int,
        mid_channels: int,
        out_channels: int,
        kernel_size : Ints            = 3,
        stride      : Ints            = 1,
        padding     : Ints | str      = 0,
        dilation    : Ints            = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        se_ratio    : float           = 0.0,
        act         : Callable | None = ReLU,
        *args, **kwargs
    ):
        super().__init__()
        has_se      = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostConv2d(
            in_channels    = in_channels,
            out_channels   = mid_channels,
            kernel_size    = 1,
            dw_kernel_size = 3,
            stride         = stride,
            padding        = padding,
            dilation       = dilation,
            groups         = groups,
            bias           = bias,
            padding_mode   = padding_mode,
            device         = device,
            dtype          = dtype,
            act            = ReLU,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = (kernel_size - 1) // 2,
                dilation     = dilation,
                groups       = mid_channels,
                bias         = False,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
            self.bn_dw = BatchNorm2d(mid_channels)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_channels, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostConv2d(
            in_channels    = mid_channels,
            out_channels   = out_channels,
            kernel_size    = 1,
            dw_kernel_size = 3,
            stride         = stride,
            padding        = padding,
            dilation       = dilation,
            groups         = groups,
            bias           = bias,
            padding_mode   = padding_mode,
            device         = device,
            dtype          = dtype,
            act            = ReLU,
        )
        
        # Shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = Sequential()
        else:
            self.shortcut = Sequential(
                Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    stride       = stride,
                    padding      = (kernel_size - 1) // 2,
                    dilation     = dilation,
                    groups       = in_channels,
                    bias         = False,
                    padding_mode = padding_mode,
                    device       = device,
                    dtype        = dtype,
                ),
                BatchNorm2d(in_channels),
                Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = 1,
                    stride       = 1,
                    padding      = 0,
                    dilation     = dilation,
                    groups       = groups,
                    bias         = False,
                    padding_mode = padding_mode,
                    device       = device,
                    dtype        = dtype,
                ),
                BatchNorm2d(out_channels),
            )

    def forward(self, input: Tensor) -> Tensor:
        x        = input
        residual = x
        # 1st ghost bottleneck
        x  = self.ghost1(x)
        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)
        # 2nd ghost bottleneck
        x  = self.ghost2(x)
        x += self.shortcut(residual)
        return x


# H2: - Convolution ------------------------------------------------------------

def conv2d_same(
    input   : Tensor,
    weight  : Tensor,
    bias    : Tensor | None = None,
    stride  : Ints          = 1,
    padding : Ints | str    = 0,
    dilation: Ints          = 1,
    groups  : int           = 1,
    *args, **kwargs
):
    """
    Functional interface for Same Padding Convolution 2D.
    """
    input = pad_same(
        input       = input,
        kernel_size = weight.shape[-2:],
        stride      = stride,
        dilation    = dilation
    )
    return F.conv2d(
        input    = input,
        weight   = weight,
        bias     = bias,
        stride   = stride,
        padding  = padding,
        dilation = dilation,
        groups   = groups
    )


class ConvAct2d(Module):
    """
    Conv2d + Act.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints            = 1,
        padding     : Ints | str      = 0,
        dilation    : Ints            = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        act         : Callable | None = ReLU(),
        inplace     : bool            = True,
        *args, **kwargs
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = to_act_layer(act=act, inplace=inplace)
    
    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.conv(input))


class ConvReLU2d(Module):
    """
    Conv2d + ReLU.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints       = 1,
        padding     : Ints | str = 0,
        dilation    : Ints       = 1,
        groups      : int        = 1,
        bias        : bool       = True,
        padding_mode: str        = "zeros",
        device      : Any        = None,
        dtype       : Any        = None,
        *args, **kwargs
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = ReLU(inplace=True)

    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.conv(input))


class ConvSame2d(Conv2d):
    """
    Tensorflow like `SAME` convolution wrapper for 2D convolutions.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints       = 1,
        padding     : Ints | str = 0,
        dilation    : Ints       = 1,
        groups      : int        = 1,
        bias        : bool       = True,
        padding_mode: str        = "zeros",
        device      : Any        = None,
        dtype       : Any        = None,
        *args, **kwargs
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )

    def forward(self, input: Tensor) -> Tensor:
        return conv2d_same(
            input    = input,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )


class ConvTF2d(Conv2d):
    """
    Implementation of 2D convolution in TensorFlow with `padding` as "same",
    which applies padding to input (if needed) so that input image gets fully
    covered by filter and stride you specified. For stride of 1, this will
    ensure that output image size is same as input. For stride of 2, output
    dimensions will be half, for example.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints       = 1,
        padding     : Ints | str = 0,
        dilation    : Ints       = 1,
        groups      : int        = 1,
        bias        : bool       = True,
        padding_mode: str        = "zeros",
        device      : Any        = None,
        dtype       : Any        = None,
        *args, **kwargs
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        
    def forward(self, input: Tensor) -> Tensor:
        img_h, img_w       = input.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) *
                    self.dilation[0] + 1 - img_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) *
                    self.dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            input = F.pad(
                input = input,
                pad   = [pad_w // 2, pad_w - pad_w // 2,
                         pad_h // 2, pad_h - pad_h // 2]
            )
        output = F.conv2d(
            input    = input,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )
        return output


class DepthwiseSeparableConv2d(Module):
    """
    Depthwise Separable Conv2d.
    """

    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        dw_kernel_size: Ints,
        pw_kernel_size: Ints,
        dw_stride     : Ints       = 1,
        dw_padding    : Ints | str = 0,
        pw_stride     : Ints       = 1,
        pw_padding    : Ints | str = 0,
        dilation      : Ints       = 1,
        groups        : int        = 1,
        bias          : bool       = True,
        padding_mode  : str        = "zeros",
        device        : Any        = None,
        dtype         : Any        = None,
        *args, **kwargs
    ):
        super().__init__()
        self.dw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = dw_kernel_size,
            stride       = dw_stride,
            padding      = dw_padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.pw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            stride       = pw_stride,
            padding      = pw_padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        
    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DepthwiseSeparableConvReLU2d(Module):
    """
    Depthwise Separable Conv2d ReLU.
    """

    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        dw_kernel_size: Ints,
        pw_kernel_size: Ints,
        dw_stride     : Ints       = 1,
        pw_stride     : Ints       = 1,
        dw_padding    : Ints | str = 0,
        pw_padding    : Ints | str = 0,
        dilation      : Ints       = 1,
        groups        : int        = 1,
        bias          : bool       = True,
        padding_mode  : str        = "zeros",
        device        : Any        = None,
        dtype         : Any        = None,
        *args, **kwargs
    ):
        super().__init__()
        self.dw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = dw_kernel_size,
            stride       = dw_stride,
            padding      = dw_padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.pw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = pw_kernel_size,
            stride       = pw_stride,
            padding      = pw_padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = ReLU(inplace=True)
        
    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.act(x)
        return x


class GhostConv2d(Module):
    """
    Ghost Conv2d adopted from the paper: "GhostNet: More Features from Cheap
    Operations," CVPR 2020.
    
    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    """

    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        ratio         : int                    = 2,
        kernel_size   : Ints                   = 1,
        dw_kernel_size: Ints                   = 3,
        stride        : Ints                   = 1,
        padding       : Ints | str             = 0,
        dilation      : Ints                   = 1,
        groups        : int                    = 1,
        bias          : bool                   = True,
        padding_mode  : str                    = "zeros",
        device        : Any                    = None,
        dtype         : Any                    = None,
        act           : Callable | bool | None = ReLU,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels     = math.ceil(out_channels / ratio)
        new_channels      = init_channels * (ratio - 1)
        
        self.primary_conv = Sequential(
            Conv2d(
                in_channels  = in_channels,
                out_channels = init_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = kernel_size // 2,
                dilation     = dilation,
                groups       = groups,
                bias         = False,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            BatchNorm2d(init_channels),
            to_act_layer(act=act, inplace=True),
        )
        self.cheap_operation = Sequential(
            nn.Conv2d(
                in_channels  = init_channels,
                out_channels = new_channels,
                kernel_size  = dw_kernel_size,
                stride       = 1,
                padding      = dw_kernel_size // 2,
                groups       = init_channels,
                bias         = False,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            BatchNorm2d(new_channels),
            to_act_layer(act=act, inplace=True),
        )
        
    def forward(self, input: Tensor) -> Tensor:
        x   = input
        x1  = self.primary_conv(x)
        x2  = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]
    

class SubspaceBlueprintSeparableConv2d(Module):
    """
    Subspace Blueprint Separable Conv2d adopted from the paper:
        "Rethinking Depthwise Separable Convolutions: How Intra-Kernel
        Correlations Lead to Improved MobileNets," CVPR 2020.
    
    References:
        https://github.com/zeiss-microscopy/BSConv
    """

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : Ints,
        stride          : Ints            = 1,
        padding         : Ints | str      = 0,
        dilation        : Ints            = 1,
        groups          : int             = 1,
        bias            : bool            = True,
        padding_mode    : str             = "zeros",
        device          : Any             = None,
        dtype           : Any             = None,
        p               : float           = 0.25,
        min_mid_channels: int             = 4,
        act             : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        mid_channels  = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        self.pw_conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act1     = to_act_layer(act=act, num_features=mid_channels)
        self.pw_conv2 = Conv2d(
            in_channels  = mid_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act2    = to_act_layer(act=act, num_features=out_channels)
        self.dw_conv = Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        
    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.pw_conv1(x)
        if self.act1 is not None:
            x = self.act1(x)
        x = self.pw_conv2(x)
        if self.act2 is not None:
            x = self.act2(x)
        x = self.dw_conv(x)
        return x
    
    def regularization_loss(self):
        w   = self.pw_conv1.weight[:, :, 0, 0]
        wwt = torch.mm(w, torch.transpose(w, 0, 1))
        i   = torch.eye(wwt.shape[0], device=wwt.device)
        return torch.norm(wwt - i, p="fro")


class UnconstrainedBlueprintSeparableConv2d(Module):
    """
    Unconstrained Blueprint Separable Conv2d adopted from the paper:
        "Rethinking Depthwise Separable Convolutions: How Intra-Kernel
        Correlations Lead to Improved MobileNets," CVPR 2020.
    
    References:
        https://github.com/zeiss-microscopy/BSConv
    """

    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : Ints,
        stride        : Ints            = 1,
        padding       : Ints | str      = 0,
        dilation      : Ints            = 1,
        groups        : int             = 1,
        bias          : bool            = True,
        padding_mode  : str             = "zeros",
        device        : Any             = None,
        dtype         : Any             = None,
        act           : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        self.pw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act     = to_act_layer(act=act, num_features=out_channels)
        self.dw_conv = Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        
    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.pw_conv(x)
        if self.act is not None:
            x = self.act(x)
        x = self.dw_conv(x)
        return x


BSConv2dS            = SubspaceBlueprintSeparableConv2d
BSConv2dU            = UnconstrainedBlueprintSeparableConv2d
Conv1d               = nn.Conv1d
Conv2d               = nn.Conv2d
Conv3d               = nn.Conv3d
ConvNormActivation   = torchvision.ops.misc.ConvNormActivation
Conv2dNormActivation = torchvision.ops.Conv2dNormActivation
ConvTranspose1d      = nn.ConvTranspose1d
ConvTranspose2d      = nn.ConvTranspose2d
ConvTranspose3d      = nn.ConvTranspose3d
LazyConv1d           = nn.LazyConv1d
LazyConv2d           = nn.LazyConv2d
LazyConv3d           = nn.LazyConv3d
LazyConvTranspose1d  = nn.LazyConvTranspose1d
LazyConvTranspose2d  = nn.LazyConvTranspose2d
LazyConvTranspose3d  = nn.LazyConvTranspose3d


# H2: - Cropping ---------------------------------------------------------------

class CropTBLR(Module):
    """
    Crop tensor with top + bottom + left + right value.
    
    Args:
        top (int): Top padding.
        bottom (int): Bottom padding.
        left (int): Left padding.
        right (int): Right padding.
        inplace (bool): If True, make this operation inplace. Defaults to False.
    """

    def __init__(
        self,
        top    : int,
        bottom : int,
        left   : int,
        right  : int,
        inplace: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.top     = top
        self.bottom  = bottom
        self.left    = left
        self.right   = right
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        import one.vision.transformation as t
        return t.crop_tblr(
            image  = input,
            top    = self.top,
            bottom = self.bottom,
            left   = self.left,
            right  = self.right,
        )
    

# H2: - Dropout ----------------------------------------------------------------

def drop_block_2d(
    input      : Tensor,
    drop_prob  : float = 0.1,
    block_size : int   = 7,
    gamma_scale: float = 1.0,
    with_noise : bool  = False,
    inplace    : bool  = False,
    batchwise  : bool  = False, 
    *args, **kwargs
) -> Tensor:
    """
    DropBlock with an experimental gaussian noise option. This layer has been
    tested on a few training runs with success, but needs further validation
    and possibly optimization for lower runtime impact.
    
    Papers: `DropBlock: A regularization method for convolutional networks`
    (https://arxiv.org/abs/1810.12890)
    """
    b, c, h, w         = input.shape
    total_size         = w * h
    clipped_block_size = min(block_size, min(w, h))
    # seed_drop_rate, the gamma parameter
    gamma = (gamma_scale * drop_prob * total_size / clipped_block_size ** 2 /
             ((w - block_size + 1) * (h - block_size + 1)))

    # Forces the block to be inside the feature map.
    w_i, h_i    = torch.meshgrid(torch.arange(w).to(input.device),
                                 torch.arange(h).to(input.device))
    valid_block = (
        ((w_i >= clipped_block_size // 2) &
         (w_i < w - (clipped_block_size - 1) // 2)) &
        ((h_i >= clipped_block_size // 2) &
         (h_i < h - (clipped_block_size - 1) // 2))
    )
    valid_block = torch.reshape(valid_block, (1, 1, h, w)).to(dtype=input.dtype)

    if batchwise:
        # One mask for whole batch, quite a bit faster
        uniform_noise = torch.rand((1, c, h, w), dtype=input.dtype,
                                   device=input.device)
    else:
        uniform_noise = torch.rand_like(input)
    block_mask = ((2 - gamma - valid_block + uniform_noise) >= 1)
    block_mask = block_mask.to(dtype=input.dtype)
    block_mask = -F.max_pool2d(
        -block_mask,
        kernel_size = clipped_block_size,
        # block_size,
        stride      = 1,
        padding     = clipped_block_size // 2
    )

    if with_noise:
        normal_noise = (
            torch.randn((1, c, h, w), dtype=input.dtype, device=input.device)
            if batchwise else torch.randn_like(input)
        )
        if inplace:
            input.mul_(block_mask).add_(normal_noise * (1 - block_mask))
        else:
            input = input * block_mask + normal_noise * (1 - block_mask)
    else:
        normalize_scale = (block_mask.numel() /
                           block_mask.to(dtype=torch.float32).sum().add(1e-7))
        normalize_scale = normalize_scale.to(input.dtype)
        if inplace:
            input.mul_(block_mask * normalize_scale)
        else:
            input = input * block_mask * normalize_scale
    return input


def drop_block_fast_2d(
    input      : Tensor,
    drop_prob  : float = 0.1,
    block_size : int   = 7,
    gamma_scale: float = 1.0,
    with_noise : bool  = False,
    inplace    : bool  = False,
    batchwise  : bool  = False,
    *args, **kwargs
) -> Tensor:
    """
    DropBlock with an experimental gaussian noise option. Simplified from
    above without concern for valid block mask at edges.

    Papers: `DropBlock: A regularization method for convolutional networks`
    (https://arxiv.org/abs/1810.12890)
    """
    b, c, h, w 		   = input.shape
    total_size		   = w * h
    clipped_block_size = min(block_size, min(w, h))
    gamma = (gamma_scale * drop_prob * total_size / clipped_block_size ** 2 /
             ((w - block_size + 1) * (h - block_size + 1)))

    if batchwise:
        # One mask for whole batch, quite a bit faster
        block_mask = torch.rand((1, c, h, w), dtype=input.dtype, device=input.device)
        block_mask = block_mask < gamma
    else:
        # Mask per batch element
        block_mask = torch.rand_like(input) < gamma
    block_mask = F.max_pool2d(
        block_mask.to(input.dtype), kernel_size=clipped_block_size, stride=1,
        padding=clipped_block_size // 2
    )

    if with_noise:
        normal_noise = (
            torch.randn((1, c, h, w), dtype=input.dtype, device=input.device)
            if batchwise else torch.randn_like(input)
        )
        if inplace:
            input.mul_(1.0 - block_mask).add_(normal_noise * block_mask)
        else:
            input = input * (1.0 - block_mask) + normal_noise * block_mask
    else:
        block_mask 	    = 1 - block_mask
        normalize_scale = (block_mask.numel() /
                           block_mask.to(dtype=torch.float32).sum().add(1e-7))
        normalize_scale = normalize_scale.to(dtype=input.dtype)
        if inplace:
            input.mul_(block_mask * normalize_scale)
        else:
            input = input * block_mask * normalize_scale
    return input


def drop_path(
    input    : Tensor,
    drop_prob: float = 0.0,
    training : bool  = False,
    *args, **kwargs
) -> Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks). We follow the implementation:
    https://github.com/rwightman/pytorch-image-models/blob
    /a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py

    Args:
        input (Tensor): Input.
        drop_prob (float): Probability of the path to be zeroed. Defaults to 0.0.
        training (bool): Is in training run?. Defaults to False.
    """
    if drop_prob == 0.0 or not training:
        return input
    
    keep_prob     = 1 - drop_prob
    shape	      = (input.shape[0],) + (1,) * (input.ndim - 1)
    random_tensor = (keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device))
    output 		  = input.div(keep_prob) * random_tensor.floor()
    return output


class DropBlock2d(Module):
    """
    DropBlock.
    """

    def __init__(
        self,
        drop_prob  : float = 0.1,
        block_size : int   = 7,
        gamma_scale: float = 1.0,
        with_noise : bool  = False,
        inplace    : bool  = False,
        batchwise  : bool  = False,
        fast       : bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.drop_prob   = drop_prob
        self.gamma_scale = gamma_scale
        self.block_size  = block_size
        self.with_noise  = with_noise
        self.inplace     = inplace
        self.batchwise   = batchwise
        self.fast        = fast  # FIXME finish comparisons of fast vs not

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or not self.drop_prob:
            return input
        if self.fast:
            return drop_block_fast_2d(
                input       = input,
                drop_prob   = self.drop_prob,
                block_size  = self.block_size,
                gamma_scale = self.gamma_scale,
                with_noise  = self.with_noise,
                inplace     = self.inplace,
                batchwise   = self.batchwise
            )
        else:
            return drop_block_2d(
                input       = input,
                drop_prob   = self.drop_prob,
                block_size  = self.block_size,
                gamma_scale = self.gamma_scale,
                with_noise  = self.with_noise,
                inplace     = self.inplace,
                batchwise   = self.batchwise
            )


class DropPath(Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Args:
        drop_prob (float): Probability of the path to be zeroed. Defaults to 0.1.
    """
    
    def __init__(self, drop_prob: float = 0.1, *args, **kwargs):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, input: Tensor) -> Tensor:
        return drop_path(
            input     = input,
            drop_prob = self.drop_prob,
            training  = self.training
        )


AlphaDropout        = nn.AlphaDropout
Dropout             = nn.Dropout
Dropout1d           = nn.Dropout1d
Dropout2d           = nn.Dropout2d
Dropout3d           = nn.Dropout3d
FeatureAlphaDropout = nn.FeatureAlphaDropout


# H2: - Extract ----------------------------------------------------------------

class ExtractFeature(Module):
    """
    Extract a feature at `index` in a tensor.
    
    Args:
        index (int): The index of the feature to extract.
    """
    
    def __init__(
        self,
        index: int,
        dim  : int = 1,
        *args, **kwargs
    ):
        super().__init__()
        self.index = index
    
    def forward(self, input: Tensor) -> Tensor:
        assert_tensor_of_ndim(input, 4)
        return input[:, self.index, :, :]


class ExtractFeatures(Module):
    """
    Extract features between `start` index and `end` index in a tensor.
    
    Args:
        start (int): The start index of the features to extract.
        end (int): The end index of the features to extract.
    """
    
    def __init__(
        self,
        start: int,
        end  : int,
        *args, **kwargs
    ):
        super().__init__()
        self.start = start
        self.end   = end
    
    def forward(self, input: Tensor) -> Tensor:
        assert_tensor_of_ndim(input, 4)
        return input[:, self.start:self.end, :, :]
    

class ExtractItem(Module):
    """
    Extract an item (feature) at `index` in a sequence of tensors.
    
    Args:
        index (int): The index of the item to extract.
    """
    
    def __init__(
        self,
        index: int,
        *args, **kwargs
    ):
        super().__init__()
        self.index = index
    
    def forward(self, input: Tensors) -> Tensor:
        if isinstance(input, Tensor):
            return input
        elif isinstance(input, (list, tuple)):
            return input[self.index]
        else:
            raise TypeError(
                f"`input` must be a list or tuple of tensors. "
                f"But got: {type(input)}."
            )
    

class ExtractItems(Module):
    """
    Extract a list of items (features) at `indexes` in a sequence of tensors.
    
    Args:
        indexes (Sequence[int]): The indexes of the items to extract.
    """
    
    def __init__(
        self,
        indexes: Sequence[int],
        *args, **kwargs
    ):
        super().__init__()
        self.indexes = indexes
    
    def forward(self, input: Tensors) -> list[Tensor]:
        if isinstance(input, Tensor):
            return [input]
        elif isinstance(input, (list, tuple)):
            return [input[i] for i in self.indexes]
        raise TypeError(
            f"`input` must be a list or tuple of tensors. "
            f"But got: {type(input)}."
        )
    
    
class Max(Module):
    """
    """
    
    def __init__(
        self,
        dim    : int,
        keepdim: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.dim     = dim
        self.keepdim = keepdim
    
    def forward(self, input: Tensors) -> Tensor:
        return torch.max(input=input, dim=self.dim, keepdim=self.keepdim)


# H2: - Fusion -----------------------------------------------------------------

class Concat(Module):
    """
    Concatenate a list of tensors along dimension.
    
    Args:
        dim (str | ellipsis | None): Dimension to concat to. Defaults to 1.
    """
    
    def __init__(
        self,
        dim: str | ellipsis | None = 1,
        *args, **kwargs
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, input: Sequence[Tensor]) -> Tensor:
        return torch.cat(to_list(input), dim=self.dim)


class Chuncat(Module):
    """
    
    Args:
        dim (str | ellipsis | None): Dimension to concat to. Defaults to 1.
    """
    
    def __init__(
        self,
        dim: str | ellipsis | None = 1,
        *args, **kwargs
    ):
        super().__init__()
        self.dim = dim

    def forward(self, input: Sequence[Tensor]) -> Tensor:
        x1 = []
        x2 = []
        for x_i in input:
            x_i_1, x_i_2 = x_i.chunk(2, self.dim)
            x1.append(x_i_1)
            x2.append(x_i_2)
        return torch.cat(x1 + x2, dim=self.dim)


class Foldcut(nn.Module):
    """
    
    Args:
        dim (str | ellipsis | None): Dimension to concat to. Defaults to 0.
    """
    
    def __init__(
        self,
        dim: str | ellipsis | None = 0,
        *args, **kwargs
    ):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        x1, x2 = input.chunk(2, dim=self.dim)
        return x1 + x2


class InterpolateConcat(Module):
    """
    Concatenate a list of tensors along dimension.
    
    Args:
        dim (str | ellipsis | None): Dimension to concat to. Defaults to 1.
    """
    
    def __init__(
        self,
        dim: str | ellipsis | None = 1,
        *args, **kwargs
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, input: Sequence[Tensor]) -> Tensor:
        sizes  = [list(i.size()) for i in input]
        hs     = [s[2] for s in sizes]
        ws     = [s[3] for s in sizes]
        h, w   = max(hs), max(ws)
        output = []
        for i in input:
            s = i.size()
            if s[2] != h or s[3] != w:
                output.append(F.interpolate(input=i, size=(h, w)))
            else:
                output.append(i)
        return torch.cat(to_list(output), dim=self.dim)


class Join(Module):
    """
    Join multiple features and return a list tensors.
    """
    
    def forward(self, input: Sequence[Tensor]) -> list[Tensor]:
        return to_list(input)
    

class Shortcut(Module):
    """
    
    Args:
        dim (str | ellipsis | None): Dimension to concat to. Defaults to 0.
    """
    
    def __init__(
        self,
        dim: str | ellipsis | None = 0,
        *args, **kwargs
    ):
        super().__init__()
        self.dim = dim

    def forward(self, input: Sequence[Tensor]) -> Tensor:
        return input[0] + input[1]


class SoftmaxFusion(Module):
    """
    Weighted sum of multiple layers https://arxiv.org/abs/1911.09070. Apply
    softmax to each weight, such that all weights are normalized to be a
    probability with value range from 0 to 1, representing the importance of
    each input
    
    Args:
        n (int): Number of inputs.
    """

    def __init__(
        self,
        n     : int,
        weight: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.weight = weight  # Apply weights boolean
        self.iter 	= range(n - 1)  # iter object
        if weight:
            # Layer weights
            self.w = Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)
    
    def forward(self, input: Tensor) -> Tensor:
        output = input[0]
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                output = output + input[i + 1] * w[i]
        else:
            for i in self.iter:
                output = output + input[i + 1]
        return output


class Sum(Module):
    
    def forward(self, input: Sequence[Tensor]) -> Tensor:
        output = input[0]
        for i in range(1, len(input)):
            output += input[i]
        return output


# H2: - Head -------------------------------------------------------------------

class AlexNetClassifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.drop1        = Dropout()
        self.linear1      = Linear(in_features=in_channels * 6 * 6, out_features=4096)
        self.act1         = ReLU(inplace=True)
        self.drop2        = Dropout()
        self.linear2      = Linear(in_features=4096, out_features=4096)
        self.act2         = ReLU(inplace=True)
        self.linear3      = Linear(in_features=4096, out_features=out_channels)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = torch.flatten(input, 1)
            x = self.drop1(x)
            x = self.act1(self.linear1(x))
            x = self.drop2(x)
            x = self.act2(self.linear2(x))
            x = self.linear3(x)
            return x
        else:
            return input


class ConvNeXtClassifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        norm        : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.norm         = norm(in_channels)
        self.flatten      = Flatten(1)
        self.linear       = Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = self.norm(input)
            x = self.flatten(x)
            x = self.linear(x)
            return x
        else:
            return input
        

class GoogleNetClassifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        dropout     : float,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.dropout      = dropout
        self.avgpool      = AdaptiveAvgPool2d((1, 1))
        self.dropout      = Dropout(p=dropout)
        self.fc           = Linear(in_features=in_channels, out_features=out_channels)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = self.avgpool(input)
            # N x 1024 x 1 x 1
            x = torch.flatten(x, 1)
            # N x 1024
            x = self.dropout(x)
            x = self.fc(x)
            # N x 1000 (num_classes)
            return x
        else:
            return input


class InceptionClassifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.avgpool      = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout      = nn.Dropout(p=0.5)
        self.fc           = nn.Linear(in_channels, out_channels)

    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = input
            # Adaptive average pooling
            x = self.avgpool(x)
            # N x 2048 x 1 x 1
            x = self.dropout(x)
            # N x 2048 x 1 x 1
            x = torch.flatten(x, 1)
            # N x 2048
            x = self.fc(x)
            # N x 1000 (num_classes)
            return x
        else:
            return input
    
    
class LeNetClassifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.linear1 = Linear(in_features=in_channels, out_features=84)
        self.act1    = Tanh()
        self.linear2 = Linear(in_features=84, out_features=out_channels)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = self.linear1(input)
            x = self.act1(x)
            x = self.linear2(x)
            return x
        else:
            return input


class LinearClassifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.linear = Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = torch.flatten(input, 1)
            x = self.linear(x)
            return x
        else:
            return input
        

class ShuffleNetV2Classifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.linear = Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = input.mean([2, 3])  # globalpool
            x = self.linear(x)
            return x
        else:
            return input
        

class SqueezeNetClassifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        dropout     : float = 0.5,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.dropout = Dropout(p=dropout)
        self.conv    = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
        )
        self.act     = ReLU(inplace=True)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
    
    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = self.dropout(input)
            x = self.conv(x)
            x = self.act(x)
            x = self.avgpool(x)
            x = torch.flatten(x, dims=1)
            return x
        else:
            return input


class VGGClassifier(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        self.linear1      = Linear(in_features=in_channels * 7 * 7, out_features=4096)
        self.act1         = ReLU(inplace=True)
        self.drop1        = Dropout()
        self.linear2      = Linear(in_features=4096, out_features=4096)
        self.act2         = ReLU(inplace=True)
        self.drop2        = Dropout()
        self.linear3      = Linear(in_features=4096, out_features=out_channels)
    
    def forward(self, input: Tensor) -> Tensor:
        if self.out_channels > 0:
            x = torch.flatten(input, 1)
            x = self.act1(self.linear1(x))
            x = self.drop1(x)
            x = self.act2(self.linear2(x))
            x = self.drop2(x)
            x = self.linear3(x)
            return x
        else:
            return input
        

# H2: - Linear -----------------------------------------------------------------

class Flatten(Module):
    """
    Flatten a tensor along a dimension.
    
    Args:
        dim (int): Dimension to flatten. Defaults to 1.
    """
    
    def __init__(
        self,
        dim: int = 1,
        *args, **kwargs
    ):
        super().__init__()
        self.dim = dim
        
    def forward(self, input: Tensor) -> Tensor:
        return torch.flatten(input, self.dim)


Bilinear   = nn.Bilinear  
Identity   = nn.Identity  
LazyLinear = nn.LazyLinear
Linear     = nn.Linear    


# H2: - Normalization ----------------------------------------------------------

class BatchNormAct2d(BatchNorm2d):
    """
    BatchNorm2d + Activation.
    
    This module performs BatchNorm2d + Activation in a manner that will remain
    backwards compatible with weights trained with separate bn, act. This is
    why we inherit from BN instead of composing it as a .bn member.
    """
    
    def __init__(
        self,
        num_features       : int,
        eps                : float           = 1e-5,
        momentum           : float           = 0.1,
        affine             : bool            = True,
        track_running_stats: bool            = True,
        device             : Any             = None,
        dtype              : Any             = None,
        act                : Callable | None = ReLU(),
        inplace            : bool            = True,
        *args, **kwargs
    ):
        super().__init__(
            num_features        = num_features,
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype
        )
        self.act = to_act_layer(act, inplace)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        if self.act is not None:
            output = self.act(output)
        return output


class BatchNormReLU2d(BatchNormAct2d):
    """
    BatchNorm2d + ReLU.

    This module performs BatchNorm2d + ReLU in a manner that will remain
    backwards compatible with weights trained with separate bn, act. This is
    why we inherit from BN instead of composing it as a .bn member.
    """
    
    def __init__(
        self,
        num_features       : int,
        eps                : float           = 1e-5,
        momentum           : float           = 0.1,
        affine             : bool            = True,
        track_running_stats: bool            = True,
        device             : Any             = None,
        dtype              : Any             = None,
        inplace            : bool            = True,
        drop_block         : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__(
            num_features        = num_features,
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype,
            act                 = ReLU(),
            inplace             = inplace,
            drop_block          = drop_block
        )
        

class FractionInstanceNorm2d(InstanceNorm2d):
    """
    Fractional Instance Normalization is a generalization of Half Instance
    Normalization.
    
    Args:
        num_features (int): Number of input features.
        alpha (float): Ratio of input features that will be normalized.
            Defaults to 0.5.
        selection (str): Feature selection mechanism.
            One of: ["linear", "random", "interleave"]
                - "linear"    : normalized only first half.
                - "random"    : randomly choose features to normalize.
                - "interleave": interleaving choose features to normalize.
            Defaults to linear.
    """
    
    def __init__(
        self,
        num_features       : int,
        alpha              : float = 0.5,
        selection          : str   = "linear",
        eps                : float = 1e-5,
        momentum           : float = 0.1,
        affine             : bool  = True,
        track_running_stats: bool  = False,
        device             : Any   = None,
        dtype              : Any   = None,
        *args, **kwargs
    ):
        self.in_channels = num_features
        self.alpha       = alpha
        self.selection   = selection
        super().__init__(
            num_features        = math.ceil(num_features * self.alpha),
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype,
        )

        if self.selection not in ["linear", "random", "interleave"]:
            raise ValueError()
     
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        _, c, _, _ = input.shape
        
        if self.alpha == 0.0:
            return input
        elif self.alpha == 1.0:
            return F.instance_norm(
                input           = input,
                running_mean    = self.running_mean,
                running_var     = self.running_var,
                weight          = self.weight,
                bias            = self.bias,
                use_input_stats = self.training or not self.track_running_stats,
                momentum        = self.momentum,
                eps             = self.eps
            )
        else:
            if self.selection == "random":
                out1_idxes = random.sample(range(self.in_channels), self.num_features)
                out2_idxes = list(set(range(self.in_channels)) - set(out1_idxes))
                out1_idxes = Tensor(out1_idxes).to(torch.int).to(input.device)
                out2_idxes = Tensor(out2_idxes).to(torch.int).to(input.device)
                out1       = torch.index_select(input, 1, out1_idxes)
                out2       = torch.index_select(input, 1, out2_idxes)
            elif self.selection == "interleave":
                skip       = int(math.floor(self.in_channels / self.num_features))
                out1_idxes = []
                for i in range(0, self.in_channels, skip):
                    if len(out1_idxes) < self.num_features:
                        out1_idxes.append(i)
                out2_idxes = list(set(range(self.in_channels)) - set(out1_idxes))
                # print(len(out1_idxes), len(out2_idxes), self.num_features)
                out1_idxes = Tensor(out1_idxes).to(torch.int).to(input.device)
                out2_idxes = Tensor(out2_idxes).to(torch.int).to(input.device)
                out1       = torch.index_select(input, 1, out1_idxes)
                out2       = torch.index_select(input, 1, out2_idxes)
            else:  # Half-Half
                split_size = [self.num_features, c - self.num_features]
                out1, out2 = torch.split(input, split_size, dim=1)
            
            out1 = F.instance_norm(
                input           = out1,
                running_mean    = self.running_mean,
                running_var     = self.running_var,
                weight          = self.weight,
                bias            = self.bias,
                use_input_stats = self.training or not self.track_running_stats,
                momentum        = self.momentum,
                eps             = self.eps
            )
            return torch.cat([out1, out2], dim=1)


class GroupNormAct(GroupNorm):
    """
    GroupNorm + Activation.

    This module performs GroupNorm + Activation in a manner that will remain
    backwards compatible with weights trained with separate gn, act. This is
    why we inherit from GN instead of composing it as a .gn member.
    """

    def __init__(
        self,
        num_groups  : int,
        num_channels: int,
        eps         : float           = 1e-5,
        affine      : bool            = True,
        device      : Any             = None,
        dtype       : Any             = None,
        act         : Callable | None = ReLU,
        inplace     : bool            = True,
        *args, **kwargs
    ):
        super().__init__(
            num_groups   = num_groups,
            num_channels = num_channels,
            eps          = eps,
            affine       = affine,
            device       = device,
            dtype        = dtype
        )
        self.act = to_act_layer(act, inplace)
    
    def forward(self, input: Tensor) -> Tensor:
        output = F.group_norm(
            input      = input,
            num_groups = self.num_groups,
            weight     = self.weight,
            bias       = self.bias,
            eps        = self.eps
        )
        return self.act(output)


class HalfGroupNorm(GroupNorm):

    def __init__(
        self,
        num_groups  : int,
        num_channels: int,
        eps         : float = 1e-5,
        affine      : bool  = True,
        device      : Any   = None,
        dtype       : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            num_groups   = num_groups,
            num_channels = num_channels,
            eps          = eps,
            affine       = affine,
            device       = device,
            dtype        = dtype
        )

    def forward(self, input: Tensor) -> Tensor:
        out_1, out_2 = torch.chunk(input, 2, dim=1)
        out_1        = F.group_norm(
            input      = out_1,
            num_groups = self.num_groups,
            weight     = self.weight,
            bias       = self.bias,
            eps        = self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


class HalfInstanceNorm2d(InstanceNorm2d):
    """
    Half instance normalization layer proposed in paper:
    
    """
    
    def __init__(
        self,
        num_features       : int,
        eps                : float = 1e-5,
        momentum           : float = 0.1,
        affine             : bool  = True,
        track_running_stats: bool  = False,
        device             : Any   = None,
        dtype              : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            num_features        = math.ceil(num_features / 2),
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype,
        )
        
    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        out_1, out_2 = torch.chunk(input, 2, dim=1)
        out_1        = F.instance_norm(
            input           = out_1,
            running_mean    = self.running_mean,
            running_var     = self.running_var,
            weight          = self.weight,
            bias            = self.bias,
            use_input_stats = self.training or not self.track_running_stats,
            momentum        = self.momentum,
            eps             = self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


class HalfLayerNorm(LayerNorm):

    def __init__(
        self,
        normalized_shape  : Any,
        eps               : float = 1e-5,
        elementwise_affine: bool  = True,
        device            : Any   = None,
        dtype             : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            normalized_shape   = normalized_shape,
            eps                = eps,
            elementwise_affine = elementwise_affine,
            device             = device,
            dtype              = dtype
        )

    def forward(self, input: Tensor) -> Tensor:
        out_1, out_2 = torch.chunk(input, 2, dim=1)
        out_1        = F.layer_norm(
            input            = out_1,
            normalized_shape = self.normalized_shape,
            weight           = self.weight,
            bias             = self.bias,
            eps              = self.eps
        )
        return torch.cat([out_1, out_2], dim=1)


class LayerNorm2d(LayerNorm):
    """
    LayerNorm for channels of 2D spatial [B, C, H, W] tensors.
    """

    def __init__(
        self,
        normalized_shape  : Any,
        eps               : float = 1e-5,
        elementwise_affine: bool  = True,
        device            : Any   = None,
        dtype             : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            normalized_shape   = normalized_shape,
            eps                = eps,
            elementwise_affine = elementwise_affine,
            device             = device,
            dtype              = dtype
        )

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input            = input.permute(0, 2, 3, 1),
            normalized_shape = self.normalized_shape,
            weight           = self.weight,
            bias             = self.bias,
            eps              = self.eps
        ).permute(0, 3, 1, 2)


BatchNorm1d        = nn.BatchNorm1d       
BatchNorm2d        = nn.BatchNorm2d       
BatchNorm3d        = nn.BatchNorm3d       
GroupNorm          = nn.GroupNorm         
LayerNorm          = nn.LayerNorm         
LazyBatchNorm1d    = nn.LazyBatchNorm1d   
LazyBatchNorm2d    = nn.LazyBatchNorm2d   
LazyBatchNorm3d    = nn.LazyBatchNorm3d   
LazyInstanceNorm1d = nn.LazyInstanceNorm1d
LazyInstanceNorm2d = nn.LazyInstanceNorm2d
LazyInstanceNorm3d = nn.LazyInstanceNorm3d
LocalResponseNorm  = nn.LocalResponseNorm 
InstanceNorm1d     = nn.InstanceNorm1d    
InstanceNorm2d     = nn.InstanceNorm2d    
InstanceNorm3d     = nn.InstanceNorm3d    
SyncBatchNorm      = nn.SyncBatchNorm     


# H2: - Padding ----------------------------------------------------------------

def get_same_padding(
    x          : int,
    kernel_size: int,
    stride     : int,
    dilation   : int
) -> int:
    """
    Calculate asymmetric TensorFlow-like 'same' padding value for 1
    dimension of the convolution.
    """
    return max((math.ceil(x / stride) - 1) * stride +
               (kernel_size - 1) * dilation + 1 - x, 0)


def get_symmetric_padding(
    kernel_size: int,
    stride     : int = 1,
    dilation   : int = 1,
    *args, **kwargs
) -> int:
    """
    Calculate symmetric padding for a convolution.
    """
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def to_same_padding(
    kernel_size: Ints,
    padding    : Ints | None = None,
    *args, **kwargs
) -> int | list | None:
    """
    It takes a kernel size and a padding, and if the padding is None, it returns
    None, otherwise it returns the kernel size divided by 2.
    
    Args:
        kernel_size (Ints): The size of the convolutional kernel.
        padding (Ints | None): The padding to use for the convolution.
    
    Returns:
        The padding is being returned.
    """
    if padding is None:
        if isinstance(kernel_size, int):
            return kernel_size // 2
        if isinstance(kernel_size, (tuple, list)):
            return [k // 2 for k in kernel_size]
    return padding


def pad_same(
    input      : Tensor,
    kernel_size: Ints,
    stride     : Ints,
    dilation   : Ints  = (1, 1),
    value      : float = 0,
    *args, **kwargs
):
    """
    Dynamically pad input tensor with 'same' padding for conv with specified
    args.
    """
    ih, iw = input.size()[-2:]
    pad_h  = get_same_padding(ih, kernel_size[0], stride[0], dilation[0])
    pad_w  = get_same_padding(iw, kernel_size[1], stride[1], dilation[1])
    if pad_h > 0 or pad_w > 0:
        input = F.pad(
            input = input,
            pad   = [pad_w // 2, pad_w - pad_w // 2,
                     pad_h // 2, pad_h - pad_h // 2],
            value = value
        )
    return input


ConstantPad1d    = nn.ConstantPad1d   
ConstantPad2d    = nn.ConstantPad2d   
ConstantPad3d    = nn.ConstantPad3d   
ReflectionPad1d  = nn.ReflectionPad1d 
ReflectionPad2d  = nn.ReflectionPad2d 
ReflectionPad3d  = nn.ReflectionPad3d 
ReplicationPad1d = nn.ReplicationPad1d
ReplicationPad2d = nn.ReplicationPad2d
ReplicationPad3d = nn.ReplicationPad3d
ZeroPad2d        = nn.ZeroPad2d       


# H2: - Pooling ----------------------------------------------------------------

def adaptive_avg_max_pool2d(
    input      : Tensor,
    output_size: int = 1,
    *args, **kwargs
) -> Tensor:
    avg = F.adaptive_avg_pool2d(input, output_size)
    max = F.adaptive_max_pool2d(input, output_size)
    return 0.5 * (avg + max)


def adaptive_cat_avg_max_pool2d(
    input      : Tensor,
    output_size: int = 1,
    *args, **kwargs
) -> Tensor:
    avg = F.adaptive_avg_pool2d(input, output_size)
    max = F.adaptive_max_pool2d(input, output_size)
    return torch.cat((avg, max), 1)


def adaptive_pool2d(
    input      : Tensor,
    pool_type  : str = "avg",
    output_size: int = 1,
    *args, **kwargs
) -> Tensor:
    """
    Selectable global pooling function with dynamic input kernel size.
    """
    if pool_type == "avg":
        input = F.adaptive_avg_pool2d(input, output_size)
    elif pool_type == "avg_max":
        input = adaptive_avg_max_pool2d(input, output_size)
    elif pool_type == "cat_avg_max":
        input = adaptive_cat_avg_max_pool2d(input, output_size)
    elif pool_type == "max":
        input = F.adaptive_max_pool2d(input, output_size)
    elif True:
        raise ValueError("Invalid pool type: %s" % pool_type)
    return input


def avg_pool_same2d(
    input            : Tensor,
    kernel_size      : Ints,
    stride           : Ints,
    padding          : Ints = 0,
    ceil_mode        : bool = False,
    count_include_pad: bool = True,
    *args, **kwargs
) -> Tensor:
    input = pad_same(input=input, kernel_size=kernel_size, stride=stride)
    return F.avg_pool2d(
        input             = input,
        kernel_size       = kernel_size,
        stride            = stride,
        padding           = padding,
        ceil_mode         = ceil_mode,
        count_include_pad = count_include_pad
    )


def max_pool_same2d(
    input      : Tensor,
    kernel_size: Ints,
    stride     : Ints,
    padding    : Ints = 0,
    dilation   : Ints = 1,
    ceil_mode  : bool = False,
    *args, **kwargs
) -> Tensor:
    input = pad_same(
        input       = input,
        kernel_size = kernel_size,
        stride      = stride,
        value       = -float("inf")
    )
    return F.max_pool2d(
        input       = input,
        kernel_size = kernel_size,
        stride      = stride,
        padding     = padding,
        dilation    = dilation,
        ceil_mode   = ceil_mode
    )


class AdaptiveAvgMaxPool2d(Module):

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_avg_max_pool2d(
            input       = input,
            output_size = self.output_size
        )


class AdaptiveCatAvgMaxPool2d(Module):

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_cat_avg_max_pool2d(
            input       = input,
            output_size = self.output_size
        )


class AdaptivePool2d(Module):
    """
    Selectable global pooling layer with dynamic input kernel size.
    """

    def __init__(
        self,
        output_size: int  = 1,
        pool_type  : str  = "fast",
        flatten    : bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.pool_type = pool_type or ""
        self.flatten   = Flatten(1) if flatten else Identity()
        if pool_type == "":
            self.pool = Identity()  # pass through
        elif pool_type == "fast":
            if output_size != 1:
                raise ValueError()
            self.pool    = FastAdaptiveAvgPool2d(flatten)
            self.flatten = Identity()
        elif pool_type == "avg":
            self.pool = AdaptiveAvgPool2d(output_size)
        elif pool_type == "avg_max":
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == "cat_avg_max":
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == "max":
            self.pool = AdaptiveMaxPool2d(output_size)
        elif True:
            raise ValueError("Invalid pool type: %s" % pool_type)

    def __repr__(self):
        return (self.__class__.__name__ + " (pool_type=" + self.pool_type +
                ", flatten=" + str(self.flatten) + ")")

    def is_identity(self) -> bool:
        return not self.pool_type

    def forward(self, input: Tensor) -> Tensor:
        output = self.pool(input)
        output = self.flatten(output)
        return output

    def feat_mult(self):
        if self.pool_type == "cat_avg_max":
            return 2
        else:
            return 1
        

class AvgPoolSame2d(AvgPool2d):
    """
    Tensorflow like 'same' wrapper for 2D average pooling.
    """

    def __init__(
        self,
        kernel_size      : Ints,
        stride           : Ints | None = None,
        padding          : Ints        = 0,
        ceil_mode        : bool        = False,
        count_include_pad: bool        = True,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        super().__init__(
            kernel_size       = kernel_size,
            stride            = stride,
            padding           = padding,
            ceil_mode         = ceil_mode,
            count_include_pad = count_include_pad
        )

    def forward(self, input: Tensor) -> Tensor:
        output = pad_same(
            input       = input,
            kernel_size = self.kernel_size,
            stride      = self.stride
        )
        return F.avg_pool2d(
            input             = output,
            kernel_size       = self.kernel_size,
            stride            = self.stride,
            padding           = self.padding,
            ceil_mode         = self.ceil_mode,
            count_include_pad = self.count_include_pad
        )


class FastAdaptiveAvgPool2d(Module):

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.flatten = flatten

    def forward(self, input: Tensor) -> Tensor:
        return input.mean((2, 3), keepdim=not self.flatten)
    

class MaxPoolSame2d(MaxPool2d):
    """
    Tensorflow like `same` wrapper for 2D max pooling.
    """

    def __init__(
        self,
        kernel_size: Ints,
        stride     : Ints | None = None,
        padding    : Ints | None = (0, 0),
        dilation   : Ints        = (1, 1),
        ceil_mode  : bool        = False,
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        super().__init__(
            kernel_size = kernel_size,
            stride      = stride,
            padding     = padding,
            dilation    = dilation,
            ceil_mode   = ceil_mode
        )

    def forward(self, input: Tensor) -> Tensor:
        output = pad_same(
            input       = input,
            kernel_size = self.kernel_size,
            stride      = self.stride,
            value       = -float("inf")
        )
        return F.max_pool2d(
            input             = output,
            kernel_size       = self.kernel_size,
            stride            = self.stride,
            padding           = self.padding,
            ceil_mode         = self.dilation,
            count_include_pad = self.ceil_mode
        )


class MedianPool2d(Module):
    """
    Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size (Ints): Size of pooling kernel.
         stride (Ints): Pool stride, int or 2-tuple
         padding (str | Ints | None): Pool padding, int or 4-tuple (ll, r, t, b)
            as in pytorch F.pad.
         same (bool): Override padding and enforce same padding.
            Defaults to False.
    """

    def __init__(
        self,
        kernel_size: Ints,
        stride     : Ints       = (1, 1),
        padding    : Ints | str = 0,
        same	   : bool	    = False,
        *args, **kwargs
    ):
        super().__init__()
        self.kernel_size = to_2tuple(kernel_size)
        self.stride 	 = to_2tuple(stride)
        self.padding 	 = to_4tuple(padding)  # convert to ll, r, t, b
        self.same	 	 = same

    def _padding(self, input: Tensor):
        if self.same:
            ih, iw = input.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.kernel_size[0] - self.stride[0], 0)
            else:
                ph = max(self.kernel_size[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.kernel_size[1] - self.stride[1], 0)
            else:
                pw = max(self.kernel_size[1] - (iw % self.stride[1]), 0)
            pl      = pw // 2
            pr      = pw - pl
            pt      = ph // 2
            pb      = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, input: Tensor) -> Tensor:
        output = F.pad(input, self._padding(input), mode="reflect")
        output = output.unfold(2, self.k[0], self.stride[0])
        output = output.unfold(3, self.k[1], self.stride[1])
        output = output.contiguous().view(output.size()[:4] + (-1,)).median(dim=-1)[0]
        return output


AdaptiveAvgPool1d   = nn.AdaptiveAvgPool1d  
AdaptiveAvgPool2d   = nn.AdaptiveAvgPool2d  
AdaptiveAvgPool3d   = nn.AdaptiveAvgPool3d  
AdaptiveMaxPool1d   = nn.AdaptiveMaxPool1d  
AdaptiveMaxPool2d   = nn.AdaptiveMaxPool2d  
AdaptiveMaxPool3d   = nn.AdaptiveMaxPool3d  
AvgPool1d           = nn.AvgPool1d          
AvgPool2d           = nn.AvgPool2d          
AvgPool3d           = nn.AvgPool3d          
FractionalMaxPool2d = nn.FractionalMaxPool2d
FractionalMaxPool3d = nn.FractionalMaxPool3d
LPPool1d            = nn.LPPool1d           
LPPool2d            = nn.LPPool2d           
MaxPool1d           = nn.MaxPool1d          
MaxPool2d           = nn.MaxPool2d          
MaxPool3d           = nn.MaxPool3d          
MaxUnpool1d         = nn.MaxUnpool1d        
MaxUnpool2d         = nn.MaxUnpool2d        
MaxUnpool3d         = nn.MaxUnpool3d        


# H2: - Scaling ----------------------------------------------------------------

class Downsample(Module):
    """
    Downsample a given multi-channel 1D (temporal), 2D (spatial) or 3D
    (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs,
    we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (Ints | None): Output spatial sizes
        scale_factor (Floats | None): Multiplier for spatial size. Has to match
            input size if it is a tuple.
        mode (str): The upsampling algorithm. One of [`nearest`, `linear`,
            `bilinear`, `bicubic`, `trilinear`]. Defaults to `nearest`.
        align_corners (bool): If True, the corner pixels of the input and
            output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when `mode` is `linear`,
            `bilinear`, `bicubic`, or `trilinear`. Defaults to False.
        recompute_scale_factor (bool): Recompute the scale_factor for use in
            the interpolation calculation.
            - If `recompute_scale_factor` is True, then `scale_factor` must be
              passed in and `scale_factor` is used to compute the output `size`.
              The computed output `size` will be used  to infer new scales for
              the interpolation. Note that when `scale_factor` is
              floating-point, it may differ from the recomputed `scale_factor`
              due to rounding and precision issues.
            - If `recompute_scale_factor` is False, then `size` or
              `scale_factor` will be used directly for interpolation.
            Defaults to False.
    """
    
    def __init__(
        self,
        size                  : Ints   | None = None,
        scale_factor          : Floats | None = None,
        mode                  : str           = "nearest",
        align_corners         : bool          = False,
        recompute_scale_factor: bool          = False,
        *args, **kwargs
    ):
        super().__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(1.0 / factor) for factor in scale_factor)
        else:
            self.scale_factor = float(1.0 / scale_factor) if scale_factor else None
        self.mode                   = mode
        self.align_corners          = align_corners
        self.recompute_scale_factor = recompute_scale_factor
    
    def forward(self, input: Tensor) -> Tensor:
        if self.size and self.size == list(input[2:]):
            return input
        if self.scale_factor \
            and (self.scale_factor == 1.0
                 or all(s == 1.0 for s in self.scale_factor)):
            return input
        return F.interpolate(
            input                  = input,
            size                   = self.size,
            scale_factor           = self.scale_factor,
            mode                   = self.mode,
            align_corners          = self.align_corners,
            recompute_scale_factor = self.recompute_scale_factor
        )
    
    
class Scale(Module):
    """
    A learnable scale parameter. This layer scales the input by a learnable
    factor. It multiplies a learnable scale parameter of shape (1,) with
    input of any shape.
    
    Args:
        scale (float): Initial value of scale factor. Defaults to 1.0.
    """
    
    def __init__(
        self,
        scale: float = 1.0,
        *args, **kwargs
    ):
        super().__init__()
        self.scale = scale
        
    def forward(self, input: Tensor) -> Tensor:
        return input * self.scale


class Interpolate(Module):
    """
    
    Args:
        size (Ints):
    """
    
    def __init__(
        self,
        size: Ints,
        *args, **kwargs
    ):
        super().__init__()
        self.size = to_size(size)
        
    def forward(self, input: Tensor) -> Tensor:
        return F.interpolate(input=input, size=self.size)


class Upsample(Module):
    """
    Upsample a given multi-channel 1D (temporal), 2D (spatial) or 3D
    (volumetric) data.

    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    Hence, for spatial inputs, we expect a 4D Tensor and for volumetric inputs,
    we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size` to
    calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size (Ints | None): Output spatial sizes
        scale_factor (Floats | None): Multiplier for spatial size. Has to match
            input size if it is a tuple.
        mode (str): The upsampling algorithm. One of [`nearest`, `linear`,
            `bilinear`, `bicubic`, `trilinear`]. Defaults to `nearest`.
        align_corners (bool): If True, the corner pixels of the input and
            output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when `mode` is `linear`,
            `bilinear`, `bicubic`, or `trilinear`. Defaults to False.
        recompute_scale_factor (bool): Recompute the scale_factor for use in
            the interpolation calculation.
            - If `recompute_scale_factor` is True, then `scale_factor` must be
              passed in and `scale_factor` is used to compute the output `size`.
              The computed output `size` will be used  to infer new scales for
              the interpolation. Note that when `scale_factor` is
              floating-point, it may differ from the recomputed `scale_factor`
              due to rounding and precision issues.
            - If `recompute_scale_factor` is False, then `size` or
              `scale_factor` will be used directly for interpolation.
            Defaults to False.
    """
    
    def __init__(
        self,
        size                  : Ints   | None = None,
        scale_factor          : Floats | None = None,
        mode                  : str           = "nearest",
        align_corners         : bool          = False,
        recompute_scale_factor: bool          = False,
        *args, **kwargs
    ):
        super().__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode                   = mode
        self.align_corners          = align_corners
        self.recompute_scale_factor = recompute_scale_factor
    
    def forward(self, input: Tensor) -> Tensor:
        if self.size and self.size == list(input[2:]):
            return input
        if self.scale_factor \
            and (self.scale_factor == 1.0
                 or all(s == 1.0 for s in self.scale_factor)):
            return input
        return F.interpolate(
            input                  = input,
            size                   = self.size,
            scale_factor           = self.scale_factor,
            mode                   = self.mode,
            align_corners          = self.align_corners,
            recompute_scale_factor = self.recompute_scale_factor
        )


UpsamplingNearest2d  = nn.UpsamplingNearest2d
UpsamplingBilinear2d = nn.UpsamplingBilinear2d


# H2: - Shuffle ----------------------------------------------------------------

class ChannelShuffle(Module):
    """
    """
    
    def __init__(
        self,
        groups: int,
        *args, **kwargs
    ):
        super().__init__()
        self.name   = type(self).__name__
        self.groups = groups
    
    def forward(self, input: Tensor) -> Tensor:
        x                  = input
        b, c, h, w         = x.size()
        channels_per_group = c // self.groups
        # reshape
        x = x.view(b, self.groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(b, -1, h, w)
        return x


# H1: - EXPERIMENTAL -----------------------------------------------------------

# H2: - ConvNeXt ---------------------------------------------------------------

class ConvNeXtLayer(Module):
    
    def __init__(
        self,
        dim                  : int,
        layer_scale          : float,
        stochastic_depth_prob: float,
        stage_block_id       : int      | None = None,
        total_stage_blocks   : int      | None = None,
        norm                 : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        sd_prob = stochastic_depth_prob
        if (stage_block_id is not None) and (total_stage_blocks is not None):
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
        if norm is None:
            norm = partial(nn.LayerNorm, eps=1e-6)
        self.block = Sequential(
            Conv2d(
                in_channels  = dim,
                out_channels = dim,
                kernel_size  = 7,
                padding      = 3,
                groups       = dim,
                bias         = True,
            ),
            Permute([0, 2, 3, 1]),
            norm(dim),
            Linear(
                in_features  = dim,
                out_features = 4 * dim,
                bias         = True,
            ),
            GELU(),
            Linear(
                in_features  = 4 * dim,
                out_features = dim,
                bias         = True,
            ),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale      = Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(sd_prob, "row")
    
    def forward(self, input: Tensor) -> Tensor:
        output  = self.layer_scale * self.block(input)
        output  = self.stochastic_depth(output)
        output += input
        return output


class ConvNeXtBlock(Module):
    
    def __init__(
        self,
        dim                  : int,
        layer_scale          : float,
        stochastic_depth_prob: float,
        num_layers           : int,
        stage_block_id       : int,
        total_stage_blocks   : int      | None = None,
        norm                 : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                ConvNeXtLayer(
                    dim                   = dim,
                    layer_scale           = layer_scale,
                    stochastic_depth_prob = stochastic_depth_prob,
                    stage_block_id        = stage_block_id,
                    total_stage_blocks    = total_stage_blocks,
                    norm= norm,
                )
            )
            stage_block_id += 1
        self.block = Sequential(*layers)
    
    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)


# H2: - Densenet ---------------------------------------------------------------

class DenseLayer(Module):
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        bn_size         : int,
        drop_rate       : float,
        memory_efficient: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.norm1 = BatchNorm2d(in_channels)
        self.relu1 = ReLU(inplace=True)
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels * bn_size,
            kernel_size  = 1,
            stride       = 1,
            bias         = False
        )
        self.norm2 = BatchNorm2d(out_channels * bn_size)
        self.relu2 = ReLU(inplace=True)
        self.conv2 = Conv2d(
            in_channels  = out_channels * bn_size,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            bias         = False
        )
        self.drop_rate        = float(drop_rate)
        self.memory_efficient = memory_efficient

    def forward(self, input: Tensors) -> Tensor:
        prev_features     = [input] if isinstance(input, Tensor) else input
        concat_features   = torch.cat(prev_features, dim=1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        new_features      = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0.0:
            new_features = F.dropout(
                input    = new_features,
                p        = self.drop_rate,
                training = self.training
            )
        return new_features


class DenseBlock(ModuleDict):

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        num_layers      : int,
        bn_size         : int,
        drop_rate       : float,
        memory_efficient: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels      = in_channels + i * out_channels,
                out_channels     = out_channels,
                bn_size          = bn_size,
                drop_rate        = drop_rate,
                memory_efficient = memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, input: Tensor) -> Tensor:
        features = [input]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseTransition(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        *args, **kwargs
    ):
        super().__init__()
        self.norm = BatchNorm2d(in_channels)
        self.relu = ReLU(inplace=True)
        self.conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            bias         = False,
        )
        self.pool = AvgPool2d(kernel_size=2, stride=2)

    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x
    

# H2: - FFANet -----------------------------------------------------------------

class FFA(Module):
    """
    This is the main feature in FFA-Net, the Feature Fusion Attention.
    
    We concatenate all feature maps output by G Group Architectures in the
    channel direction. Furthermore, We fuse features by multiplying the adaptive
    learning weights which are obtained by Feature Attention (FA) mechanism.
    
    Args:
        num_groups (int): Number of groups used in FFA-Net.
    """
    
    def __init__(
        self,
        channels  : int,
        num_groups: int,
        *args, **kwargs
    ):
        super().__init__()
        self.channels   = channels
        self.num_groups = num_groups
        self.ca         = nn.Sequential(*[
            AdaptiveAvgPool2d(1),
            Conv2d(
                in_channels  = self.channels * self.num_groups,
                out_channels = self.channels // 16,
                kernel_size  = 1,
                padding      = 0,
                bias         = True,
            ),
            ReLU(inplace=True),
            Conv2d(
                in_channels  = self.channels // 16,
                out_channels = self.channels * self.num_groups,
                kernel_size  = 1,
                padding      = 0,
                bias         = True
            ),
            Sigmoid()
        ])
        self.pa = PixelAttention(
            channels    = self.channels,
            reduction   = 8,
            kernel_size = 1,
        )
        
    def forward(self, input: Sequence[Tensor]) -> Tensor:
        assert_sequence_of_length(input, self.num_groups)
        w      = self.ca(torch.cat(to_list(input), dim=1))
        w      = w.view(-1, self.num_groups, self.channels)[:, :, :, None, None]
        output = w[:, 0, ::] * input[0]
        for i in range(1, len(input)):
            output += w[:, i, ::] * input[i]
        return output


class FFABlock(Module):
    """
    A basic block structure in FFA-Net that consists of:
        input --> Conv2d --> ReLU --> Conv2d --> Channel Attention --> Pixel Attention --> output
          |                       ^                                                          ^
          |_______________________|__________________________________________________________|
    """
    
    def __init__(
        self,
        channels   : int,
        kernel_size: Ints,
        *args, **kwargs
    ):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size // 2),
            bias         = True
        )
        self.act1  = ReLU(inplace=True)
        self.conv2 = Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size // 2),
            bias         = True
        )
        self.ca = ChannelAttention(
            channels    = channels,
            reduction   = 8,
            kernel_size = 1,
            stride      = 1,
            padding     = 0,
            bias        = True,
            max_pool    = False,
        )
        self.pa = PixelAttention(
            channels     = channels,
            reduction    = 8,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = True,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        x  = input
        y  = self.act1(self.conv1(x))
        y  = y + x
        y  = self.conv2(y)
        y  = self.ca(y)
        y  = self.pa(y)
        y += x
        return y


class FFAGroup(Module):
    """
    Our Group Architecture combines B Basic Block structures with skip
    connections module. Continuous B blocks increase the depth and
    expressiveness of the FFA-Net. And skip connections make FFA-Net get around
    training difficulty. At the end of the FFA-Net, we add a recovery part
    using a two-layer convolutional network implementation and a long shortcut
    global residual learning module. Finally, we restore our desired haze-free
    image.
    """
    
    def __init__(
        self,
        channels   : int,
        kernel_size: Ints,
        num_blocks : int,
        *args, **kwargs
    ):
        super().__init__()
        m: list[Module] = [
            FFABlock(channels=channels, kernel_size=kernel_size)
            for _ in range(num_blocks)
        ]
        m.append(
            Conv2d(
                in_channels  = channels,
                out_channels = channels,
                kernel_size  = kernel_size,
                padding      = (kernel_size // 2),
                bias         = True
            )
        )
        self.gp = nn.Sequential(*m)
    
    def forward(self, input: Tensor) -> Tensor:
        x  = input
        y  = self.gp(x)
        y += x
        return y


class FFAPostProcess(Module):
    """
    Post-process module in FFA-Net.
    """
    
    def __init__(
        self,
        in_channels : int  = 64,
        out_channels: int  = 3,
        kernel_size : Ints = 3,
        *args, **kwargs
    ):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size // 2),
            bias         = True
        )
        self.conv2 = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            padding      = (kernel_size // 2),
            bias         = True
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return self.conv2(self.conv1(input))


class FFAPreProcess(Module):
    """
    Pre-process module in FFA-Net.
    """

    def __init__(
        self,
        in_channels : int  = 3,
        out_channels: int  = 64,
        kernel_size : Ints = 3,
        *args, **kwargs
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = 1,
            padding      = (kernel_size // 2),
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)


# H2: - FINet ------------------------------------------------------------------

class FINetConvBlock(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        downsample  : bool,
        relu_slope  : float,
        use_csff    : bool  = False,
        use_hin     : bool  = False,
        alpha       : float = 0.5,
        selection   : str   = "linear",
        *args, **kwargs
    ):
        super().__init__()
        self.downsample = downsample
        self.use_csff   = use_csff
        self.use_hin    = use_hin
        self.alpha      = alpha
        
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            padding      = 1,
            bias         = True
        )
        self.relu1 = LeakyReLU(relu_slope, inplace=False)
        self.conv2 = Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            padding      = 1,
            bias         = True
        )
        self.relu2    = LeakyReLU(relu_slope, inplace=False)
        self.identity = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
        )
        
        if downsample and use_csff:
            self.csff_enc = Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            )
            self.csff_dec = Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            )
        
        if self.use_hin:
            self.norm = FractionInstanceNorm2d(
                alpha        = self.alpha,
                num_features = out_channels,
                selection    = selection,
            )

        if downsample:
            self.downsample = Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1,
                bias         = False
            )
    
    def forward(self, input: Tensors) -> tuple[Tensor | None, Tensor]:
        """
        
        Args:
            input (Tensors): A single tensor for the first UNet or a list of 3
                tensors for the second UNet.

        Returns:
            Output tensors.
        """
        enc = dec = None
        if isinstance(input, Tensor):
            x = input
        elif isinstance(input, Sequence):
            x = input[0]  # Input
            if len(input) == 2:
                enc = input[1]  # Encode path
            if len(input) == 3:
                dec = input[2]  # Decode path
        else:
            raise TypeError()
        
        y  = self.conv1(x)
        if self.use_fin:
            y = self.norm(y)
        y  = self.relu1(y)
        y  = self.relu2(self.conv2(y))
        y += self.identity(x)
        
        if enc is not None and dec is not None:
            if not self.use_csff:
                raise ValueError()
            y = y + self.csff_enc(enc) + self.csff_dec(dec)
       
        if self.downsample:
            y_down = self.downsample(y)
            return y_down, y
        else:
            return None, y
    
    
# H2: - HINet ------------------------------------------------------------------

class HINetConvBlock(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        downsample  : bool,
        relu_slope  : float,
        use_csff    : bool = False,
        use_hin     : bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.downsample = downsample
        self.use_csff   = use_csff
        
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            padding      = 1,
            bias         = True
        )
        self.relu1 = LeakyReLU(relu_slope, inplace=False)
        self.conv2 = Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            padding      = 1,
            bias         = True
        )
        self.relu2    = LeakyReLU(relu_slope, inplace=False)
        self.identity = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
        )
        
        if downsample and use_csff:
            self.csff_enc = Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            )
            self.csff_dec = Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 3,
                stride       = 1,
                padding      = 1,
            )
        
        self.use_hin = use_hin
        if self.use_hin:
            self.norm = InstanceNorm2d(out_channels // 2, affine=True)

        if downsample:
            self.downsample = Conv2d(
                in_channels  = out_channels,
                out_channels = out_channels,
                kernel_size  = 4,
                stride       = 2,
                padding      = 1,
                bias         = False
            )
    
    def forward(self, input: Tensors) -> tuple[Tensor | None, Tensor]:
        """
        
        Args:
            input (Tensors): A single tensor for the first UNet or a list of 3
                tensors for the second UNet.

        Returns:
            Output tensors.
        """
        enc = dec = None
        if isinstance(input, Tensor):
            x = input
        elif isinstance(input, Sequence):
            x = input[0]  # Input
            if len(input) == 2:
                enc = input[1]  # Encode path
            if len(input) == 3:
                dec = input[2]  # Decode path
        else:
            raise TypeError()
        
        y   = self.conv1(x)
        if self.use_hin:
            y1, y2 = torch.chunk(y, 2, dim=1)
            y      = torch.cat([self.norm(y1), y2], dim=1)
        y  = self.relu1(y)
        y  = self.relu2(self.conv2(y))
        y += self.identity(x)
        
        if enc is not None and dec is not None:
            if not self.use_csff:
                raise ValueError()
            y = y + self.csff_enc(enc) + self.csff_dec(dec)
       
        if self.downsample:
            y_down = self.downsample(y)
            return y_down, y
        else:
            return None, y
    

class HINetUpBlock(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        relu_slope  : float,
        *args, **kwargs
    ):
        super().__init__()
        self.up = ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 2,
            stride       = 2,
            bias         = True,
        )
        self.conv = HINetConvBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            downsample   = False,
            relu_slope   = relu_slope,
        )
    
    def forward(self, input: Sequence[Tensor]) -> Tensor:
        assert_sequence_of_length(input, 2)
        x    = input[0]
        skip = input[1]

        x_up = self.up(x)
        y    = torch.cat([x_up, skip], dim=1)
        y    = self.conv(y)
        return y[-1]


class HINetSkipBlock(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        mid_channels: int = 128,
        repeat_num  : int = 1,
        *args, **kwargs
    ):
        super().__init__()
        self.repeat_num = repeat_num
        self.shortcut   = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            bias         = True
        )
        
        blocks = []
        blocks.append(
            HINetConvBlock(
                in_channels  = in_channels,
                out_channels = mid_channels,
                downsample   = False,
                relu_slope   = 0.2
            )
        )
        for i in range(self.repeat_num - 2):
            blocks.append(
                HINetConvBlock(
                    in_channels  = mid_channels,
                    out_channels = mid_channels,
                    downsample   = False,
                    relu_slope   = 0.2
                )
            )
        blocks.append(
            HINetConvBlock(
                in_channels  = mid_channels,
                out_channels = out_channels,
                downsample   = False,
                relu_slope   = 0.2
            )
        )
        self.blocks = Sequential(*blocks)
    
    def forward(self, input: Tensor) -> Tensor:
        shortcut = self.shortcut(input)
        input    = self.blocks(input)
        return input + shortcut
    
    
# H2: - Inception --------------------------------------------------------------

class InceptionBasicConv2d(Module):
    """
    Conv2d + BN + ReLU.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints       = 1,
        padding     : Ints | str = 0,
        dilation    : Ints       = 1,
        groups      : int        = 1,
        bias        : bool       = False,
        padding_mode: str        = "zeros",
        device      : Any        = None,
        dtype       : Any        = None,
        apply_act   : bool       = True,
        eps         : float      = 0.001,
        *args, **kwargs
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        self.conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = to_same_padding(kernel_size, padding),
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.bn  = BatchNorm2d(out_channels, eps)
        self.act = ReLU()
    
    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(input)))
    
    
class Inception(Module):
    
    def __init__(
        self,
        in_channels: int,
        ch1x1      : int,
        ch3x3red   : int,
        ch3x3      : int,
        ch5x5red   : int,
        ch5x5      : int,
        pool_proj  : int,
        conv       : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch1 = conv(
            in_channels  = in_channels,
            out_channels = ch1x1,
            kernel_size  = 1,
        )
        self.branch2 = Sequential(
            conv(
                in_channels  = in_channels,
                out_channels = ch3x3red,
                kernel_size  = 1,
            ),
            conv(
                in_channels  = ch3x3red,
                out_channels = ch3x3,
                kernel_size  = 3,
                padding      = 1,
            )
        )
        self.branch3 = Sequential(
            conv(
                in_channels  = in_channels,
                out_channels = ch5x5red,
                kernel_size  = 1,
            ),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for
            # details.
            conv(
                in_channels  = ch5x5red,
                out_channels = ch5x5,
                kernel_size  = 3,
                padding      = 1,
            ),
        )
        self.branch4 = Sequential(
            MaxPool2d(
                kernel_size = 3,
                stride      = 1,
                padding     = 1,
                ceil_mode   = True,
            ),
            conv(
                in_channels  = in_channels,
                out_channels = pool_proj,
                kernel_size  = 1,
            ),
        )

    def forward(self, input: Tensor) -> Tensor:
        x       = input
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, dim=1)


class InceptionA(Module):
    
    base_out_channels: int = 224  # + pool_features
    
    def __init__(
        self,
        in_channels  : int,
        pool_features: int,
        conv         : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch1x1 = conv(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = 1,
            eps          = 0.001,
        )
        self.branch5x5_1 = conv(
            in_channels  = in_channels,
            out_channels = 48,
            kernel_size  = 1,
            eps          = 0.001,
        )
        self.branch5x5_2 = conv(
            in_channels  = 48,
            out_channels = 64,
            kernel_size  = 5,
            padding      = 2,
            eps          = 0.001,
        )
        self.branch3x3dbl_1 = conv(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = 1,
            eps          = 0.001,
        )
        self.branch3x3dbl_2 = conv(
            in_channels  = 64,
            out_channels = 96,
            kernel_size  = 3,
            padding      = 1,
            eps          = 0.001,
        )
        self.branch3x3dbl_3 = conv(
            in_channels  = 96,
            out_channels = 96,
            kernel_size  = 3,
            padding      = 1,
            eps          = 0.001,
        )
        self.branch_pool = conv(
            in_channels  = in_channels,
            out_channels = pool_features,
            kernel_size  = 1,
            eps          = 0.001,
        )

    def forward(self, input: Tensor) -> Tensor:
        branch1x1    = self.branch1x1(input)
        branch5x5    = self.branch5x5_1(input)
        branch5x5    = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(input)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool  = F.avg_pool2d(input, kernel_size=3, stride=1, padding=1)
        branch_pool  = self.branch_pool(branch_pool)
        outputs      = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(Module):
    
    base_out_channels: int = 480   # + in_channels
    
    def __init__(
        self,
        in_channels: int,
        conv       : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch3x3 = conv(
            in_channels  = in_channels,
            out_channels = 384,
            kernel_size  = 3,
            stride       = 2,
        )
        self.branch3x3dbl_1 = conv(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = 1,
        )
        self.branch3x3dbl_2 = conv(
            in_channels  = 64,
            out_channels = 96,
            kernel_size  = 3,
            padding      = 1,
        )
        self.branch3x3dbl_3 = conv(
            in_channels  = 96,
            out_channels = 96,
            kernel_size  = 3,
            stride       = 2,
        )

    def forward(self, input: Tensor) -> Tensor:
        branch3x3    = self.branch3x3(input)
        branch3x3dbl = self.branch3x3dbl_1(input)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool  = F.max_pool2d(input, kernel_size=3, stride=2)
        outputs      = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(Module):
    
    base_out_channels: int = 768
    
    def __init__(
        self,
        in_channels : int,
        channels_7x7: int,
        conv        : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        c7 = channels_7x7
        
        self.branch1x1 = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )
        self.branch7x7_1 = conv(
            in_channels  = in_channels,
            out_channels = c7,
            kernel_size  = 1,
        )
        self.branch7x7_2 = conv(
            in_channels  = c7,
            out_channels = c7,
            kernel_size  = (1, 7),
            padding      = (0, 3),
        )
        self.branch7x7_3 = conv(
            in_channels  = c7,
            out_channels = 192,
            kernel_size  = (7, 1),
            padding      = (3, 0),
        )
        self.branch7x7dbl_1 = conv(
            in_channels  = in_channels,
            out_channels = c7,
            kernel_size  = 1,
        )
        self.branch7x7dbl_2 = conv(
            in_channels  = c7,
            out_channels = c7,
            kernel_size  = (7, 1),
            padding      = (3, 0),
        )
        self.branch7x7dbl_3 = conv(
            in_channels  = c7,
            out_channels = c7,
            kernel_size  = (1, 7),
            padding      = (0, 3),
        )
        self.branch7x7dbl_4 = conv(
            in_channels  = c7,
            out_channels = c7,
            kernel_size  = (7, 1),
            padding      = (3, 0),
        )
        self.branch7x7dbl_5 = conv(
            in_channels  = c7,
            out_channels = 192,
            kernel_size  = (1 , 7),
            padding      = (0 , 3),
        )
        self.branch_pool = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )

    def forward(self, input: Tensor) -> Tensor:
        branch1x1    = self.branch1x1(input)
        branch7x7    = self.branch7x7_1(input)
        branch7x7    = self.branch7x7_2(branch7x7)
        branch7x7    = self.branch7x7_3(branch7x7)
        branch7x7dbl = self.branch7x7dbl_1(input)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)
        branch_pool  = F.avg_pool2d(input, kernel_size=3, stride=1, padding=1)
        branch_pool  = self.branch_pool(branch_pool)
        outputs      = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)
    
    
class InceptionD(Module):
    
    base_out_channels: int = 512   # + in_channels
    
    def __init__(
        self,
        in_channels: int,
        conv       : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch3x3_1 = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )
        self.branch3x3_2 = conv(
            in_channels  = 192,
            out_channels = 320,
            kernel_size  = 3,
            stride       = 2,
        )
        self.branch7x7x3_1 = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )
        self.branch7x7x3_2 = conv(
            in_channels  = 192,
            out_channels = 192,
            kernel_size  = (1, 7),
            padding      = (0, 3),
        )
        self.branch7x7x3_3 = conv(
            in_channels  = 192,
            out_channels = 192,
            kernel_size  = (7, 1),
            padding      = (3, 0),
        )
        self.branch7x7x3_4 = conv(
            in_channels  = 192,
            out_channels = 192,
            kernel_size  = 3,
            stride       = 2,
        )

    def forward(self, input: Tensor) -> Tensor:
        branch3x3   = self.branch3x3_1(input)
        branch3x3   = self.branch3x3_2(branch3x3)
        branch7x7x3 = self.branch7x7x3_1(input)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)
        branch_pool = F.max_pool2d(input, kernel_size=3, stride=2)
        outputs     = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)
    

class InceptionE(Module):
    
    base_out_channels: int = 2048
    
    def __init__(
        self,
        in_channels: int,
        conv       : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
        
        self.branch1x1 = conv(
            in_channels  = in_channels,
            out_channels = 320,
            kernel_size  = 1,
        )
        self.branch3x3_1 = conv(
            in_channels  = in_channels,
            out_channels = 384,
            kernel_size  = 1,
        )
        self.branch3x3_2a = conv(
            in_channels  = 384,
            out_channels = 384,
            kernel_size  = (1, 3),
            padding      = (0, 1),
        )
        self.branch3x3_2b = conv(
            in_channels  = 384,
            out_channels = 384,
            kernel_size  = (3, 1),
            padding      = (1, 0),
        )
        self.branch3x3dbl_1 = conv(
            in_channels  = in_channels,
            out_channels = 448,
            kernel_size  = 1,
        )
        self.branch3x3dbl_2 = conv(
            in_channels  = 448,
            out_channels = 384,
            kernel_size  = 3,
            padding      = 1,
        )
        self.branch3x3dbl_3a = conv(
            in_channels  = 384,
            out_channels = 384,
            kernel_size  = (1, 3),
            padding      = (0, 1),
        )
        self.branch3x3dbl_3b = conv(
            in_channels  = 384,
            out_channels = 384,
            kernel_size  = (3, 1),
            padding      = (1, 0),
        )

        self.branch_pool = conv(
            in_channels  = in_channels,
            out_channels = 192,
            kernel_size  = 1,
        )

    def forward(self, input: Tensor) -> Tensor:
        branch1x1    = self.branch1x1(input)
        branch3x3    = self.branch3x3_1(input)
        branch3x3    = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3    = torch.cat(branch3x3, 1)
        branch3x3dbl = self.branch3x3dbl_1(input)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)
        branch_pool  = F.avg_pool2d(input, kernel_size=3, stride=1, padding=1)
        branch_pool  = self.branch_pool(branch_pool)
        outputs      = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux1(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        conv        : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
       
        self.conv0 = conv(
            in_channels  = in_channels,
            out_channels = 128,
            kernel_size  = 1,
        )
        self.conv1 = conv(
            in_channels  = 128,
            out_channels = 768,
            kernel_size  = 5,
        )
        self.conv1.stddev = 0.01  # type: ignore[assignment]
        self.fc           = nn.Linear(768, out_channels)
        self.fc.stddev    = 0.001  # type: ignore[assignment]

    def forward(self, input: Tensor) -> Tensor:
        x = input
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class InceptionAux2(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        dropout     : float           = 0.7,
        conv        : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv is None:
            conv = InceptionBasicConv2d
       
        self.conv = conv(
            in_channels  = in_channels,
            out_channels = 128,
            kernel_size  = 1,
        )
        self.fc1     = Linear(in_features=2048, out_features=1024)
        self.fc2     = Linear(in_features=1024, out_features=out_channels)
        self.dropout = Dropout(p=dropout)

    def forward(self, input: Tensor) -> Tensor:
        x = input
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)
        return x
    

# H2: - MBLLEN -----------------------------------------------------------------

class EnhancementModule(Module):
    """
    Enhancement regression (EM) has a symmetric structure to first apply
    convolutions and then deconvolutions.
    
    Args:
        in_channels (int): Number of input channels. Defaults to 32.
        mid_channels (int): Number of input and output channels for middle
            Conv2d layers used in each EM block. Defaults to 8.
        out_channels (int): Number of output channels. Defaults to 3.
        kernel_size (Ints): Kernel size for Conv2d layers used in each EM block.
            Defaults to 5.
    """
    
    def __init__(
        self,
        in_channels : int  = 32,
        mid_channels: int  = 8,
        out_channels: int  = 3,
        kernel_size : Ints = 5,
        *args, **kwargs
    ):
        super().__init__()
        self.convs = Sequential(
            Conv2d(
                in_channels  = in_channels,
                out_channels = mid_channels,
                kernel_size  = 3,
                padding      = 1,
                padding_mode = "replicate"
            ),
            ReLU(),
            Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels,
                kernel_size  = kernel_size
            ),
            ReLU(),
            Conv2d(
                in_channels  = mid_channels,
                out_channels = mid_channels * 2,
                kernel_size  = kernel_size
            ),
            ReLU(),
            Conv2d(
                in_channels  = mid_channels * 2,
                out_channels = mid_channels * 4,
                kernel_size  = kernel_size
            ),
            ReLU()
        )
        self.deconvs = Sequential(
            ConvTranspose2d(
                in_channels  = mid_channels * 4,
                out_channels = mid_channels * 2,
                kernel_size  = kernel_size,
            ),
            ReLU(),
            ConvTranspose2d(
                in_channels  = mid_channels * 2,
                out_channels = mid_channels,
                kernel_size  = kernel_size
            ),
            ReLU(),
            Conv2d(
                in_channels  = mid_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size
            ),
            ReLU(),
        )
    
    def forward(self, input: Tensor) -> Tensor:
        return self.deconvs(self.convs(input))


EM = EnhancementModule


# H2: - ResNet -----------------------------------------------------------------

class ResNetBasicBlock(Module):
    
    expansion: int = 1

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int             = 1,
        groups      : int             = 1,
        dilation    : int             = 1,
        base_width  : int             = 64,
        downsample  : Module   | None = None,
        norm        : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if norm is None:
            norm = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                "`BasicBlock` only supports `groups=1` and `base_width=64`"
            )
        if dilation > 1:
            raise NotImplementedError(
                "dilation > 1 not supported in `BasicBlock`"
            )
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn1        = norm(out_channels)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn2        = norm(out_channels)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        output   = self.conv1(input)
        output   = self.bn1(output)
        output   = self.relu(output)
        output   = self.conv2(output)
        output   = self.bn2(output)
        if self.downsample is not None:
            identity = self.downsample(input)
        output += identity
        output  = self.relu(output)
        return output


class ResNetBottleneck(Module):
    """
    Bottleneck in torchvision places the stride for down-sampling at
    3x3 convolution(self.conv2) while original implementation places the stride
    at the first 1x1 convolution(self.conv1) according to
    "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy according
    to https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    """
    
    expansion: int = 4

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int             = 1,
        groups      : int             = 1,
        dilation    : int             = 1,
        base_width  : int             = 64,
        downsample  : Module   | None = None,
        norm        : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1      = Conv2d(
            in_channels  = in_channels,
            out_channels = width,
            kernel_size  = 1,
            stride       = stride,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn1        = norm(width)
        self.conv2      = Conv2d(
            in_channels  = width,
            out_channels = width,
            kernel_size  = 3,
            stride       = stride,
            padding      = dilation,
            groups       = groups,
            bias         = False,
            dilation     = dilation,
        )
        self.bn2        = norm(width)
        self.conv3      = Conv2d(
            in_channels  = width,
            out_channels = out_channels * self.expansion,
            kernel_size  = 1,
            stride       = stride,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False
        )
        self.bn3        = norm(out_channels * self.expansion)
        self.relu       = ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, input: Tensor) -> Tensor:
        identity = input
        output   = self.conv1(input)
        output   = self.bn1(output)
        output   = self.relu(output)
        output   = self.conv2(output)
        output   = self.bn2(output)
        output   = self.relu(output)
        output   = self.conv3(output)
        output   = self.bn3(output)
        if self.downsample is not None:
            identity = self.downsample(input)
        output += identity
        output  = self.relu(output)
        return output


class ResNetBlock(Module):
    
    def __init__(
        self,
        block        : Type[ResNetBasicBlock | ResNetBottleneck],
        num_blocks   : int,
        in_channels  : int,
        out_channels : int,
        stride       : int             = 1,
        groups       : int             = 1,
        dilation     : int             = 1,
        base_width   : int             = 64,
        dilate       : bool            = False,
        norm         : Callable | None = BatchNorm2d,
        *args, **kwargs
    ):
        super().__init__()
        downsample    = None
        prev_dilation = dilation
        if dilate:
            dilation *= stride
            stride    = 1
        
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels * block.expansion,
                    kernel_size  = 1,
                    stride       = stride,
                    bias         = False,
                ),
                norm(out_channels * block.expansion),
            )
      
        layers = []
        layers.append(
            block(
                in_channels  = in_channels,
                out_channels = out_channels,
                stride       = stride,
                groups       = groups,
                dilation     = prev_dilation,
                base_width   = base_width,
                downsample   = downsample,
                norm= norm,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels  = out_channels * block.expansion,
                    out_channels = out_channels,
                    stride       = 1,
                    groups       = groups,
                    dilation     = dilation,
                    base_width   = base_width,
                    downsample   = None,
                    norm= norm,
                )
            )
        self.convs = nn.Sequential(*layers)
    
    def forward(self, input: Tensor) -> Tensor:
        return self.convs(input)


# H2: - ShuffleNet -------------------------------------------------------------

class InvertedResidual(Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int,
        *args, **kwargs
    ):
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("Illegal stride value.")
        self.stride = stride

        branch_features = out_channels // 2
        if (self.stride == 1) and (in_channels != branch_features << 1):
            raise ValueError(
                f"Invalid combination of `stride` {stride}, "
                f"`in_channels` {in_channels} and `out_channels` {out_channels} "
                f"values. If stride == 1 then `in_channels` should be equal "
                f"to `out_channels` // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                Conv2d(
                    in_channels  = in_channels,
                    out_channels = in_channels,
                    kernel_size  = 3,
                    stride       = self.stride,
                    padding      = 1,
                    groups       = in_channels,
                    bias         = False,
                ),
                BatchNorm2d(in_channels),
                Conv2d(
                    in_channels  = in_channels,
                    out_channels = branch_features,
                    kernel_size  = 1,
                    stride       = 1,
                    padding      = 0,
                    bias         = False
                ),
                BatchNorm2d(branch_features),
                ReLU(inplace=True),
            )
        else:
            self.branch1 = Sequential()

        self.branch2 = Sequential(
            Conv2d(
                in_channels  = in_channels if (self.stride > 1) else branch_features,
                out_channels = branch_features,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = False,
            ),
            BatchNorm2d(branch_features),
            ReLU(inplace=True),
            Conv2d(
                in_channels  = branch_features,
                out_channels = branch_features,
                kernel_size  = 3,
                stride       = self.stride,
                padding      = 1,
                groups       = branch_features,
                bias         = False,
            ),
            BatchNorm2d(branch_features),
            Conv2d(
                in_channels  = branch_features,
                out_channels = branch_features,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = False,
            ),
            BatchNorm2d(branch_features),
            ReLU(inplace=True),
        )
    
    @staticmethod
    def channel_shuffle(x: Tensor, groups: int) -> Tensor:
        b, c, h, w         = x.size()
        channels_per_group = c // groups
        # reshape
        x = x.view(b, groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(b, -1, h, w)
        return x
    
    def forward(self, input: Tensor) -> Tensor:
        x = input
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            output = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            output = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        output = self.channel_shuffle(output, 2)
        return output


# H2: - SqueezeNet -------------------------------------------------------------

class Fire(Module):
    """
    """
    
    def __init__(
        self,
        in_channels     : int,
        squeeze_planes  : int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze = Conv2d(
            in_channels  = in_channels,
            out_channels = squeeze_planes,
            kernel_size  = 1,
        )
        self.squeeze_activation = ReLU(inplace=True)
        self.expand1x1 = Conv2d(
            in_channels  = squeeze_planes,
            out_channels = expand1x1_planes,
            kernel_size  = 1,
        )
        self.expand1x1_activation = ReLU(inplace=True)
        self.expand3x3 = Conv2d(
            in_channels  = squeeze_planes,
            out_channels = expand3x3_planes,
            kernel_size  = 3,
            padding      = 1,
        )
        self.expand3x3_activation = ReLU(inplace=True)
        
    def forward(self, input: Tensor) -> Tensor:
        x  = input
        x  = self.squeeze_activation(self.squeeze(x))
        x1 = self.expand1x1_activation(self.expand1x1(x))
        x3 = self.expand3x3_activation(self.expand3x3(x))
        return torch.cat([x1, x3], dim=1)


# H2: - SRCNN ------------------------------------------------------------------

class SRCNN(Module):
    """
    SRCNN (Super-Resolution Convolutional Neural Network).
    
    In SRCNN, actually the network is not deep. There are only 3 parts, patch
    extraction and representation, non-linear mapping, and reconstruction.
    
    References:
        https://medium.com/coinmonks/review-srcnn-super-resolution-3cb3a4f67a7c
        https://github.com/jspan/dualcnn/blob/master/Denoise/code/srcnn.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size1: Ints,
        stride1     : Ints,
        padding1    : Ints | str,
        kernel_size2: Ints,
        stride2     : Ints,
        padding2    : Ints | str,
        kernel_size3: Ints,
        stride3     : Ints,
        padding3    : Ints | str,
        dilation    : Ints = 1,
        groups      : int  = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = kernel_size1,
            stride       = stride1,
            padding      = padding1,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv2 = Conv2d(
            in_channels  = 64,
            out_channels = 32,
            kernel_size  = kernel_size2,
            stride       = stride2,
            padding      = padding2,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv3 = Conv2d(
            in_channels  = 32,
            out_channels = out_channels,
            kernel_size  = kernel_size3,
            stride       = stride3,
            padding      = padding3,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.relu = ReLU()
    
    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# H2: - UNet -------------------------------------------------------------------

class UNetBlock(Module):
    """
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints       = 3,
        stride      : Ints       = 1,
        padding     : Ints | str = 1,
        dilation    : Ints       = 1,
        groups      : int        = 1,
        bias        : bool       = False,
        padding_mode: str        = "zeros",
        device      : Any        = None,
        dtype       : Any        = None,
        *args, **kwargs
    ):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.norm1 = BatchNorm2d(num_features=out_channels)
        self.relu1 = ReLU(inplace=True)
        self.conv2 = Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.norm2 = BatchNorm2d(num_features=out_channels)
        self.relu2 = ReLU(inplace=True)
        
    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x


# H2: - VDSR -------------------------------------------------------------------

class VDSR(Module):
    """
    VDSR (Very Deep Super-Resolution).
    
    References:
        https://cv.snu.ac.kr/research/VDSR/
        https://github.com/twtygqyy/pytorch-vdsr
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints       = 3,
        stride      : Ints       = 1,
        padding     : Ints | str = 1,
        dilation    : Ints       = 1,
        groups      : int        = 1,
        bias        : bool       = False,
        padding_mode: str        = "zeros",
        device      : Any        = None,
        dtype       : Any        = None,
        *args, **kwargs
    ):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = 64,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.residual_layer = Sequential(*[
            ConvReLU2d(
                in_channels  = 64,
                out_channels = 64,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            )
            for _ in range(18)
        ])
        self.conv2 = Conv2d(
            in_channels  = 64,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.relu = ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

        
    def forward(self, input: Tensor) -> Tensor:
        x        = input
        residual = input
        x        = self.relu(self.conv1(x))
        x        = self.residual_layer(x)
        x        = self.conv2(x)
        x        = torch.add(x, residual)
        return x


# H2: - ZeroDCE ----------------------------------------------------------------

class AttentionSubspaceBlueprintSeparableConv2d(Module):
    """
    Subspace Blueprint Separable Conv2d with Self-Attention adopted from the
    paper:
        "Rethinking Depthwise Separable Convolutions: How Intra-Kernel
        Correlations Lead to Improved MobileNets," CVPR 2020.
    
    References:
        https://github.com/zeiss-microscopy/BSConv
    """

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : Ints,
        stride          : Ints            = 1,
        padding         : Ints | str      = 0,
        dilation        : Ints            = 1,
        groups          : int             = 1,
        bias            : bool            = True,
        padding_mode    : str             = "zeros",
        device          : Any             = None,
        dtype           : Any             = None,
        p               : float           = 0.25,
        min_mid_channels: int             = 4,
        act             : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        assert_number_in_range(p, 0.0, 1.0)
        mid_channels  = min(in_channels, max(min_mid_channels, math.ceil(p * in_channels)))
        self.pw_conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act1     = act(num_features=mid_channels) if act is not None else None
        self.pw_conv2 = Conv2d(
            in_channels  = mid_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act2    = act(num_features=out_channels) if act is not None else None
        self.dw_conv = Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.simam = SimAM()
        """
        self.ca = ChannelAttention(
            channels    = out_channels,
            reduction   = 16,
            kernel_size = 1,
            groups      = 1
        )
        self.sa = SpatialAttention(kernel_size=7)
        """
        
    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.pw_conv1(x)
        x = self.simam(x)
        if self.act1 is not None:
            x = self.act1(x)
        x = self.pw_conv2(x)
        if self.act2 is not None:
            x = self.act2(x)
        x = self.dw_conv(x)
        # x = self.ca(x) * x
        # x = self.sa(x) * x
        # x = x + input
        return x
    
    def regularization_loss(self):
        w   = self.pw_conv1.weight[:, :, 0, 0]
        wwt = torch.mm(w, torch.transpose(w, 0, 1))
        i   = torch.eye(wwt.shape[0], device=wwt.device)
        return torch.norm(wwt - i, p="fro")


class AttentionUnconstrainedBlueprintSeparableConv2d(Module):
    """
    Subspace Blueprint Separable Conv2d with Self-Attention adopted from the
    paper:
        "Rethinking Depthwise Separable Convolutions: How Intra-Kernel
        Correlations Lead to Improved MobileNets," CVPR 2020.
    
    References:
        https://github.com/zeiss-microscopy/BSConv
    """

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : Ints,
        stride          : Ints            = 1,
        padding         : Ints | str      = 0,
        dilation        : Ints            = 1,
        groups          : int             = 1,
        bias            : bool            = True,
        padding_mode    : str             = "zeros",
        device          : Any             = None,
        dtype           : Any             = None,
        p               : float           = 0.25,
        min_mid_channels: int             = 4,
        act             : Callable | None = None,
        *args, **kwargs
    ):
        super().__init__()
        self.pw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dilation     = 1,
            groups       = 1,
            bias         = False,
            padding_mode = "zeros",
            device       = device,
            dtype        = dtype,
        )
        self.act = act(num_features=out_channels) if act is not None else None
        self.dw_conv = Conv2d(
            in_channels  = out_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = out_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.sa = SpatialAttention(kernel_size=7)
        
    def forward(self, input: Tensor) -> Tensor:
        x = input
        x = self.pw_conv(x)
        if self.act is not None:
            x = self.act(x)
        x = self.dw_conv(x)
        # x = self.ca(x) * x
        x = self.sa(x) * x
        # x = x + input
        return x


ABSConv2dS = AttentionSubspaceBlueprintSeparableConv2d
ABSConv2dU = AttentionUnconstrainedBlueprintSeparableConv2d


class DCE(Module):
    """
    """
    
    def __init__(
        self,
        in_channels : int  = 3,
        out_channels: int  = 24,
        mid_channels: int  = 32,
        kernel_size : Ints = 3,
        stride      : Ints = 1,
        padding     : Ints = 1,
        dilation    : Ints = 1,      
        groups      : int  = 1,      
        bias        : bool = True,   
        padding_mode: str  = "zeros",
        device      : Any  = None,   
        dtype       : Any  = None,   
        *args, **kwargs
    ):
        super().__init__()
        self.relu  = ReLU(inplace=True)
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,    
            groups       = groups,     
            bias         = bias,        
            padding_mode = padding_mode,
            device       = device,     
            dtype        = dtype,      
        ) 
        self.conv2 = Conv2d(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,    
            groups       = groups,     
            bias         = bias,        
            padding_mode = padding_mode,
            device       = device,     
            dtype        = dtype, 
        ) 
        self.conv3 = Conv2d(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,    
            groups       = groups,     
            bias         = bias,        
            padding_mode = padding_mode,
            device       = device,     
            dtype        = dtype, 
        ) 
        self.conv4 = Conv2d(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,    
            groups       = groups,     
            bias         = bias,        
            padding_mode = padding_mode,
            device       = device,     
            dtype        = dtype, 
        ) 
        self.conv5 = Conv2d(
            in_channels  = mid_channels * 2,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,    
            groups       = groups,     
            bias         = bias,        
            padding_mode = padding_mode,
            device       = device,     
            dtype        = dtype, 
        ) 
        self.conv6 = Conv2d(
            in_channels  = mid_channels * 2,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,    
            groups       = groups,     
            bias         = bias,        
            padding_mode = padding_mode,
            device       = device,     
            dtype        = dtype, 
        ) 
        self.conv7 = Conv2d(
            in_channels  = mid_channels * 2,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,    
            groups       = groups,     
            bias         = bias,        
            padding_mode = padding_mode,
            device       = device,     
            dtype        = dtype,       
        ) 
    
    def forward(self, input: Tensor) -> Tensor:
        x  = input
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        a  = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        return a


class DCEV2(Module):
    """
    """
    
    def __init__(
        self,
        in_channels : int      = 3,
        out_channels: int      = 3,
        mid_channels: int      = 32,
        conv        : Callable = BSConv2dS,
        kernel_size : Ints     = 3,
        stride      : Ints     = 1,
        padding     : Ints     = 1,
        dilation    : Ints     = 1,
        groups      : int      = 1,
        bias        : bool     = True,
        padding_mode: str      = "zeros",
        device      : Any      = None,
        dtype       : Any      = None,
        *args, **kwargs
    ):
        super().__init__()
        self.downsample = Downsample(None, 1, "bilinear")
        self.upsample   = UpsamplingBilinear2d(None, 1)
        self.relu       = ReLU(inplace=True)
        self.conv1 = conv(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv2 = conv(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv3 = conv(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv4 = conv(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv5 = conv(
            in_channels  = mid_channels * 2,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv6 = conv(
            in_channels  = mid_channels * 2,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv7 = Conv2d(
            in_channels  = mid_channels * 2,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, input: Tensor) -> Tensor:
        x  = input
        x  = self.downsample(x)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))
        a  = torch.tanh(self.conv7(torch.cat([x1, x6], dim=1)))
        a  = self.upsample(a)
        return a
       

class PixelwiseHigherOrderLECurve(Module):
    """
    Pixelwise Light-Enhancement Curve is a higher-order curves that can be
    applied iteratively to enable more versatile adjustment to cope with
    challenging low-light conditions:
        LE_{n}(x) = LE_{n−1}(x) + A_{n}(x) * LE_{n−1}(x)(1 − LE_{n−1}(x)),
        
        where `A` is a parameter map with the same size as the given image, and
        `n` is the number of iteration, which controls the curvature.
    
    This module is designed to accompany both:
        - ZeroDCE   (estimate 3*n curve parameter maps)
        - ZeroDCE++ (estimate 3   curve parameter maps)
    
    Args:
        n (int): Number of iterations.
    """
    
    def __init__(
        self,
        n: int,
        *args, **kwargs
    ):
        super().__init__()
        self.n = n
    
    def forward(self, input: list[Tensor]) -> tuple[Tensor, Tensor]:
        # Split
        a = input[0]  # Trainable curve parameter learned from previous layer
        x = input[1]  # Original input image
        
        # Prepare curve parameter
        _, c1, _, _ = x.shape  # Should be 3
        _, c2, _, _ = a.shape  # Should be 3*n
        single_map  = True
        if c2 == c1 * self.n:
            single_map = False
            a = torch.split(a, c1, dim=1)
        elif c2 == 3:
            pass
        else:
            raise ValueError(
                f"Curve parameter maps `a` must be `3` or `3 * {self.n}`. "
                f"But got: {c2}."
            )

        # Estimate curve parameter
        for i in range(self.n):
            a_i = a if single_map else a[i]
            x   = x + a_i * (torch.pow(x, 2) - x)

        a = list(a)             if isinstance(a, tuple) else a
        a = torch.cat(a, dim=1) if isinstance(a, list)  else a
        return a, x
