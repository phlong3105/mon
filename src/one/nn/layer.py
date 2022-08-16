#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom layers and blocks.

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
input
output
"""

from __future__ import annotations

import math

import torch.nn.functional as F
from torch import Tensor
from torch.nn import *

from one.constants import *
from one.core import *


# H1: - Activation -------------------------------------------------------------

@LAYERS.register(name="argmax")
class ArgMax(Module):
    """
    Find the indices of the maximum value of all elements in the input
    image.

    Args:
        dim (int | None): Dimension to find the indices of the maximum value.
            Defaults to None.
    """
    
    def __init__(self, dim: int | None = None):
        super().__init__()
        self.dim = dim
        
    def forward(self, input: Tensor) -> Tensor:
        return torch.argmax(input, dim=self.dim)


@LAYERS.register(name="clamp")
@LAYERS.register(name="clip")
class Clamp(Module):
    """
    Clamp the feature value within [min, max]. More details can be found in
    `torch.clamp()`.

    Args:
        min (float): Lower-bound of the range to be clamped to. Defaults to -1.0
        max (float): Upper-bound of the range to be clamped to. Defaults to -1.0
    """
    
    def __init__(self, min: float = -1.0, max: float = 1.0):
        super().__init__()
        self.min = min
        self.max = max
        
    def forward(self, input: Tensor) -> Tensor:
        return torch.clamp(input, min=self.min, max=self.max)


@LAYERS.register(name="frelu")
class FReLU(Module):
    
    def __init__(self, c1: int, k: Ints = 3):
        super().__init__()
        k         = to_2tuple(k)
        self.conv = Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.bn   = BatchNorm2d(c1)
        
    def forward(self, input: Tensor) -> Tensor:
        return torch.max(input, self.bn(self.conv(input)))


Clip = Clamp

LAYERS.register(name="celu",                module=CELU)
LAYERS.register(name="elu",                 module=ELU)
LAYERS.register(name="gelu",                module=GELU)
LAYERS.register(name="glu",                 module=GLU)
LAYERS.register(name="hard_shrink",         module=Hardshrink)
LAYERS.register(name="hard_sigmoid",        module=Hardsigmoid)
LAYERS.register(name="hard_swish", 	        module=Hardswish)
LAYERS.register(name="hard_tanh",           module=Hardtanh)
LAYERS.register(name="leaky_relu",          module=LeakyReLU)
LAYERS.register(name="log_sigmoid",         module=LogSigmoid)
LAYERS.register(name="log_softmax",         module=LogSoftmax)
LAYERS.register(name="mish",                module=Mish)
LAYERS.register(name="multihead_attention", module=MultiheadAttention)
LAYERS.register(name="prelu",               module=PReLU)
LAYERS.register(name="relu", 		        module=ReLU)
LAYERS.register(name="relu6", 		        module=ReLU6)
LAYERS.register(name="rrelu", 		        module=RReLU)
LAYERS.register(name="selu", 		        module=SELU)
LAYERS.register(name="sigmoid",		        module=Sigmoid)
LAYERS.register(name="silu", 		        module=SiLU)
LAYERS.register(name="softmax",             module=Softmax)
LAYERS.register(name="softmax_2d",          module=Softmax2d)
LAYERS.register(name="softmin",             module=Softmin)
LAYERS.register(name="softplus", 	        module=Softplus)
LAYERS.register(name="softshrink",          module=Softshrink)
LAYERS.register(name="softsign",            module=Softsign)
LAYERS.register(name="tanh",		        module=Tanh)
LAYERS.register(name="tanhshrink",          module=Tanhshrink)
LAYERS.register(name="threshold",           module=Threshold)


def to_act_layer(
    act    : Callable | None = ReLU(),
    inplace: bool            = True,
    **_
) -> Module:
    """
    Create activation layer.
    """
    if isinstance(act, str):
        act = LAYERS.build(name=act)
    if isinstance(act, types.FunctionType):
        act_args  = dict(inplace=True) if inplace else {}
        act = act(**act_args)
    if act is None:
        act = Identity()
    return act


# H1: - Attention --------------------------------------------------------------

@LAYERS.register(name="channel_attention_layer")
@LAYERS.register(name="cal")
class ChannelAttentionLayer(Module):
    """
    Channel Attention Layer.
    
    Args:
        channels (int): Number of input and output channels.
        reduction (int): Reduction factor. Defaults to 16.
        bias (bool): Defaults to False.
    """
    
    def __init__(
        self,
        channels    : int,
        reduction   : int  = 16,
        stride      : Ints = 1,
        dilation    : Ints = 1,
        groups      : int  = 1,
        bias        : bool = False,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None,
        **_
    ):
        super().__init__()
        # Global average pooling: feature --> point
        self.avg_pool = AdaptiveAvgPool2d(1)
        # Feature channel downscale and upscale --> channel weight
        self.ca = Sequential(
            Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction,
                kernel_size  = 1,
                stride       = stride,
                padding      = 0,
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
                kernel_size  = 1,
                stride       = stride,
                padding      = 0,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            Sigmoid()
        )
        
    def forward(self, input: Tensor) -> Tensor:
        output = self.avg_pool(input)
        output = self.ca(output)
        return input * output
 
    
@LAYERS.register(name="channel_attention_block")
@LAYERS.register(name="cab")
class ChannelAttentionBlock(Module):
    """
    Channel Attention Block.
    """
    
    def __init__(
        self,
        channels    : int,
        reduction   : int,
        kernel_size : Ints,
        stride      : Ints            = 1,
        dilation    : Ints            = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        act         : Callable | None = ReLU(),
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        padding     = kernel_size[0] // 2
        self.cal    = ChannelAttentionLayer(channels, reduction, bias)
        self.body   = Sequential(
            Conv2d(
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
            ),
            to_act_layer(act=act),
            Conv2d(
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
            ),
        )
        
    def forward(self, input: Tensor) -> Tensor:
        output = self.body(input)
        output = self.cal(output)
        output += input
        return output


@LAYERS.register(name="pixel_attention_layer")
@LAYERS.register(name="pal")
class PixelAttentionLayer(Module):
    """
    Pixel Attention Layer.
    
    Args:
        reduction (int): Reduction factor. Defaults to 16.
    """
    
    def __init__(
        self,
        channels    : int,
        reduction   : int  = 16,
        stride      : Ints = 1,
        dilation    : Ints = 1,
        groups      : int  = 1,
        bias        : bool = False,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None,
        **_
    ):
        super().__init__()
        stride   = to_2tuple(stride)
        dilation = to_2tuple(dilation)
        self.pa  = Sequential(
            Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction,
                kernel_size  = 1,
                stride       = stride,
                padding      = 0,
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
                kernel_size  = 1,
                stride       = stride,
                padding      = 0,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype,
            ),
            Sigmoid()
        )
        
    def forward(self, input: Tensor) -> Tensor:
        output = self.pa(input)
        return input * output


@LAYERS.register(name="supervised_attention_module")
@LAYERS.register(name="sam")
class SupervisedAttentionModule(Module):
    """
    Supervised Attention Module.
    """
    
    def __init__(
        self,
        channels    : int,
        kernel_size : Ints,
        dilation    : Ints = 1,
        groups      : int  = 1,
        bias        : bool = False,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = 1
        padding     = kernel_size[0] // 2
        dilation    = to_2tuple(dilation)
        self.conv1  = Conv2d(
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
        
    def forward(
        self, prev_output: Tensor, input: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Run forward pass.

        Args:
            prev_output (Tensor): Output from previous steps.
            input (Tensor): Current step input.
        """
        pred     = self.conv1(prev_output)
        output   = self.conv2(prev_output) + input
        sigmoid  = torch.sigmoid(self.conv3(output))
        pred    *= sigmoid
        pred    += prev_output
        return pred, output


CAB = ChannelAttentionBlock
CAL = ChannelAttentionLayer
PAL = PixelAttentionLayer
SAM = SupervisedAttentionModule


# H1: - Bottleneck -------------------------------------------------------------

@LAYERS.register(name="bottleneck")
class Bottleneck(Module):
    """
    Standard bottleneck.
    
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        shortcut (bool): Use shortcut connection?. Defaults to True.
        groups (int): Defaults to 1.
        expansion (float): Defaults to 0.5.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        shortcut    : bool  = True,
        groups      : int   = 1,
        expansion   : float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # Hidden channels
        self.conv1 = ConvBnMish2d(
            in_channels  = in_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.conv2 = ConvBnMish2d(
            in_channels  = hidden_channels,
            out_channels = out_channels,
            kernel_size  = 3,
            stride       = 1,
            groups       = groups
        )
        self.add = shortcut and in_channels == out_channels
        
    def forward(self, input: Tensor) -> Tensor:
        output = ((input + self.conv2(self.conv1(input))) if self.add
                  else self.conv2(self.conv1(input)))
        return output


@LAYERS.register(name="bottleneck_csp")
class BottleneckCSP(Module):
    """
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        number (int): Number of bottleneck layers to use. Defaults to 1.
        shortcut (bool): Use shortcut connection?. Defaults to True.
        groups (int): Defaults to 1.
        expansion (float): Defaults to 0.5.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        shortcut    : bool  = True,
        groups      : int   = 1,
        expansion   : float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # Hidden channels
        self.conv1 = ConvBnMish2d(
            in_channels  = in_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.conv2 = Conv2d(
            in_channels  = in_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1,
            bias         = False
        )
        self.conv3 = Conv2d(
            in_channels  = hidden_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1,
            bias         = False
        )
        self.conv4 = ConvBnMish2d(
            in_channels  = 2 * hidden_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1
        )
        # Applied to cat(cv2, cv3)
        self.bn    = BatchNorm2d(2 * hidden_channels)
        self.act   = Mish()
        self.m     = Sequential(*[
            Bottleneck(
                in_channels  = hidden_channels,
                out_channels = hidden_channels,
                shortcut     = shortcut,
                groups       = groups,
                expansion    = 1.0
            )
            for _ in range(number)
        ])
        
    def forward(self, input: Tensor) -> Tensor:
        y1     = self.conv3(self.m(self.conv1(input)))
        y2     = self.conv2(input)
        output = self.conv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        return output


@LAYERS.register(name="bottleneck_csp2")
class BottleneckCSP2(Module):
    """
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        number (int): Number of bottleneck layers to use. Defaults to 1.
        groups (int): Defaults to 1.
        expansion (float):  Defaults to 0.5.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        shortcut    : bool  = False,
        groups      : int   = 1,
        expansion   : float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels)  # Hidden channels
        self.conv1 = ConvBnMish2d(
            in_channels  = in_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.conv2 = Conv2d(
            in_channels  = hidden_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1,
            bias         = False
        )
        self.conv3 = ConvBnMish2d(
            in_channels  = 2 * hidden_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.bn    = BatchNorm2d(2 * hidden_channels)
        self.act   = Mish()
        self.m     = Sequential(*[
            Bottleneck(
                in_channels  = hidden_channels,
                out_channels = hidden_channels,
                shortcut     = shortcut,
                groups       = groups,
                expansion    = 1.0
            )
            for _ in range(number)
        ])
        
    def forward(self, input: Tensor) -> Tensor:
        x1 = self.conv1(input)
        y1 = self.m(x1)
        y2 = self.conv2(x1)
        return self.conv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@LAYERS.register(name="vov_csp")
class VoVCSP(Module):
    """
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        number (int): Number of bottleneck layers to use.
        groups (int): Defaults to 1.
        expansion (float): Defaults to 0.5.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        shortcut    : bool  = True,
        groups      : int   = 1,
        expansion   : float = 0.5,
    ):
        super().__init__()
        hidden_channels = int(out_channels)  # Hidden channels
        self.conv1 = ConvBnMish2d(
            in_channels  = in_channels // 2,
            out_channels = hidden_channels // 2,
            kernel_size  = 3,
            stride       = 1
        )
        self.conv2 = ConvBnMish2d(
            in_channels  = hidden_channels // 2,
            out_channels = hidden_channels // 2,
            kernel_size  = 3,
            stride       = 1
        )
        self.conv3       = ConvBnMish2d(
            in_channels  = hidden_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1
        )
        
    def forward(self, input: Tensor) -> Tensor:
        _, x1  = input.chunk(2, dim=1)
        x1     = self.conv1(x1)
        x2     = self.conv2(x1)
        output = self.conv3(torch.cat((x1, x2), dim=1))
        return output


# H1: - Convolution ------------------------------------------------------------

def conv2d_same(
    input   : Tensor,
    weight  : Tensor,
    bias    : Tensor | None     = None,
    stride  : Ints              = 1,
    padding : str | Ints | None = 0,
    dilation: Ints              = 1,
    groups  : int               = 1,
    **_
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


@LAYERS.register(name="cond_act2d")
class ConvAct2d(Module):
    """
    Conv2d + Act.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        act         : Callable | None   = ReLU(),
        inplace     : bool              = True,
        **_
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
        

@LAYERS.register(name="cond_bn_mish2d")
class ConvBnMish2d(Module):
    """
    Conv2d + BN + Mish.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints              = 1,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        **_
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
        self.bn   = BatchNorm2d(out_channels)
        self.act  = Mish()
    
    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(input)))

    def fuse_forward(self, input: Tensor) -> Tensor:
        return self.act(self.conv(input))
    

@LAYERS.register(name="cond_bn_relu2d")
class ConvBnReLU2d(Module):
    """
    Conv2d + BN + ReLU.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        apply_act   : bool              = True,
        **_
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
        self.bn  = BatchNorm2d(out_channels)
        self.act = ReLU()
    
    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(input)))
    
    
@LAYERS.register(name="cond_bn_relu62d")
class ConvBnReLU62d(Module):
    """
    Conv2d + BN + ReLU6.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        apply_act   : bool              = True,
        **_
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
        self.bn  = BatchNorm2d(out_channels)
        self.act = ReLU6()
    
    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(input)))
    

@LAYERS.register(name="cond_mish2d")
class ConvMish2d(Module):
    """
    Conv2d + Mish.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
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
        self.act = Mish()
    
    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.conv(input))
    

@LAYERS.register(name="cond_relu2d")
class ConvReLU2d(Module):
    """
    Conv2d + ReLU.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        **_
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

 
@LAYERS.register(name="conv_same2d")
class ConvSame2d(Conv2d):
    """
    Tensorflow like `SAME` convolution wrapper for 2D convolutions.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        **_
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
    

@LAYERS.register(name="conv_sigmoid2d")
class ConvSigmoid2d(Module):
    """
    Conv2d + Sigmoid.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        **_
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
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
        self.act = Sigmoid()
    
    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.conv(input))


@LAYERS.register(name="conv_tf2d")
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
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        **_
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


@LAYERS.register(name="conv_transpose_act2d")
class ConvTransposeAct2d(Module):
    """
    ConvTranspose2d + Act.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        act         : Callable | None   = ReLU(),
        inplace     : bool              = True,
        **_
    ):
        super().__init__()
        act   = to_act_layer(act=act, inplace=inplace)
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        self.conv = ConvTranspose2d(
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
        self.act = to_act_layer(act=act, inplace=inplace)
    
    def forward(self, input: Tensor) -> Tensor:
        return self.act(self.conv(input))


@LAYERS.register(name="cross_conv2d")
class CrossConv2d(Module):
    """
    Cross Convolution Downsample.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int               = 3,
        stride      : int               = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        expansion   : float             = 1.0,
        shortcut    : bool              = False,
        **_
    ):
        super().__init__()
        c = int(out_channels * expansion)  # Hidden channels
        self.cv1 = ConvBnMish2d(
            in_channels  = in_channels,
            out_channels = c,
            kernel_size  = (1, kernel_size),
            stride       = (1, stride),
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.cv2 = ConvBnMish2d(
            in_channels  = c,
            out_channels = out_channels,
            kernel_size  = (kernel_size, 1),
            stride       = (stride     , 1),
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.add = shortcut and in_channels == out_channels
        
    def forward(self, input: Tensor) -> Tensor:
        return input + self.cv2(self.cv1(input)) if self.add \
            else self.cv2(self.cv1(input))


@LAYERS.register(name="cross_conv_csp")
class CrossConvCSP(Module):
    """
    Cross Convolution CSP.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        groups      : int   = 1,
        expansion   : float = 0.5,
        shortcut    : bool  = True,
        **_
    ):
        super().__init__()
        c        = int(out_channels * expansion)  # Hidden channels
        self.cv1 = ConvBnMish2d(
            in_channels  = in_channels,
            out_channels = c,
            kernel_size  = 1,
            stride       = 1
        )
        self.cv2 = Conv2d(
            in_channels  = in_channels,
            out_channels = c,
            kernel_size  = 1,
            stride       = 1,
            bias         = False
        )
        self.cv3 = Conv2d(
            in_channels  = c,
            out_channels = c,
            kernel_size  = 1,
            stride       = 1,
            bias         = False
        )
        self.cv4 = ConvBnMish2d(
            in_channels  = 2 * c,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.bn  = BatchNorm2d(2 * c)  # Applied to cat(cv2, cv3)
        self.act = LeakyReLU(0.1, inplace=True)
        self.m   = Sequential(*[
            CrossConv2d(
                in_channels  = c,
                out_channels = c,
                kernel_size  = 3,
                stride       = 1,
                groups       = groups,
                expansion    = 1.0,
                shortcut     = shortcut
            )
            for _ in range(number)
        ])
        
    def forward(self, input: Tensor) -> Tensor:
        y1 = self.cv3(self.m(self.cv1(input)))
        y2 = self.cv2(input)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


@LAYERS.register(name="depthwise_conv2d")
class DepthwiseConv2d(Conv2d):
    """
    Depthwise Conv2d with 3x3 kernel size, 1 stride, and groups == out_channels.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        padding     : str | Ints | None = 0,
        groups      : int | None        = 0,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        **_
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = (3, 3),
            stride       = (1, 1),
            padding      = padding,
            groups       = groups if groups is not None else math.gcd(in_channels, out_channels),
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )


@LAYERS.register(name="depthwise_conv_bn_mish2d")
class DepthwiseConvBnMish2d(ConvBnMish2d):
    """
    Depthwise Conv2d + Bn + Mish.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints              = 1,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        apply_act   : bool              = True,
        **_
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = math.gcd(in_channels, out_channels),
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
            apply_act    = apply_act,
        )


@LAYERS.register(name="pointwise_conv2d")
class PointwiseConv2d(Conv2d):
    """
    Pointwise Conv2d with 1x1 kernel size, 1 stride, and groups == 1.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        padding     : str | Ints | None = 0,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        **_
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = padding,
            groups       = 1,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        

@LAYERS.register(name="scaled_std_conv2d")
class ScaledStdConv2d(Conv2d):
    """
    Conv2d layer with Scaled Weight Standardization.

    Paper: `Characterizing signal propagation to close the performance gap in
    un-normalized ResNets` - https://arxiv.org/abs/2101.08692

    The operations used in this impl differ slightly from the DeepMind Haiku
    implementation impact is minor.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        gamma       : float             = 1.0,
        eps         : float             = 1e-6,
        gain_init   : float             = 1.0
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        if padding is None:
            padding = get_symmetric_padding(
                kernel_size = kernel_size[0],
                stride      = stride[0],
                dilation    = dilation[0]
            )
            
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
        self.gain  = Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        # gamma * 1 / sqrt(fan-in)
        self.scale = gamma * self.weight[0].numel() ** -0.5
        self.eps   = eps

    def forward(self, input: Tensor) -> Tensor:
        weight = F.batch_norm(
            input        = self.weight.reshape(1, self.out_channels, -1),
            running_mean = None,
            running_var  = None,
            weight       = (self.gain * self.scale).view(-1),
            training     = True,
            momentum     = 0.0,
            eps          = self.eps
        ).reshape_as(self.weight)
        return F.conv2d(
            input    = input,
            weight   = weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )


@LAYERS.register(name="std_conv2d")
class StdConv2d(Conv2d):
    """
    Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight
    Standardization` - https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Ints,
        stride      : Ints              = 1,
        padding     : str | Ints | None = 0,
        dilation    : Ints              = 1,
        groups      : int               = 1,
        bias        : bool              = True,
        padding_mode: str               = "zeros",
        device      : Any               = None,
        dtype       : Any               = None,
        eps         : float             = 1e-6
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        if padding is None:
            padding = get_symmetric_padding(
                kernel_size = kernel_size[0],
                stride      = stride[0],
                dilation    = dilation[0]
            )
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
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        weight = F.batch_norm(
            input        = self.weight.reshape(1, self.out_channels, -1),
            running_mean = None,
            running_var  = None,
            training     = True,
            momentum     = 0.0,
            eps          = self.eps
        ).reshape_as(self.weight)
        return F.conv2d(
            input    = x,
            weight   = weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )


LAYERS.register(name="conv1d",                module=Conv1d)
LAYERS.register(name="conv2d",                module=Conv2d)
LAYERS.register(name="conv3d",                module=Conv3d)
LAYERS.register(name="conv_transpose1d",      module=ConvTranspose1d)
LAYERS.register(name="conv_transpose2d",      module=ConvTranspose2d)
LAYERS.register(name="conv_transpose3d",      module=ConvTranspose3d)
LAYERS.register(name="lazy_conv1d",           module=LazyConv1d)
LAYERS.register(name="lazy_conv2d",           module=LazyConv2d)
LAYERS.register(name="lazy_conv3d",           module=LazyConv3d)
LAYERS.register(name="lazy_conv_transpose1d", module=LazyConvTranspose1d)
LAYERS.register(name="lazy_conv_transpose2d", module=LazyConvTranspose2d)
LAYERS.register(name="lazy_conv_transpose3d", module=LazyConvTranspose3d)


# H1: - Drop -------------------------------------------------------------------

def drop_block_2d(
    input      : Tensor,
    drop_prob  : float = 0.1,
    block_size : int   = 7,
    gamma_scale: float = 1.0,
    with_noise : bool  = False,
    inplace    : bool  = False,
    batchwise  : bool  = False
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
    batchwise  : bool  = False
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
    training : bool  = False
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


@LAYERS.register(name="drop_block2d")
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
        fast       : bool  = True
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


@LAYERS.register(name="drop_path")
class DropPath(Module):
    """
    Drop paths (Stochastic Depth) per sample.
    
    Args:
        drop_prob (float): Probability of the path to be zeroed. Defaults to 0.1.
    """
    
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, input: Tensor) -> Tensor:
        return drop_path(
            input     = input,
            drop_prob = self.drop_prob,
            training  = self.training
        )


LAYERS.register(name="alpha_dropout",         module=AlphaDropout)
LAYERS.register(name="dropout",               module=Dropout)
LAYERS.register(name="dropout1d",             module=Dropout1d)
LAYERS.register(name="dropout2d",             module=Dropout2d)
LAYERS.register(name="dropout3d",             module=Dropout3d)
LAYERS.register(name="feature_alpha_dropout", module=FeatureAlphaDropout)


# H1: - Embedding --------------------------------------------------------------

@LAYERS.register(name="patch_embedding")
class PatchEmbedding(Module):
    """
    2D Image to Patch Embedding.
    """

    def __init__(
        self,
        img_size   : Ints            = 224,
        patch_size : Ints            = 16,
        in_channels: int             = 3,
        embed_dim  : int             = 768,
        norm_layer : Callable | None = None,
        flatten    : bool            = True
    ):
        super().__init__()
        img_size         = to_2tuple(img_size)
        patch_size       = to_2tuple(patch_size)
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.grid_size   = (img_size[0] // patch_size[0],
                            img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten     = flatten
        self.proj = Conv2d(
            in_channels  = in_channels,
            out_channels = embed_dim,
            kernel_size  = patch_size,
            stride       = patch_size
        )
        self.norm = (norm_layer(embed_dim) if norm_layer else Identity())

    def forward(self, input: Tensor) -> Tensor:
        b, c, h, w = input.shape
        if h != self.img_size[0] or w != self.img_size[1]:
            raise ValueError(
                f"Input image size ({h}*{w}) doesn't match model "
                f"input size ({self.img_size[0]}*{self.img_size[1]})."
            )
        output = self.proj(input)
        if self.flatten:
            output = output.flatten(2).transpose(1, 2)  # BCHW -> BNC
        output = self.norm(output)
        return output


@LAYERS.register(name="rotary_embedding")
class RotaryEmbedding(Module):
    """
    Rotary position embedding.

    This is my initial attempt at impl rotary embedding for spatial use, it
    has not been well tested, and will  likely change. It will be moved to
    its own file.

    Following impl/resources were referenced for this impl:
        https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
        https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(self, dim: int, max_freq: int = 4):
        super().__init__()
        self.dim = dim
        self.register_buffer(
            name       = "bands",
            tensor     = 2 ** torch.linspace(0., max_freq - 1, self.dim // 4),
            persistent = False
        )

    def forward(self, input: Tensor) -> Tensor:
        # Assuming channel-first image where spatial dim are >= 2
        sin_emb, cos_emb = self.get_embed(input.shape[2:])
        rot = torch.stack([-input[..., 1::2], input[..., ::2]], -1).reshape(input.shape)
        return input * cos_emb + rot * sin_emb

    def get_embed(
        self,
        shape : torch.Size,
        device: torch.device = None,
        dtype : torch.dtype  = None
    ):
        device = device or self.bands.device
        dtype  = dtype  or self.bands.dtype
        if not isinstance(shape, torch.Size):
            shape = torch.Size(shape)
        n    = shape.numel()
        grid = torch.stack(
            torch.meshgrid(
                [torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=dtype)
                 for s in shape]
            ), dim=-1
        ).unsqueeze(-1)
        emb = grid * math.pi * self.bands
        sin = emb.sin().reshape(n, -1).repeat_interleave(2, -1)
        cos = emb.cos().reshape(n, -1).repeat_interleave(2, -1)
        return sin, cos


# H1: - Fusion -----------------------------------------------------------------

@LAYERS.register(name="concat")
class Concat(Module):
    """
    Concatenate a list of tensors along dimension.
    
    Args:
        dim (str | ellipsis | None): Dimension to concat to. Defaults to 1.
    """
    
    def __init__(self, dim: str | ellipsis | None = 1):
        super().__init__()
        self.dim = dim
        
    def forward(self, input: Sequence[Tensor]) -> Tensor:
        return torch.cat(to_list(input), dim=self.dim)


@LAYERS.register(name="softmax_fusion")
class SoftmaxFusion(Module):
    """
    Weighted sum of multiple layers https://arxiv.org/abs/1911.09070. Apply
    softmax to each weight, such that all weights are normalized to be a
    probability with value range from 0 to 1, representing the importance of
    each input
    
    Args:
        n (int): Number of inputs.
    """

    def __init__(self, n: int, weight: bool = False):
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


# H1: - Linear -----------------------------------------------------------------

LAYERS.register(name="bilinear",    module=Bilinear)
LAYERS.register(name="identity",    module=Identity)
LAYERS.register(name="lazy_linear", module=LazyLinear)
LAYERS.register(name="linear",      module=Linear)


# H1: - MLP --------------------------------------------------------------------

@LAYERS.register(name="conv_mlp")
class ConvMlp(Module):
    """
    MLP using 1x1 Convs that keeps spatial dims.
    """
    
    def __init__(
        self,
        in_features    : int,
        hidden_features: int | None      = None,
        out_features   : int | None      = None,
        act            : Callable        = ReLU,
        norm           : Callable | None = None,
        drop           : float           = 0.0
    ):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = Conv2d(
            in_channels  = in_features,
            out_channels = hidden_features,
            kernel_size  = (1, 1),
            bias         = True
        )
        self.norm = norm(hidden_features) if norm else Identity()
        self.act  = to_act_layer(act=act)
        self.fc2  = Conv2d(
            in_channels  = hidden_features,
            out_channels = out_features,
            kernel_size  = (1, 1),
            bias         = True
        )
        self.drop = Dropout(drop)
        
    def forward(self, input: Tensor) -> Tensor:
        output = self.fc1(input)
        output = self.norm(output)
        output = self.act(output)
        output = self.drop(output)
        output = self.fc2(output)
        return output


@LAYERS.register(name="glu_mlp")
class GluMlp(Module):
    """
    MLP w/ GLU style gating. See:
        https://arxiv.org/abs/1612.08083,
        https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        in_features    : int,
        hidden_features: int | None = None,
        out_features   : int | None = None,
        act            : Callable   = Sigmoid,
        drop           : float      = 0.0
    ):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        if hidden_features % 2 != 0:
            raise ValueError
        drop_probs = to_2tuple(drop)

        self.fc1   = Linear(in_features, hidden_features)
        self.act   = to_act_layer(act=act)
        self.drop1 = Dropout(drop_probs[0])
        self.fc2   = Linear(hidden_features // 2, out_features)
        self.drop2 = Dropout(drop_probs[1])

    def init_weights(self):
        # override init of fc1 w/ gate portion set to weight near zero, bias=1
        fc1_mid = self.fc1.bias.shape[0] // 2
        init.ones_(self.fc1.bias[fc1_mid:])
        init.normal_(self.fc1.weight[fc1_mid:], std=1e-6)

    def forward(self, input: Tensor) -> Tensor:
        output        = self.fc1(input)
        output, gates = output.chunk(2, dim=-1)
        output        = output * self.act(gates)
        output        = self.drop1(output)
        output        = self.fc2(output)
        output        = self.drop2(output)
        return output


@LAYERS.register(name="gated_mlp")
class GatedMlp(Module):
    """
    MLP as used in gMLP.
    """

    def __init__(
        self,
        in_features    : int,
        hidden_features: int | None      = None,
        out_features   : int | None      = None,
        act            : Callable        = GELU,
        gate           : Callable | None = None,
        drop           : float           = 0.0
    ):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        drop_probs      = to_2tuple(drop)

        self.fc1   = Linear(in_features, hidden_features)
        self.act   = to_act_layer(act=act)
        self.drop1 = Dropout(drop_probs[0])
        if gate is not None:
            if hidden_features % 2 != 0:
                raise ValueError
            self.gate = gate(hidden_features)
            # FIXME base reduction on gate property?
            hidden_features = hidden_features // 2
        else:
            self.gate = Identity()
        self.fc2   = Linear(hidden_features, out_features)
        self.drop2 = Dropout(drop_probs[1])

    def forward(self, input: Tensor) -> Tensor:
        output = self.fc1(input)
        output = self.act(output)
        output = self.drop1(output)
        output = self.gate(output)
        output = self.fc2(output)
        output = self.drop2(output)
        return output


@LAYERS.register(name="mlp")
class Mlp(Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks.
    """
    
    def __init__(
        self,
        in_features    : int,
        hidden_features: int | None = None,
        out_features   : int | None = None,
        act            : Callable   = GELU,
        drop           : float      = 0.0
    ):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs      = to_2tuple(drop)
        
        self.fc1   = Linear(in_features, hidden_features)
        self.act   = to_act_layer(act=act)
        self.drop1 = Dropout(drop_probs[0])
        self.fc2   = Linear(hidden_features, out_features)
        self.drop2 = Dropout(drop_probs[1])
        
    def forward(self, input: Tensor) -> Tensor:
        output = self.fc1(input)
        output = self.act(output)
        output = self.drop1(output)
        output = self.fc2(output)
        output = self.drop2(output)
        return output


# H1: - Normalization ----------------------------------------------------------

@LAYERS.register(name="batch_norm_act2d")
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
            dtype               = dtype
        )
        self.act = to_act_layer(act, inplace)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        if self.act is not None:
            output = self.act(output)
        return output


@LAYERS.register(name="batch_norm_relu2d")
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
        

@LAYERS.register(name="fraction_instance_norm2d")
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


@LAYERS.register(name="group_norm_act")
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
        drop_block  : Callable | None = None
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


@LAYERS.register(name="half_group_norm")
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


@LAYERS.register(name="half_instance_norm2d")
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
            num_features        = num_features // 2,
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


@LAYERS.register(name="half_layer_norm")
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


@LAYERS.register(name="layer_norm2d")
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


LAYERS.register(name="batch_norm1d",             module=BatchNorm1d)
LAYERS.register(name="batch_norm2d",             module=BatchNorm2d)
LAYERS.register(name="batch_norm3d",             module=BatchNorm3d)
LAYERS.register(name="group_norm",               module=GroupNorm)
LAYERS.register(name="layer_norm",               module=LayerNorm)
LAYERS.register(name="lazy_batch_norm1d",        module=LazyBatchNorm1d)
LAYERS.register(name="lazy_batch_norm2d",        module=LazyBatchNorm2d)
LAYERS.register(name="lazy_batch_norm3d",        module=LazyBatchNorm3d)
LAYERS.register(name="lazy_instance_norm1d",     module=LazyInstanceNorm1d)
LAYERS.register(name="lazy_instance_norm2d",     module=LazyInstanceNorm2d)
LAYERS.register(name="lazy_instance_norm3d",     module=LazyInstanceNorm3d)
LAYERS.register(name="local_response_norm",      module=LocalResponseNorm)
LAYERS.register(name="instance_norm1d",          module=InstanceNorm1d)
LAYERS.register(name="instance_norm2d",          module=InstanceNorm2d)
LAYERS.register(name="instance_norm3d",          module=InstanceNorm3d)
LAYERS.register(name="sync_batch_norm",          module=SyncBatchNorm)


# H1: - Padding ----------------------------------------------------------------

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
) -> int:
    """
    Calculate symmetric padding for a convolution.
    """
    return ((stride - 1) + dilation * (kernel_size - 1)) // 2


def to_same_padding(
    kernel_size: Ints,
    padding    : Ints | None = None,
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
    value      : float = 0
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


LAYERS.register(name="constant_pad1d",    module=ConstantPad1d)
LAYERS.register(name="constant_pad2d",    module=ConstantPad2d)
LAYERS.register(name="constant_pad3d",    module=ConstantPad3d)
LAYERS.register(name="reflection_pad1d",  module=ReflectionPad1d)
LAYERS.register(name="reflection_pad2d",  module=ReflectionPad2d)
LAYERS.register(name="reflection_pad3d",  module=ReflectionPad3d)
LAYERS.register(name="replication_pad1d", module=ReplicationPad1d)
LAYERS.register(name="replication_pad2d", module=ReplicationPad2d)
LAYERS.register(name="replication_pad3d", module=ReplicationPad3d)
LAYERS.register(name="zero_pad2d",        module=ZeroPad2d)


# H1: - Pooling ----------------------------------------------------------------

def adaptive_avg_max_pool2d(input: Tensor, output_size: int = 1) -> Tensor:
    avg = F.adaptive_avg_pool2d(input, output_size)
    max = F.adaptive_max_pool2d(input, output_size)
    return 0.5 * (avg + max)


def adaptive_cat_avg_max_pool2d(input: Tensor, output_size: int = 1) -> Tensor:
    avg = F.adaptive_avg_pool2d(input, output_size)
    max = F.adaptive_max_pool2d(input, output_size)
    return torch.cat((avg, max), 1)


def adaptive_pool2d(
    input      : Tensor,
    pool_type  : str = "avg",
    output_size: int = 1
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
    count_include_pad: bool = True
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
    ceil_mode  : bool = False
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


@LAYERS.register(name="adaptive_avg_max_pool2d")
class AdaptiveAvgMaxPool2d(Module):

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_avg_max_pool2d(
            input       = input,
            output_size = self.output_size
        )


@LAYERS.register(name="adaptive_cat_avg_max_pool2d")
class AdaptiveCatAvgMaxPool2d(Module):

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_cat_avg_max_pool2d(
            input       = input,
            output_size = self.output_size
        )


@LAYERS.register(name="adaptive_pool2d")
class AdaptivePool2d(Module):
    """
    Selectable global pooling layer with dynamic input kernel size.
    """

    def __init__(
        self,
        output_size: int  = 1,
        pool_type  : str  = "fast",
        flatten    : bool = False
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
        

@LAYERS.register(name="avg_pool_same2d")
class AvgPoolSame2d(AvgPool2d):
    """
    Tensorflow like 'same' wrapper for 2D average pooling.
    """

    def __init__(
        self,
        kernel_size      : Ints,
        stride           : Ints | None = None,
        padding          : Ints | None = 0,
        ceil_mode        : bool        = False,
        count_include_pad: bool        = True
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


@LAYERS.register(name="fast_adaptive_avg_pool2d")
class FastAdaptiveAvgPool2d(Module):

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.flatten = flatten

    def forward(self, input: Tensor) -> Tensor:
        return input.mean((2, 3), keepdim=not self.flatten)
    

@LAYERS.register(name="max_pool_same2d")
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
        ceil_mode  : bool        = False
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


@LAYERS.register(name="median_pool2d")
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
        stride     : Ints    	       = (1, 1),
        padding    : str | Ints | None = 0,
        same	   : bool			   = False
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


@LAYERS.register(name="spatial_pyramid_pooling")
class SpatialPyramidPooling(Module):
    """
    Spatial Pyramid Pooling layer used in YOLOv3-SPP.
    
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Sizes of several convolving kernels.
            Defaults to (5, 9, 13).
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : tuple = (5, 9, 13),
    ):
        super().__init__()
        hidden_channels = in_channels // 2  # Hidden channels
        in_channels2    = hidden_channels * (len(kernel_size) + 1)

        self.conv1 = ConvBnMish2d(
            in_channels  = in_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.conv2 = ConvBnMish2d(
            in_channels  = in_channels2,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.m = ModuleList([
            MaxPool2d(kernel_size=input, stride=1, padding=input // 2)
            for input in kernel_size
        ])
        
    def forward(self, input: Tensor) -> Tensor:
        output = self.conv1(input)
        output = self.conv2(torch.cat([output] + [m(output) for m in self.m], 1))
        return output


@LAYERS.register(name="spatial_pyramid_pooling_csp")
class SpatialPyramidPoolingCSP(Module):
    """
    Cross Stage Partial Spatial Pyramid Pooling layer used in YOLOv3-SPP.
    
    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        number (int): Number of bottleneck layers to use.
        shortcut (bool): Use shortcut connection?. Defaults to True.
        groups (int): Defaults to 1.
        expansion (float): Defaults to 0.5.
        kernel_size (tuple): Sizes of several convolving kernels.
            Defaults to (5, 9, 13).
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        number      : int   = 1,
        shortcut    : bool  = False,
        groups      : int   = 1,
        expansion   : float = 0.5,
        kernel_size : tuple = (5, 9, 13),
    ):
        super().__init__()
        hidden_channels = int(2 * out_channels * expansion)  # Hidden channels
        self.conv1 = ConvBnMish2d(
            in_channels  = in_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.conv2 = Conv2d(
            in_channels  = in_channels,
            out_channels = hidden_channels,
            kernel_size  = (1, 1),
            stride       = (1, 1),
            bias         = False
        )
        self.conv3 = ConvBnMish2d(
            in_channels  = hidden_channels,
            out_channels = hidden_channels,
            kernel_size  = 3,
            stride       = 1
        )
        self.conv4 = ConvBnMish2d(
            in_channels  = hidden_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.m = ModuleList([
            MaxPool2d(kernel_size=input, stride=(1, 1), padding=input // 2)
            for input in kernel_size
        ])
        self.conv5 = ConvBnMish2d(
            in_channels  = 4 * hidden_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1
        )
        self.conv6 = ConvBnMish2d(
            in_channels  = hidden_channels,
            out_channels = hidden_channels,
            kernel_size  = 3,
            stride       = 1
        )
        self.bn    = BatchNorm2d(2 * hidden_channels)
        self.act   = Mish()
        self.conv7 = ConvBnMish2d(
            in_channels  = 2 * hidden_channels,
            out_channels = hidden_channels,
            kernel_size  = 1,
            stride       = 1
        )
        
    def forward(self, input: Tensor) -> Tensor:
        x1     = self.conv4(self.conv3(self.conv1(input)))
        y1     = self.conv6(self.conv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2     = self.conv2(input)
        output = self.conv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        return output


LAYERS.register(name="adaptive_avg_pool1d",         module=AdaptiveAvgPool1d)
LAYERS.register(name="adaptive_avg_pool2d",         module=AdaptiveAvgPool2d)
LAYERS.register(name="adaptive_avg_pool3d",         module=AdaptiveAvgPool3d)
LAYERS.register(name="adaptive_max_pool1d",         module=AdaptiveMaxPool1d)
LAYERS.register(name="adaptive_max_pool2d",         module=AdaptiveMaxPool2d)
LAYERS.register(name="adaptive_max_pool3d",         module=AdaptiveMaxPool3d)
LAYERS.register(name="avg_pool1d",		            module=AvgPool1d)
LAYERS.register(name="avg_pool2d",		            module=AvgPool2d)
LAYERS.register(name="avg_pool3d", 		            module=AvgPool3d)
LAYERS.register(name="fractional_max_pool2d",       module=FractionalMaxPool2d)
LAYERS.register(name="fractional_max_pool3d",       module=FractionalMaxPool3d)
LAYERS.register(name="lp_pool_1d", 			        module=LPPool1d)
LAYERS.register(name="lp_pool_2d", 			        module=LPPool2d)
LAYERS.register(name="max_pool1d", 		            module=MaxPool1d)
LAYERS.register(name="max_pool2d", 		            module=MaxPool2d)
LAYERS.register(name="max_pool3d", 		            module=MaxPool3d)
LAYERS.register(name="max_unpool1d", 		        module=MaxUnpool1d)
LAYERS.register(name="max_unpool2d", 		        module=MaxUnpool2d)
LAYERS.register(name="max_unpool3d", 		        module=MaxUnpool3d)


# H1: - Residual ---------------------------------------------------------------

@LAYERS.register(name="residual_conv_act2d")
class ResidualConvAct2d(Module):
    """
    Basic Residual Conv2d + Act block.
    """
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : Ints,
        stride        : Ints              = 1,
        padding       : str | Ints | None = 0,
        dilation      : Ints              = 1,
        groups        : int               = 1,
        bias          : bool              = True,
        padding_mode  : str               = "zeros",
        device        : Any               = None,
        dtype         : Any               = None,
        act           : Callable | None   = ReLU,
        inplace       : bool              = True,
        **_
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
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
        self.act = to_act_layer(act=act, inplace=inplace)
        
    def forward(self, input: Tensor) -> Tensor:
        return torch.cat([input, self.act(self.conv(input))], 1)


@LAYERS.register(name="residual_dense_block")
@LAYERS.register(name="rdb")
class ResidualDenseBlock(Module):
    """
    Densely-Connected Residual block with activation layer.
    
    This is a more generalize version of:
    https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py

    Args:
        num_layers (int): Number of conv layers in the residual block.
        in_channels (int): Number of channels in the input image.
        growth_channels (int): Growth channel, i.e. intermediate channels.
        kernel_size (Ints): Size of the convolving kernel.
        lff_kernel_size (Ints): Size of the convolving kernel for the last
            conv layer.
        stride (Ints): Stride of the convolution. Default: `(1, 1)`.
        lff_stride (Ints):
            Stride of the convolution of the last layer. Default: `(1, 1)`.
        padding (str | Ints | None, optional):
            Padding added to both sides of the input. Defaults to 0.
        lff_padding:
            Padding added to both sides of the input of the last conv layer.
            Defaults to 0.
        dilation (Ints): Defaults to (1, 1).
        groups (int): Defaults to 1.
        bias (bool): Defaults to True.
        padding_mode (str): Defaults to zeros.
        device (Any): Defaults to None.
        dtype (Any): Defaults: None.
        apply_act (bool): Should use activation layer. Defaults to True.
        act (Callable | None): Activation layer or the name to build the
            activation layer.
        inplace (bool): Perform in-place activation. Defaults to True.
        residual_scale (float): It scales down the residuals by multiplying a
            constant between 0 and 1 before adding them to the main path to
            prevent instability. Defaults to 0.2.
    """

    def __init__(
        self,
        num_layers     : int,
        in_channels    : int,
        growth_channels: int,
        kernel_size    : Ints,
        lff_kernel_size: Ints,
        stride         : Ints              = 1,
        lff_stride     : Ints              = 1,
        padding        : str | Ints | None = 0,
        lff_padding    : str | Ints | None = 0,
        dilation       : Ints              = 1,
        groups         : int               = 1,
        bias           : bool              = True,
        padding_mode   : str               = "zeros",
        device         : Any               = None,
        dtype          : Any               = None,
        act            : Callable | None   = ReLU(),
        inplace        : bool              = True,
        residual_scale : float             = 0.2,
        **_
    ):
        super().__init__()
        self.num_layers     = num_layers
        self.residual_scale = residual_scale
        self.layers = Sequential(
            *[
                ResidualConvAct2d(
                    in_channels  = in_channels + i * growth_channels,
                    out_channels = growth_channels,
                    kernel_size  = kernel_size,
                    stride       = stride,
                    padding      = padding,
                    dilation     = dilation,
                    groups       = groups,
                    bias         = bias,
                    padding_mode = padding_mode,
                    device       = device,
                    dtype        = dtype,
                    act_layer    = act,
                    inplace      = inplace,
                )
                for i in range(num_layers)
            ]
        )
        #  local feature fusion
        self.lff = Conv2d(
            in_channels  = in_channels + num_layers * growth_channels,
            out_channels = growth_channels,
            kernel_size  = lff_kernel_size,
            stride       = lff_stride,
            padding      = lff_padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
 
    def forward(self, input: Tensor) -> Tensor:
        return input + self.residual_scale * self.lff(self.layers(input))


@LAYERS.register(name="residual_dense_block_5Conv_lrelu")
@LAYERS.register(name="rdb_5conv_lrelu")
class ResidualDenseBlock5ConvLReLU(ResidualDenseBlock):
    """
    Densely-Connected Residual block with 5 convolution layers + Leaky ReLU.
    
    References:
        https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py
    """

    def __init__(
        self,
        in_channels    : int               = 64,
        growth_channels: int               = 32,
        kernel_size    : Ints              = 3,
        lff_kernel_size: Ints              = 3,
        stride         : Ints              = 1,
        lff_stride     : Ints              = 1,
        padding        : str | Ints | None = 0,
        lff_padding    : str | Ints | None = 0,
        dilation       : Ints              = 1,
        groups         : int               = 1,
        bias           : bool              = True,
        padding_mode   : str               = "zeros",
        device         : Any               = None,
        dtype          : Any               = None,
        apply_act      : bool              = True,
        inplace        : bool              = True,
        residual_scale : float             = 0.2,
        **_
    ):
        super().__init__(
            num_layers      = 4,
            in_channels     = in_channels,
            growth_channels = growth_channels,
            kernel_size     = kernel_size,
            lff_kernel_size = lff_kernel_size,
            stride          = stride,
            lff_stride      = lff_stride,
            padding         = padding,
            lff_padding     = lff_padding,
            dilation        = dilation,
            groups          = groups,
            bias            = bias,
            padding_mode    = padding_mode,
            device          = device,
            dtype           = dtype,
            apply_act       = apply_act,
            act             = LeakyReLU(0.2, inplace=True),
            inplace         = inplace,
            residual_scale  = residual_scale,
        )
        self.initialize_weights([self.layers, self.lff], 0.1)
    
    # noinspection PyMethodMayBeStatic
    def initialize_weights(self, net_l: list | Module, scale: float = 1.0):
        if not isinstance(net_l, list):
            net_l = [net_l]
       
        for net in net_l:
            for m in net.modules():
                if isinstance(m, Conv2d):
                    init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                    m.weight.data *= scale  # For residual block
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, Linear):
                    init.kaiming_normal_(m.weight, a=0, mode="fan_in")
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias.data, 0.0)
                    
                    
@LAYERS.register(name="residual_in_residual_dense_block")
@LAYERS.register(name="rrdb")
class ResidualInResidualDenseBlock(Module):
    """
    Residual in Residual Dense Block with 3 Residual Dense Blocks.
    
    References:
        https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        growth_channels: int   = 32,
        residual_scale : float = 0.2,
        *args, **kwargs
    ):
        super().__init__()
        self.residual_scale = residual_scale
        self.rdb1 = ResidualDenseBlock5ConvLReLU(
            in_channels     = in_channels,
            growth_channels = growth_channels,
            residual_scale  = 0.2,
            *args, **kwargs
        )
        self.rdb2 = ResidualDenseBlock5ConvLReLU(
            in_channels     = in_channels,
            growth_channels = growth_channels,
            residual_scale  = 0.2,
            *args, **kwargs
        )
        self.rdb3 = ResidualDenseBlock5ConvLReLU(
            in_channels     = in_channels,
            growth_channels = growth_channels,
            residual_scale  = 0.2,
            *args, **kwargs
        )
    
    def forward(self, input: Tensor) -> Tensor:
        output = self.rdb1(input)
        output = self.rdb2(output)
        output = self.rdb3(output)
        return output * self.residual_scale + input


@LAYERS.register(name="residual_wide_activation_block")
@LAYERS.register(name="rwab")
class ResidualWideActivationBlock(Module):
    """
    Conv2d + BN + Act + Conv2d + BN.
    """

    def __init__(
        self,
        in_channels    : int,
        expand         : int             = 4,
        kernel_size    : Ints            = 3,
        stride         : Ints            = 1,
        padding        : str | Ints      = 1,
        dilation       : Ints            = 1,
        groups         : int             = 1,
        bias           : bool            = False,
        padding_mode   : str             = "zeros",
        device         : Any             = None,
        dtype          : Any             = None,
        act            : Callable | None = ReLU(inplace=True),
        inplace        : bool            = True,
        residual_scale : float           = 0.2,
        **_
    ):
        super().__init__()
        kernel_size         = to_2tuple(kernel_size)
        stride              = to_2tuple(stride)
        dilation            = to_2tuple(dilation)
        self.residual_scale = residual_scale
        
        self.conv1 = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels * expand,
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
        self.bn1   = BatchNorm2d(in_channels * expand)
        self.act   = to_act_layer(act=act, inplace=inplace)
        self.conv2 = Conv2d(
            in_channels  = in_channels * expand,
            out_channels = in_channels,
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
        self.bn12  = BatchNorm2d(in_channels)

    def forward(self, input: Tensor) -> Tensor:
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.act(output)
        output = self.conv2(output)
        output = self.bn1(output)
        output = output * self.residual_scale + input
        return output


RDB             = ResidualDenseBlock
RDB5ConvLReLu   = ResidualDenseBlock5ConvLReLU
RRDB            = ResidualInResidualDenseBlock
RWAB            = ResidualWideActivationBlock


# H1: - Sampling ---------------------------------------------------------------

@LAYERS.register(name="downsample_conv2d")
class DownsampleConv2d(Module):
    """
    
    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Scale factor. Defaults to 0.
        mode (str): Upsampling algorithm. One of: [`nearest`, `linear`,
            `bilinear`, `bicubic`, `trilinear`]. Defaults to bilinear.
        align_corners (bool): If True, the corner pixels of the input and output
            tensors are aligned, and thus preserving the values at those pixels.
            This only has effect when `mode` is `linear`, `bilinear`, or
            `trilinear`. Defaults to True.
    """
    
    def __init__(
        self,
        in_channels  : int,
        scale_factor : int  = 0,
        mode         : str  = "bilinear",
        align_corners: bool = False
    ):
        super().__init__()
        self.upsample = UpsampleConv2d(
            scale_factor  = 0.5,
            mode          = mode,
            align_corners = align_corners
        )
        self.conv = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels + scale_factor,
            kernel_size  = (1, 1),
            stride       = (1, 1),
            padding      = 0,
            bias         = False
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(self.upsample(input))
        

@LAYERS.register(name="inverse_pixel_shuffle")
class InversePixelShuffle(Module):
    
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
    def forward(self, input: Tensor) -> Tensor:
        ratio = self.scale_factor
        b     = input.size(0)
        c     = input.size(1)
        h     = input.size(2)
        w     = input.size(3)
        return (
            input.view(b, c, h // ratio, ratio, w // ratio, ratio)
                .permute(0, 1, 3, 5, 2, 4)
                .contiguous()
                .view(b, -1, h // ratio, w // ratio)
        )


@LAYERS.register(name="pixel_shuffle")
class PixelShuffle(Module):
    """
    Pixel Shuffle upsample layer. This module packs `F.pixel_shuffle()`
    and a Conv2d module together to achieve a simple upsampling with pixel
    shuffle.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (Ints): Kernel size of the conv layer to expand the
            channels.
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        scale_factor   : int,
        upsample_kernel: Ints,
    ):
        super().__init__()
        self.upsample_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels * scale_factor * scale_factor,
            kernel_size  = upsample_kernel,
            padding      = (upsample_kernel - 1) // 2
        )
        self.init_weights()
    
    def forward(self, input: Tensor) -> Tensor:
        output = self.upsample_conv(input)
        output = F.pixel_shuffle(output, self.scale_factor)
        return output


@LAYERS.register(name="scale")
class Scale(Module):
    """
    A learnable scale parameter. This layer scales the input by a learnable
    factor. It multiplies a learnable scale parameter of shape (1,) with
    input of any shape.
    
    Args:
        scale (float): Initial value of scale factor. Defaults to 1.0.
    """
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
        
    def forward(self, input: Tensor) -> Tensor:
        return input * self.scale


@LAYERS.register(name="skip_upsample_conv2d")
class SkipUpsampleConv2d(Module):
    """

    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Scale factor. Defaults to 0.
        mode (str): Upsampling algorithm. One of: [`nearest`, `linear`,
            `bilinear`, `bicubic`, `trilinear`]. Defaults to bilinear.
        align_corners (bool): If True, the corner pixels of the input and output
            tensors are aligned, and thus preserving the values at those pixels.
            This only has effect when :attr:`mode` is `linear`, `bilinear`, or
            `trilinear`. Defaults to True.
    """
    
    def __init__(
        self,
        in_channels  : int,
        scale_factor : int  = 0,
        mode         : str  = "bilinear",
        align_corners: bool = False
    ):
        
        super().__init__()
        self.up = Sequential(
            UpsampleConv2d(
                scale_factor  = 2.0,
                mode          = mode,
                align_corners = align_corners
            ),
            Conv2d(
                in_channels  = in_channels + scale_factor,
                out_channels = in_channels,
                kernel_size  = (1, 1),
                stride       = (1, 1),
                padding      = 0,
                bias         = False
            )
        )
        
    def forward(self, input: Tensor, skip: Tensor) -> Tensor:
        output  = self.up(input)
        output += skip
        return output


@LAYERS.register(name="upsample_conv2d")
class UpsampleConv2d(Module):
    """

    Args:
        in_channels (int): Number of input channels.
        scale_factor (int): Scale factor. Defaults to 0.
        mode (str): Upsampling algorithm. One of: [`nearest`, `linear`,
            `bilinear`, `bicubic`, `trilinear`]. Defaults to bilinear.
        align_corners (bool): If True, the corner pixels of the input and
            output tensors are aligned, and thus preserving the values at those
            pixels. This only has effect when :attr:`mode` is `linear`,
            `bilinear`, or `trilinear`. Defaults to False.
    """
    
    def __init__(
        self,
        in_channels  : int,
        scale_factor : int  = 0,
        mode         : str  = "bilinear",
        align_corners: bool = False
    ):
        
        super().__init__()
        self.upsample = UpsampleConv2d(
            scale_factor  = 2.0,
            mode          = mode,
            align_corners = align_corners
        )
        self.conv = Conv2d(
            in_channels  = in_channels + scale_factor,
            out_channels = in_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = False
        )
        
    def forward(self, input: Tensor) -> Tensor:
        return self.conv(self.upsample(input))


# H1: - Model Specific Layers --------------------------------------------------

# H2: - ZeroDCE/ZeroDCE++ ------------------------------------------------------

@LAYERS.register(name="le_curve")
class LECurve(Module):
    """
    Light-Enhancement Curve used in ZeroDCE model.
    """
    
    def forward(self, input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # Split
        x   = input[0]
        x_r = input[1]
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        # Merge
        x  = x  + r1 * (torch.pow(x,  2) - x)
        x  = x  + r2 * (torch.pow(x,  2) - x)
        x  = x  + r3 * (torch.pow(x,  2) - x)
        x1 = x  + r4 * (torch.pow(x,  2) - x)
        x  = x1 + r5 * (torch.pow(x1, 2) - x1)
        x  = x  + r6 * (torch.pow(x,  2) - x)
        x  = x  + r7 * (torch.pow(x,  2) - x)
        x2 = x  + r8 * (torch.pow(x,  2) - x)
        x  = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return x, x1, x2


# H2: - Misc -------------------------------------------------------------------

@LAYERS.register(name="context_block")
class ContextBlock(Module):
    """
    ContextBlock module in GCNet. See 'GCNet: Non-local Networks Meet
    Squeeze-Excitation Networks and Beyond' (https://arxiv.org/abs/1904.11492)
    for details.
    
    Args:
        in_channels (int): Channels of the input feature map.
        ratio (float): Ratio of channels of transform bottleneck
        pooling_type (str): Pooling method for context modeling.
            One of: [`att`, `avg`]. `att` stands for attention pooling and
            `avg` stands for average pooling. Defaults to `att`.
        fusion_types (Strs):
            Fusion method for feature fusion, One of: [`channels_add`,
            `channel_mul`]. `channels_add` stands for channel-wise addition
            and `channel_mul` stands for multiplication.
            Defaults to (`channel_add`,).
    """
    
    def __init__(
        self,
        in_channels : int,
        ratio       : float,
        pooling_type: str  = "att",
        fusion_types: Strs = ("channel_add", ),
        *args, **kwargs
    ):
        super().__init__()
        
        if pooling_type not in ["avg", "att"]:
            raise ValueError
        if not isinstance(fusion_types, (list, tuple)):
            raise ValueError
        
        valid_fusion_types = ["channel_add", "channel_mul"]
        if not all([f in valid_fusion_types for f in fusion_types]):
            raise ValueError
        if len(fusion_types) <= 0:
            raise ValueError("At least one fusion should be used.")
        
        planes = int(in_channels * ratio)
        
        if pooling_type == "att":
            self.conv_mask = Conv2d(
                in_channels  = in_channels,
                out_channels = 1,
                kernel_size  = (1, 1)
            )
            self.softmax = Softmax(dim=2)
        else:
            self.avg_pool = AdaptiveAvgPool2d(1)
        
        if "channel_add" in fusion_types:
            self.channel_add_conv = Sequential(
                Conv2d(
                    in_channels  = in_channels,
                    out_channels = planes,
                    kernel_size  = (1, 1)
                ),
                LayerNorm(normalized_shape=[planes, 1, 1]),
                ReLU(inplace=True),
                Conv2d(
                    in_channels  = planes,
                    out_channels = in_channels,
                    kernel_size  = (1, 1)
                )
            )
        else:
            self.channel_add_conv = None
        
        if "channel_mul" in fusion_types:
            self.channel_mul_conv = Sequential(
                Conv2d(
                    in_channels  = in_channels,
                    out_channels = planes,
                    kernel_size  = (1, 1)
                ),
                LayerNorm(normalized_shape=[planes, 1, 1]),
                ReLU(inplace=True),
                Conv2d(
                    in_channels  = planes,
                    out_channels = in_channels,
                    kernel_size  = (1, 1)
                )
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()
    
    def forward(self, input: Tensor) -> Tensor:
        # [N, C, 1, 1]
        context = self.spatial_pool(input=input)
        yhat    = input
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term  = torch.sigmoid(self.channel_mul_conv(context))
            yhat             *= channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            yhat             = yhat + channel_add_term
        return yhat
    
    def spatial_pool(self, input: Tensor) -> Tensor:
        b, c, h, w = input.size()
        if self.pooling_type == "att":
            input_x = input
            # [N, C, H * W]
            input_x = input_x.view(b, c, h * w)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(input)
            # [N, 1, H * W]
            context_mask = context_mask.view(b, 1, h * w)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(b, c, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(input)
        return context


@LAYERS.register(name="flatten")
class Flatten(Module):
    """
    Flatten the image. Commonly used after `AdaptiveAvgPool2d(1)` to remove
    last 2 dimensions.
    
    Args:
        channels (int): Channels to flatten the features to. Default: `-1`.
    """
    
    def __init__(self, channels: int = -1):
        super().__init__()
        self.channels = channels
        
    def forward(self, input: Tensor) -> Tensor:
        output = input.view(input.shape[0], self.channels)
        return output


@LAYERS.register(name="mean")
class Mean(Module):
    """
    Calculate mean of the image.
    
    Attributes:
        dim: Specify the dimension to calculate mean. Defaults to None.
        keepdim (bool): Defaults to False.
    """
    
    def __init__(
        self,
        dim    : Sequence[str | ellipsis | None] = None,
        keepdim: bool                            = False,
    ):
        super().__init__()
        self.dim     = dim
        self.keepdim = keepdim
        
    def forward(self, input: Tensor) -> Tensor:
        return input.mean(dim=self.dim, keepdim=self.keepdim)


@LAYERS.register(name="original_resolution_block")
@LAYERS.register(name="ors")
class OriginalResolutionBlock(Module):
    """
    Original Resolution Block.
    
    Args:
        channels (int): Number of input and output channels.
        kernel_size (Ints): Kernel size of the convolution layer.
        reduction (int): Reduction factor. Defaults to 16.
        bias (bool): Defaults to False.
        act (Callable): Activation function.
        num_cab (int): Number of CAB modules used.
    """
    
    def __init__(
        self,
        channels   : int,
        kernel_size: Ints,
        reduction  : int,
        bias       : bool,
        act        : Callable,
        num_cab    : int,
    ):
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        padding 	= kernel_size[0] // 2
        act         = to_act_layer(act)
        body = [
            CAB(
                channels    = channels,
                reduction   = reduction,
                kernel_size = kernel_size,
                bias        = bias,
                act         = act
            )
            for _ in range(num_cab)
        ]
        body.append(
            Conv2d(
                in_channels  = channels,
                out_channels = channels,
                kernel_size  = kernel_size,
                stride       = (1, 1),
                padding      = padding,
                bias         = bias
            )
        )
        self.body = Sequential(*body)
        
    def forward(self, input: Tensor) -> Tensor:
        output  = self.body(input)
        output += input
        return output


@LAYERS.register(name="squeeze_and_excite_layer")
@LAYERS.register(name="se_layer")
class SqueezeAndExciteLayer(Module):
    
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc       = Sequential(
            Linear(channel, int(channel / reduction), bias=False),
            ReLU(inplace=True),
            Linear(int(channel / reduction), channel, bias=False),
            Sigmoid()
        )
    
    def forward(self, input: Tensor) -> Tensor:
        b, c, _, _ = input.size()
        y          = self.avg_pool(input).view(b, c)
        y          = self.fc(y).view(b, c, 1, 1)
        return input * y.expand_as(input)


ORS     = OriginalResolutionBlock
SELayer = SqueezeAndExciteLayer
