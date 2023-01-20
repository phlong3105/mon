#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements convolutional layers."""

from __future__ import annotations

__all__ = [
    "BSConv2dS", "BSConv2dU", "Conv1d", "Conv2d", "conv2d_same", "Conv2dBn",
    "Conv2dNormActivation", "Conv2dSame", "Conv3d", "Conv3dNormActivation",
    "ConvNormActivation", "ConvTranspose1d", "ConvTranspose2d",
    "ConvTranspose3d", "DepthwiseSeparableConv2d",
    "DepthwiseSeparableConv2dReLU", "GhostConv2d", "LazyConv1d", "LazyConv2d",
    "LazyConv3d", "LazyConvTranspose1d", "LazyConvTranspose2d",
    "LazyConvTranspose3d", "SubspaceBlueprintSeparableConv2d", "Conv2dTF",
    "UnconstrainedBlueprintSeparableConv2d",
]

from typing import Any

import torch
from torch import nn
from torch.nn import functional
from torchvision.ops import misc as torchvision_misc

from mon.coreml import constant
from mon.coreml.layer import base
from mon.coreml.layer.common import (
    activation, normalization, padding as pad,
)
from mon.coreml.typing import CallableType, Int2T
from mon.foundation import math


# region Convolution

def conv2d_same(
    input   : torch.Tensor,
    weight  : torch.Tensor,
    bias    : torch.Tensor | None = None,
    stride  : Int2T               = 1,
    padding : Int2T | str         = 0,
    dilation: Int2T               = 1,
    groups  : int                 = 1,
    *args, **kwargs
):
    """Functional interface for Same Padding Convolution 2D."""
    x = input
    y = pad.pad_same(
        input       = x,
        kernel_size = weight.shape[-2:],
        stride      = stride,
        dilation    = dilation
    )
    y = functional.conv2d(
        input    = y,
        weight   = weight,
        bias     = bias,
        stride   = stride,
        padding  = padding,
        dilation = dilation,
        groups   = groups
    )
    return y


@constant.LAYER.register()
class Conv1d(base.ConvLayerParsingMixin, nn.Conv1d):
    pass


@constant.LAYER.register()
class Conv2d(base.ConvLayerParsingMixin, nn.Conv2d):
    pass


@constant.LAYER.register()
class Conv2dBn(base.ConvLayerParsingMixin, nn.Module):
    """Conv2d + BatchNorm."""

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T       = 1,
        padding     : Int2T | str = 0,
        dilation    : Int2T       = 1,
        groups      : int         = 1,
        bias        : bool        = False,
        padding_mode: str         = "zeros",
        device      : Any         = None,
        dtype       : Any         = None,
        bn          : bool | None = True,
        inplace     : bool        = True,
        eps         : float       = 1e-5,
        momentum    : float       = 0.01,
        affine      : bool        = True,
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
        self.bn   = normalization.BatchNorm2d(
            num_features = out_channels,
            eps          = eps,
            momentum     = momentum,
            affine       = affine,
        ) if bn is True else None
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv(x)
        if self.bn is not None:
            y = self.bn(y)
        return y


@constant.LAYER.register()
class Conv2dSame(base.ConvLayerParsingMixin, nn.Conv2d):
    """Tensorflow like `SAME` convolution wrapper for 2D convolutions."""

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T       = 1,
        padding     : Int2T | str = 0,
        dilation    : Int2T       = 1,
        groups      : int         = 1,
        bias        : bool        = True,
        padding_mode: str         = "zeros",
        device      : Any         = None,
        dtype       : Any         = None,
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = conv2d_same(
            input    = x,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )
        return y


@constant.LAYER.register()
class Conv2dTF(base.ConvLayerParsingMixin, nn.Conv2d):
    """Implementation of 2D convolution in TensorFlow with `padding` as "same",
    which applies padding to input (if needed) so that input image gets fully
    covered by filter and stride you specified. For stride of 1, this will
    ensure that output image size is same as input. For stride of 2, output
    dimensions will be half, for example.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : Int2T,
        stride      : Int2T       = 1,
        padding     : Int2T | str = 0,
        dilation    : Int2T       = 1,
        groups      : int         = 1,
        bias        : bool        = True,
        padding_mode: str         = "zeros",
        device      : Any         = None,
        dtype       : Any         = None,
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
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x                  = input
        img_h, img_w       = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h    = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0)
        pad_w    = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = functional.pad(
                input = x,
                pad   = [pad_w // 2, pad_w - pad_w // 2,
                         pad_h // 2, pad_h - pad_h // 2]
            )
        y = functional.conv2d(
            input    = x,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )
        return y


@constant.LAYER.register()
class Conv3d(base.ConvLayerParsingMixin, nn.Conv3d):
    pass


@constant.LAYER.register()
class ConvNormActivation(
    base.ConvLayerParsingMixin,
    torchvision_misc.ConvNormActivation,
):
    pass


@constant.LAYER.register()
class Conv2dNormActivation(
    base.ConvLayerParsingMixin,
    torchvision_misc.Conv2dNormActivation,
):
    pass


@constant.LAYER.register()
class Conv3dNormActivation(
    base.ConvLayerParsingMixin,
    torchvision_misc.Conv3dNormActivation
):
    pass


@constant.LAYER.register()
class LazyConv1d(base.ConvLayerParsingMixin, nn.LazyConv1d):
    pass


@constant.LAYER.register()
class LazyConv2d(base.ConvLayerParsingMixin, nn.LazyConv2d):
    pass


@constant.LAYER.register()
class LazyConv3d(base.ConvLayerParsingMixin, nn.LazyConv3d):
    pass

# endregion


# region Blueprint Separable Convolution

@constant.LAYER.register()
class SubspaceBlueprintSeparableConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Subspace Blueprint Separable Conv2d adopted from the paper: "Rethinking
    Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to
    Improved MobileNets".
    
    References:
        https://github.com/zeiss-microscopy/BSConv
    """

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : Int2T,
        stride          : Int2T               = 1,
        padding         : Int2T | str         = 0,
        dilation        : Int2T               = 1,
        groups          : int                 = 1,
        bias            : bool                = True,
        padding_mode    : str                 = "zeros",
        device          : Any                 = None,
        dtype           : Any                 = None,
        p               : float               = 0.25,
        min_mid_channels: int                 = 4,
        act             : CallableType | None = None,
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
        self.act1     = activation.to_act_layer(act=act, num_features=mid_channels)  # if act else None
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
        self.act2    = activation.to_act_layer(act=act, num_features=out_channels)  # if act else None
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
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv1(x)
        if self.act1 is not None:
            y = self.act1(y)
        y = self.pw_conv2(y)
        if self.act2 is not None:
            y = self.act2(y)
        y = self.dw_conv(y)
        return y
    
    def regularization_loss(self):
        w   = self.pw_conv1.weight[:, :, 0, 0]
        wwt = torch.mm(w, torch.transpose(w, 0, 1))
        i   = torch.eye(wwt.shape[0], device=wwt.device)
        return torch.norm(wwt - i, p="fro")


@constant.LAYER.register()
class UnconstrainedBlueprintSeparableConv2d(
    base.ConvLayerParsingMixin,
    nn.Module,
):
    """Unconstrained Blueprint Separable Conv2d adopted from the paper:
    "Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations
    Lead to Improved MobileNets," CVPR 2020.
    
    References:
        https://github.com/zeiss-microscopy/BSConv
    """

    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : Int2T,
        stride        : Int2T               = 1,
        padding       : Int2T | str         = 0,
        dilation      : Int2T               = 1,
        groups        : int                 = 1,
        bias          : bool                = True,
        padding_mode  : str                 = "zeros",
        device        : Any                 = None,
        dtype         : Any                 = None,
        act           : CallableType | None = None,
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
        self.act     = activation.to_act_layer(act=act, num_features=out_channels)
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
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pw_conv(x)
        if self.act is not None:
            y = self.act(y)
        y = self.dw_conv(y)
        return y


BSConv2dS = SubspaceBlueprintSeparableConv2d
BSConv2dU = UnconstrainedBlueprintSeparableConv2d
constant.LAYER.register(BSConv2dS)
constant.LAYER.register(BSConv2dU)

# endregion


# region Depthwise Separable Convolution

@constant.LAYER.register()
class DepthwiseSeparableConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d."""

    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        dw_kernel_size: Int2T,
        pw_kernel_size: Int2T,
        dw_stride     : Int2T       = 1,
        dw_padding    : Int2T | str = 0,
        pw_stride     : Int2T       = 1,
        pw_padding    : Int2T | str = 0,
        dilation      : Int2T       = 1,
        groups        : int         = 1,
        bias          : bool        = True,
        padding_mode  : str         = "zeros",
        device        : Any         = None,
        dtype         : Any         = None,
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
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.dw_conv(x)
        y = self.pw_conv(y)
        return y


@constant.LAYER.register()
class DepthwiseSeparableConv2dReLU(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d ReLU."""

    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        dw_kernel_size: Int2T,
        pw_kernel_size: Int2T,
        dw_stride     : Int2T       = 1,
        pw_stride     : Int2T       = 1,
        dw_padding    : Int2T | str = 0,
        pw_padding    : Int2T | str = 0,
        dilation      : Int2T       = 1,
        groups        : int         = 1,
        bias          : bool        = True,
        padding_mode  : str         = "zeros",
        device        : Any         = None,
        dtype         : Any         = None,
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
        self.act = activation.ReLU(inplace=True)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.dw_conv(x)
        y = self.pw_conv(y)
        y = self.act(y)
        return y

# endregion


# region Ghost Convolution

class GhostConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Ghost Conv2d adopted from the paper: "GhostNet: More Features from Cheap
    Operations," CVPR 2020.
    
    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/ghostnet_pytorch/ghostnet.py
    """
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        ratio         : int                        = 2,
        kernel_size   : Int2T                      = 1,
        dw_kernel_size: Int2T                      = 3,
        stride        : Int2T                      = 1,
        padding       : Int2T | str | None         = None,
        dilation      : Int2T                      = 1,
        groups        : int                        = 1,
        bias          : bool                       = True,
        padding_mode  : str                        = "zeros",
        device        : Any                        = None,
        dtype         : Any                        = None,
        act           : CallableType | bool | None = activation.ReLU,
        *args, **kwargs
    ):
        super().__init__()
        self.out_channels = out_channels
        init_channels     = math.ceil(out_channels / ratio)
        new_channels      = init_channels * (ratio - 1)
        
        self.primary_conv = nn.Sequential(
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
            normalization.BatchNorm2d(init_channels),
            activation.to_act_layer(act=act, inplace=True),
        )
        self.cheap_operation = nn.Sequential(
            Conv2d(
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
            normalization.BatchNorm2d(new_channels),
            activation.to_act_layer(act=act, inplace=True),
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y1 = self.primary_conv(x)
        y2 = self.cheap_operation(y1)
        y  = torch.cat([y1, y2], dim=1)
        y  = y[:, :self.out_channels, :, :]
        return y

# endregion


# region Transposed Convolution

class ConvTranspose1d(base.ConvLayerParsingMixin, nn.ConvTranspose1d):
    pass


class ConvTranspose2d(base.ConvLayerParsingMixin, nn.ConvTranspose2d):
    pass


class ConvTranspose3d(base.ConvLayerParsingMixin, nn.ConvTranspose3d):
    pass


class LazyConvTranspose1d(base.ConvLayerParsingMixin, nn.LazyConvTranspose1d):
    pass


class LazyConvTranspose2d(base.ConvLayerParsingMixin, nn.LazyConvTranspose2d):
    pass


class LazyConvTranspose3d(base.ConvLayerParsingMixin, nn.LazyConvTranspose3d):
    pass

# endregion
