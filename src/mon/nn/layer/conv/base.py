#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements convolutional layers."""

from __future__ import annotations

__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv2dBn",
    "Conv2dNormAct",
    "Conv2dSame",
    "Conv2dTF",
    "Conv3d",
    "Conv3dNormAct",
    "ConvNormAct",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DSConv2d",
    "DSConv2dReLU",
    "DSConvAct2d",
    "DWConv2d",
    "DepthwiseConv2d",
    "DepthwiseSeparableConv2d",
    "DepthwiseSeparableConv2dReLU",
    "DepthwiseSeparableConvAct2d",
    "LazyConv1d",
    "LazyConv2d",
    "LazyConv3d",
    "LazyConvTranspose1d",
    "LazyConvTranspose2d",
    "LazyConvTranspose3d",
    "PWConv2d",
    "PointwiseConv2d",
    "conv2d_same",
]

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import misc

from mon import core
from mon.core.typing import _callable, _size_2_t, _size_any_t
from mon.globals import LAYERS
from mon.nn.layer import activation, base, normalization, padding as pad

math = core.math


# region Convolution

def conv2d_same(
    input   : torch.Tensor,
    weight  : torch.Tensor,
    bias    : torch.Tensor | None = None,
    stride  : _size_any_t         = 1,
    padding : _size_any_t | str   = 0,
    dilation: _size_any_t         = 1,
    groups  : int                 = 1,
):
    """Functional interface for Same Padding Convolution 2D."""
    x = input
    y = pad.pad_same(
        input       = x,
        kernel_size = weight.shape[-2: ],
        stride      = stride,
        dilation    = dilation
    )
    y = F.conv2d(
        input    = y,
        weight   = weight,
        bias     = bias,
        stride   = stride,
        padding  = padding,
        dilation = dilation,
        groups   = groups
    )
    return y


@LAYERS.register()
class Conv1d(base.ConvLayerParsingMixin, nn.Conv1d):
    pass


@LAYERS.register()
class Conv2d(base.ConvLayerParsingMixin, nn.Conv2d):
    pass


@LAYERS.register()
class Conv2dBn(base.ConvLayerParsingMixin, nn.Module):
    """Conv2d + BatchNorm."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = False,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
        bn          : bool | None     = True,
        eps         : float           = 1e-5,
        momentum    : float           = 0.01,
        affine      : bool            = True,
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
        self.bn = normalization.BatchNorm2d(
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


@LAYERS.register()
class Conv2dSame(base.ConvLayerParsingMixin, nn.Conv2d):
    """TensorFlow like ``SAME`` convolution wrapper for 2D convolutions."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
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


@LAYERS.register()
class Conv2dTF(base.ConvLayerParsingMixin, nn.Conv2d):
    """Implementation of 2D convolution in TensorFlow with :param:`padding` as
    ``'same'``, which applies padding to input (if needed) so that input image
    gets fully covered by filter and stride you specified. For stride of ``1``,
    this will ensure that the output image size is the same as input. For stride
    of ``2``, output dimensions will be half, for example.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
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
        img_h, img_w = x.size()[-2:]
        kernel_h, kernel_w = self.weight.size()[-2:]
        stride_h, stride_w = self.stride
        output_h = math.ceil(img_h / stride_h)
        output_w = math.ceil(img_w / stride_w)
        pad_h = max((output_h - 1) * self.stride[0] + (kernel_h - 1) * self.dilation[0] + 1 - img_h, 0)
        pad_w = max((output_w - 1) * self.stride[1] + (kernel_w - 1) * self.dilation[1] + 1 - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                input = x,
                pad   = [pad_w // 2, pad_w - pad_w // 2,
                         pad_h // 2, pad_h - pad_h // 2]
            )
        y = F.conv2d(
            input    = x,
            weight   = self.weight,
            bias     = self.bias,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = self.groups
        )
        return y


@LAYERS.register()
class Conv3d(base.ConvLayerParsingMixin, nn.Conv3d):
    pass


@LAYERS.register()
class ConvNormAct(base.ConvLayerParsingMixin, misc.ConvNormActivation):
    pass


@LAYERS.register()
class Conv2dNormAct(base.ConvLayerParsingMixin, misc.Conv2dNormActivation):
    pass


@LAYERS.register()
class Conv3dNormAct(base.ConvLayerParsingMixin, misc.Conv3dNormActivation):
    pass


@LAYERS.register()
class LazyConv1d(base.ConvLayerParsingMixin, nn.LazyConv1d):
    pass


@LAYERS.register()
class LazyConv2d(base.ConvLayerParsingMixin, nn.LazyConv2d):
    pass


@LAYERS.register()
class LazyConv3d(base.ConvLayerParsingMixin, nn.LazyConv3d):
    pass


LAYERS.register(name="ConvNormAct",   module=ConvNormAct)
LAYERS.register(name="Conv2dNormAct", module=Conv2dNormAct)
LAYERS.register(name="Conv3dNormAct", module=Conv3dNormAct)

# endregion


# region Depthwise Separable Convolution

@LAYERS.register()
class DepthwiseConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Conv2d."""
    
    def __init__(
        self,
        in_channels : int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.dw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.dw_conv(x)
        return y


@LAYERS.register()
class PointwiseConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Pointwise Conv2d."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        groups      : int             = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.pw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
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
        y = self.pw_conv(x)
        return y
    

@LAYERS.register()
class DepthwiseSeparableConv2d(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.dw_conv = Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
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
            kernel_size  = 1,
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


@LAYERS.register()
class DepthwiseSeparableConvAct2d(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d + Activation."""
    
    def __init__(
        self,
        in_channels   : int,
        out_channels  : int,
        kernel_size   : _size_2_t,
        stride        : _size_2_t       = 1,
        padding       : _size_2_t | str = 0,
        dilation      : _size_2_t       = 1,
        bias          : bool            = True,
        padding_mode  : str             = "zeros",
        device        : Any             = None,
        dtype         : Any             = None,
        act_layer     : _callable       = nn.ReLU,
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = act_layer()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.ds_conv(x)
        y = self.act(y)
        return y


@LAYERS.register()
class DepthwiseSeparableConv2dReLU(base.ConvLayerParsingMixin, nn.Module):
    """Depthwise Separable Conv2d ReLU."""
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t       = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t       = 1,
        bias        : bool            = True,
        padding_mode: str             = "zeros",
        device      : Any             = None,
        dtype       : Any             = None,
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.act = activation.ReLU(inplace=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.ds_conv(x)
        y = self.act(y)
        return y


DWConv2d     = DepthwiseConv2d
PWConv2d     = PointwiseConv2d
DSConv2d     = DepthwiseSeparableConv2d
DSConvAct2d  = DepthwiseSeparableConvAct2d
DSConv2dReLU = DepthwiseSeparableConv2dReLU
LAYERS.register(name="DWConv2d",     module=DWConv2d)
LAYERS.register(name="PWConv2d",     module=PWConv2d)
LAYERS.register(name="DSConv2d",     module=DSConv2d)
LAYERS.register(name="DSConvAct2d",  module=DSConvAct2d)
LAYERS.register(name="DSConv2dReLU", module=DSConv2dReLU)

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
