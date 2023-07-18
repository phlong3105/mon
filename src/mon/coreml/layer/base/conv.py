#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements convolutional layers."""

from __future__ import annotations

__all__ = [
    "Conv1d", "Conv2d", "conv2d_same", "Conv2dBn", "Conv2dNormActivation",
    "Conv2dSame", "Conv3d", "Conv3dNormActivation", "ConvNormActivation",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "LazyConv1d",
    "LazyConv2d", "LazyConv3d", "LazyConvTranspose1d", "LazyConvTranspose2d",
    "LazyConvTranspose3d", "Conv2dTF",
]

from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional
from torchvision.ops import misc

from mon.coreml.layer.base import (
    activation, base, normalization, padding as pad,
)
from mon.coreml.layer.typing import _size_2_t
from mon.foundation import math
from mon.globals import LAYERS


# region Convolution

def conv2d_same(
    input   : torch.Tensor,
    weight  : torch.Tensor,
    bias    : torch.Tensor | None = None,
    stride  : _size_2_t           = 1,
    padding : _size_2_t | str     = 0,
    dilation: _size_2_t           = 1,
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
    """Tensorflow like `SAME` convolution wrapper for 2D convolutions."""
    
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
    'same', which applies padding to input (if needed) so that input image gets
    fully covered by filter and stride you specified. For stride of 1, this will
    ensure that the output image size is the same as input. For stride of 2,
    output dimensions will be half, for example.
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


@LAYERS.register()
class Conv3d(base.ConvLayerParsingMixin, nn.Conv3d):
    pass


@LAYERS.register()
class ConvNormActivation(base.ConvLayerParsingMixin, misc.ConvNormActivation):
    pass


@LAYERS.register()
class Conv2dNormActivation(base.ConvLayerParsingMixin, misc.Conv2dNormActivation):
    pass


@LAYERS.register()
class Conv3dNormActivation(base.ConvLayerParsingMixin, misc.Conv3dNormActivation):
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
