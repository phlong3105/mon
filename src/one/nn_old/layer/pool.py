#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Pooling Layers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Mish

from one.core import Int2T
from one.core import Padding2T
from one.core import Padding4T
from one.core import POOL_LAYERS
from one.core import to_2tuple
from one.core import to_4tuple
from one.nn.layer.conv_norm_act import ConvBnMish
from one.nn.layer.padding import get_padding
from one.nn.layer.padding import get_padding_value
from one.nn.layer.padding import pad_same

__all__ = [
    "adaptive_avg_max_pool2d",
    "adaptive_cat_avg_max_pool2d",
    "avg_pool2d_same",
    "create_pool2d",
    "max_pool2d_same",
    "select_adaptive_pool2d",
    "AdaptiveAvgMaxPool2d",
    "AdaptiveCatAvgMaxPool2d",
    "AvgPool2dSame",
    "BlurPool2d",
    "FastAdaptiveAvgPool2d",
    "MaxPool2dSame", 
    "MedianPool2d",
    "SelectAdaptivePool2d",
    "SpatialPyramidPooling",
    "SpatialPyramidPoolingCSP",
    "AdaptiveAvgMaxPool",
    "AvgPoolSame",
    "BlurPool",
    "FastAdaptiveAvgPool",
    "MaxPoolSame",
    "MedianPool",
    "SelectAdaptivePool",
    "SPP",
    "SPPCSP",
]


# MARK: - Functional

def adaptive_avg_max_pool2d(x: Tensor, output_size: int = 1) -> Tensor:
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_cat_avg_max_pool2d(x: Tensor, output_size: int = 1) -> Tensor:
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def avg_pool2d_same(
    x                : Tensor,
    kernel_size      : Int2T,
    stride           : Int2T,
    padding          : Padding2T = (0, 0),
    ceil_mode        : bool      = False,
    count_include_pad: bool      = True
) -> Tensor:
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(
        x, kernel_size, stride, padding, ceil_mode, count_include_pad
    )


def max_pool2d_same(
    x          : Tensor,
    kernel_size: Int2T,
    stride     : Int2T,
    padding    : Padding2T = (0, 0),
    dilation   : Int2T    = (1, 1),
    ceil_mode  : bool      = False
) -> Tensor:
    x = pad_same(x, kernel_size, stride, value=-float("inf"))
    return F.max_pool2d(x, kernel_size, stride, padding, dilation, ceil_mode)


def select_adaptive_pool2d(
    x: Tensor, pool_type: str = "avg", output_size: int = 1
) -> Tensor:
    """Selectable global pooling function with dynamic input kernel size."""
    if pool_type == "avg":
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == "avg_max":
        x = adaptive_avg_max_pool2d(x, output_size)
    elif pool_type == "cat_avg_max":
        x = adaptive_cat_avg_max_pool2d(x, output_size)
    elif pool_type == "max":
        x = F.adaptive_max_pool2d(x, output_size)
    elif True:
        raise ValueError("Invalid pool type: %s" % pool_type)
    return x


# MARK: - Modules

@POOL_LAYERS.register(name="adaptive_avg_max_pool2d")
class AdaptiveAvgMaxPool2d(nn.Module):

    # MARK: Magic Functions

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_avg_max_pool2d(x, self.output_size)


@POOL_LAYERS.register(name="adaptive_cat_avg_max_pool2d")
class AdaptiveCatAvgMaxPool2d(nn.Module):

    # MARK: Magic Functions

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_cat_avg_max_pool2d(x, self.output_size)


@POOL_LAYERS.register(name="avg_pool2d_same")
class AvgPool2dSame(nn.AvgPool2d):
    """Tensorflow like 'same' wrapper for 2D average pooling."""

    # MARK: Magic Functions

    def __init__(
        self,
        kernel_size      : Int2T,
        stride           : Optional[Int2T]    = None,
        padding          : Optional[Padding2T] = 0,
        ceil_mode        : bool                = False,
        count_include_pad: bool                = True
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad
        )

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x = pad_same(x, self.kernel_size, self.stride)
        return F.avg_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode,
            self.count_include_pad
        )


@POOL_LAYERS.register(name="blur_pool2d")
class BlurPool2d(nn.Module):
    """Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details. Corresponds to the
    Downsample class, which does blurring and subsampling
    
    Args:
        channels (int):
            Number of input channels
        filter_size (int):
            Binomial filter size for blurring. currently supports 3 and 5.
            Default: `3`.
        stride (int):
            Downsampling filter stride.
            
    Returns:
        input (Tensor):
            Transformed image.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, channels: int, filter_size: int = 3, stride: int = 2):
        super(BlurPool2d, self).__init__()
        if filter_size <= 1:
            raise ValueError()
        
        self.channels    = channels
        self.filter_size = filter_size
        self.stride      = stride
        self.padding     = [get_padding(filter_size, stride, dilation=1)] * 4
        
        poly1d = np.poly1d((0.5, 0.5))
        coeffs = torch.tensor(
            (poly1d ** (self.filt_size - 1)).coeffs.astype(np.float32)
        )
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :].repeat(self.channels, 1, 1, 1)
        self.register_buffer("filter", blur_filter, persistent=False)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.padding, "reflect")
        return F.conv2d(x, self.filt, stride=self.stride, groups=x.shape[1])


@POOL_LAYERS.register(name="fast_adaptive_avg_pool2d")
class FastAdaptiveAvgPool2d(nn.Module):

    # MARK: Magic Functions

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.flatten = flatten

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        return x.mean((2, 3), keepdim=not self.flatten)
    

@POOL_LAYERS.register(name="max_pool2d_same")
class MaxPool2dSame(nn.MaxPool2d):
    """Tensorflow like `same` wrapper for 2D max pooling."""

    # MARK: Magic Functions

    def __init__(
        self,
        kernel_size: Int2T,
        stride     : Optional[Int2T]    = None,
        padding    : Optional[Padding2T] = (0, 0),
        dilation   : Int2T              = (1, 1),
        ceil_mode  : bool                = False
    ):
        kernel_size = to_2tuple(kernel_size)
        stride      = to_2tuple(stride)
        dilation    = to_2tuple(dilation)
        super().__init__(kernel_size, stride, padding, dilation, ceil_mode)

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x = pad_same(x, self.kernel_size, self.stride, value=-float("inf"))
        return F.max_pool2d(
            x, self.kernel_size, self.stride, self.padding, self.dilation,
            self.ceil_mode
        )


@POOL_LAYERS.register(name="median_pool2d")
class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Attributes:
         kernel_size (Int2T):
            Size of pooling kernel.
         stride (Int2T):
            Pool stride, int or 2-tuple
         padding (Size4T, str, optional):
            Pool padding, int or 4-tuple (ll, r, t, b) as in pytorch F.pad.
         same (bool):
            Override padding and enforce same padding. Default: `False`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        kernel_size: Int2T,
        stride     : Int2T    			 = (1, 1),
        padding    : Optional[Padding4T] = 0,
        same	   : bool				 = False
    ):
        super().__init__()
        self.kernel_size = to_2tuple(kernel_size)
        self.stride 	 = to_2tuple(stride)
        self.padding 	 = to_4tuple(padding)  # convert to ll, r, t, b
        self.same	 	 = same

    # MARK: Configure

    def _padding(self, x: Tensor):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.kernel_size[0] - self.stride[0], 0)
            else:
                ph = max(self.kernel_size[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.kernel_size[1] - self.stride[1], 0)
            else:
                pw = max(self.kernel_size[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0])
        x = x.unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


@POOL_LAYERS.register(name="select_adaptive_pool2d")
class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size."""

    # MARK: Magic Functions

    def __init__(
        self,
        output_size: int  = 1,
        pool_type  : str  = "fast",
        flatten    : bool = False
    ):
        super().__init__()
        # Convert other falsy values to empty string for consistent TS typing
        self.pool_type = pool_type or ""
        self.flatten   = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == "":
            self.pool = nn.Identity()  # pass through
        elif pool_type == "fast":
            if output_size != 1:
                raise ValueError()
            self.pool    = FastAdaptiveAvgPool2d(flatten)
            self.flatten = nn.Identity()
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == "avg_max":
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == "cat_avg_max":
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        elif True:
            raise ValueError("Invalid pool type: %s" % pool_type)

    def __repr__(self):
        return (self.__class__.__name__ + " (pool_type=" + self.pool_type +
                ", flatten=" + str(self.flatten) + ")")

    # MARK: Properties

    def is_identity(self) -> bool:
        return not self.pool_type

    # MARK: Forward Pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        if self.pool_type == "cat_avg_max":
            return 2
        else:
            return 1
        

@POOL_LAYERS.register(name="spatial_pyramid_pool")
class SpatialPyramidPooling(nn.Module):
    """Spatial Pyramid Pooling layer used in YOLOv3-SPP.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        kernel_size (tuple):
            Sizes of several convolving kernels. Default: `(5, 9, 13)`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : tuple = (5, 9, 13),
    ):
        super().__init__()
        hidden_channels = in_channels // 2  # Hidden channels
        in_channels2    = hidden_channels * (len(kernel_size) + 1)

        self.conv1 = ConvBnMish(in_channels,  hidden_channels, kernel_size=1, stride=1)
        self.conv2 = ConvBnMish(in_channels2, out_channels,    kernel_size=1, stride=1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=input, stride=1, padding=input // 2)
             for input in kernel_size]
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x    = self.conv1(x)
        pred = self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))
        return pred


@POOL_LAYERS.register(name="spatial_pyramid_pooling_csp")
class SpatialPyramidPoolingCSP(nn.Module):
    """Cross Stage Partial Spatial Pyramid Pooling layer used in YOLOv3-SPP.
    
    Args:
        in_channels (int):
            Number of channels in the input image.
        out_channels (int):
            Number of channels produced by the convolution.
        number (int):
            Number of bottleneck layers to use.
        shortcut (bool):
            Use shortcut connection?. Default: `True`.
        groups (int):
            Default: `1`.
        expansion (float):
            Default: `0.5`.
        kernel_size (tuple):
            Sizes of several convolving kernels. Default: `(5, 9, 13)`.
    """
    
    # MARK: Magic Functions
    
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
        self.conv1 = ConvBnMish(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.conv3 = ConvBnMish(hidden_channels, hidden_channels, kernel_size=3, stride=1)
        self.conv4 = ConvBnMish(hidden_channels, hidden_channels, kernel_size=1, stride=1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=input, stride=(1, 1), padding=input // 2)
             for input in kernel_size]
        )
        self.conv5 = ConvBnMish(4 * hidden_channels, hidden_channels, kernel_size=1, stride=1)
        self.conv6 = ConvBnMish(hidden_channels, hidden_channels, kernel_size=3, stride=1)
        self.bn    = nn.BatchNorm2d(2 * hidden_channels)
        self.act   = Mish()
        self.conv7 = ConvBnMish(2 * hidden_channels, hidden_channels, kernel_size=1, stride=1)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        x1   = self.conv4(self.conv3(self.conv1(x)))
        y1   = self.conv6(self.conv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2   = self.conv2(x)
        pred = self.conv7(self.act(self.bn(torch.cat((y1, y2), dim=1))))
        return pred


# MARK: - Alias

AdaptiveAvgMaxPool    = AdaptiveAvgMaxPool2d
AdaptiveCatAvgMaxPool = AdaptiveCatAvgMaxPool2d
AvgPoolSame           = AvgPool2dSame
BlurPool              = BlurPool2d
FastAdaptiveAvgPool   = FastAdaptiveAvgPool2d
MaxPoolSame           = MaxPool2dSame
MedianPool            = MedianPool2d
SelectAdaptivePool    = SelectAdaptivePool2d
SPP                   = SpatialPyramidPooling
SPPCSP                = SpatialPyramidPoolingCSP


# MARK: - Register
POOL_LAYERS.register(name="adaptive_avg_pool",         module=nn.AdaptiveAvgPool2d)
POOL_LAYERS.register(name="adaptive_avg_pool1d",       module=nn.AdaptiveAvgPool1d)
POOL_LAYERS.register(name="adaptive_avg_pool2d",       module=nn.AdaptiveAvgPool2d)
POOL_LAYERS.register(name="adaptive_avg_pool3d",       module=nn.AdaptiveAvgPool3d)
POOL_LAYERS.register(name="adaptive_avg_max_pool",     module=AdaptiveAvgMaxPool)
POOL_LAYERS.register(name="adaptive_cat_avg_max_pool", module=AdaptiveCatAvgMaxPool)
POOL_LAYERS.register(name="adaptive_max_pool",         module=nn.AdaptiveMaxPool2d)
POOL_LAYERS.register(name="adaptive_max_pool1d",       module=nn.AdaptiveMaxPool1d)
POOL_LAYERS.register(name="adaptive_max_pool2d",       module=nn.AdaptiveMaxPool2d)
POOL_LAYERS.register(name="adaptive_max_pool3d",       module=nn.AdaptiveMaxPool3d)
POOL_LAYERS.register(name="avg_pool",		           module=nn.AvgPool2d)
POOL_LAYERS.register(name="avg_pool1d",		           module=nn.AvgPool1d)
POOL_LAYERS.register(name="avg_pool2d",		           module=nn.AvgPool2d)
POOL_LAYERS.register(name="avg_pool3d", 		       module=nn.AvgPool3d)
POOL_LAYERS.register(name="avg_pool_same",             module=AvgPoolSame)
POOL_LAYERS.register(name="blur_pool",                 module=BlurPool)
POOL_LAYERS.register(name="fast_adaptive_avg_pool",    module=FastAdaptiveAvgPool)
POOL_LAYERS.register(name="fractional_max_pool2d",     module=nn.FractionalMaxPool2d)
POOL_LAYERS.register(name="fractional_max_pool3d",     module=nn.FractionalMaxPool3d)
POOL_LAYERS.register(name="lp_pool1d", 			       module=nn.LPPool1d)
POOL_LAYERS.register(name="lp_pool2d", 			       module=nn.LPPool2d)
POOL_LAYERS.register(name="max_pool", 			       module=nn.MaxPool2d)
POOL_LAYERS.register(name="max_pool1d", 		       module=nn.MaxPool1d)
POOL_LAYERS.register(name="max_pool2d", 		       module=nn.MaxPool2d)
POOL_LAYERS.register(name="max_pool3d", 		       module=nn.MaxPool3d)
POOL_LAYERS.register(name="max_pool_same",             module=MaxPoolSame)
POOL_LAYERS.register(name="max_unpool", 		       module=nn.MaxUnpool2d)
POOL_LAYERS.register(name="max_unpool1d", 		       module=nn.MaxUnpool1d)
POOL_LAYERS.register(name="max_unpool2d", 		       module=nn.MaxUnpool2d)
POOL_LAYERS.register(name="max_unpool3d", 		       module=nn.MaxUnpool3d)
POOL_LAYERS.register(name="median_pool",               module=MedianPool)
POOL_LAYERS.register(name="select_adaptive_pool",      module=SelectAdaptivePool)
POOL_LAYERS.register(name="spp",                       module=SPP)
POOL_LAYERS.register(name="spp_csp",                   module=SPPCSP)


# MARK: - Builder

def create_pool2d(
    pool_type  : str,
    kernel_size: Int2T,
    stride	   : Optional[Int2T] = None,
    **kwargs
):
    stride              = stride or kernel_size
    padding             = kwargs.pop("padding", "")
    padding, is_dynamic = get_padding_value(
		padding, kernel_size, stride=stride, **kwargs
	)

    if is_dynamic:
        if pool_type == "avg":
            return AvgPool2dSame(kernel_size, stride, **kwargs)
        elif pool_type == "max":
            return MaxPool2dSame(kernel_size, stride, **kwargs)
        elif True:
            raise ValueError(f"Unsupported pool type {pool_type}")
    else:
        if pool_type == "avg":
            return nn.AvgPool2d(kernel_size, stride, padding, **kwargs)
        elif pool_type == "max":
            return nn.MaxPool2d(kernel_size, stride, padding, **kwargs)
        elif True:
            raise ValueError(f"Unsupported pool type {pool_type}")
