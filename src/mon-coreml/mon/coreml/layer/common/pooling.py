#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements pooling layers."""

from __future__ import annotations

__all__ = [
    "adaptive_avg_max_pool2d", "adaptive_cat_avg_max_pool2d", "adaptive_pool2d",
    "AdaptiveAvgMaxPool2d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d", "AdaptiveCatAvgMaxPool2d", "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AdaptivePool2d",
    "avg_pool2d_same", "AvgPool1d", "AvgPool2d", "AvgPool2dSame", "AvgPool3d",
    "ChannelPool", "FastAdaptiveAvgPool2d", "FractionalMaxPool2d",
    "FractionalMaxPool3d", "LPPool1d", "LPPool2d", "lse_pool2d",
    "max_pool2d_same", "MaxPool1d", "MaxPool2d", "MaxPool2dSame", "MaxPool3d",
    "MaxUnpool1d", "MaxUnpool2d", "MaxUnpool3d", "MedianPool2d",

]

import torch
from torch import nn
from torch.nn import functional

from mon import foundation
from mon.coreml import constant
from mon.coreml.layer import base
from mon.coreml.layer.common import padding as pad
from mon.coreml.typing import Int2T


# region Adaptive Pool

def adaptive_avg_max_pool2d(
    input      : torch.Tensor,
    output_size: int = 1,
    *args, **kwargs
) -> torch.Tensor:
    x     = input
    x_avg = functional.adaptive_avg_pool2d(x, output_size)
    x_max = functional.adaptive_max_pool2d(x, output_size)
    y     = 0.5 * (x_avg + x_max)
    return y


def adaptive_cat_avg_max_pool2d(
    input      : torch.Tensor,
    output_size: int = 1,
    *args, **kwargs
) -> torch.Tensor:
    x     = input
    x_avg = functional.adaptive_avg_pool2d(x, output_size)
    x_max = functional.adaptive_max_pool2d(x, output_size)
    y     = torch.cat((x_avg, x_max), 1)
    return y


def adaptive_pool2d(
    input      : torch.Tensor,
    pool_type  : str = "avg",
    output_size: int = 1,
    *args, **kwargs
) -> torch.Tensor:
    """Selectable global pooling function with dynamic input kernel size."""
    x = input
    if pool_type == "avg":
        x = functional.adaptive_avg_pool2d(x, output_size)
    elif pool_type == "avg_max":
        x = adaptive_avg_max_pool2d(x, output_size)
    elif pool_type == "cat_avg_max":
        x = adaptive_cat_avg_max_pool2d(x, output_size)
    elif pool_type == "max":
        x = functional.adaptive_max_pool2d(x, output_size)
    elif True:
        raise ValueError("Invalid pool type: %s" % pool_type)
    y = x
    return y


@constant.LAYER.register()
class AdaptiveAvgMaxPool2d(base.PassThroughLayerParsingMixin, nn.Module):

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = adaptive_avg_max_pool2d(
            input       = x,
            output_size = self.output_size
        )
        return y


@constant.LAYER.register()
class AdaptiveAvgPool1d(base.PassThroughLayerParsingMixin, nn.AdaptiveAvgPool1d):
    pass


@constant.LAYER.register()
class AdaptiveAvgPool2d(base.PassThroughLayerParsingMixin, nn.AdaptiveAvgPool2d):
    pass


@constant.LAYER.register()
class AdaptiveAvgPool3d(base.PassThroughLayerParsingMixin, nn.AdaptiveAvgPool3d):
    pass


@constant.LAYER.register()
class AdaptiveCatAvgMaxPool2d(base.PassThroughLayerParsingMixin, nn.Module):

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = adaptive_cat_avg_max_pool2d(
            input       = x,
            output_size = self.output_size
        )
        return y


@constant.LAYER.register()
class AdaptiveMaxPool1d(base.PassThroughLayerParsingMixin, nn.AdaptiveMaxPool1d):
    pass


@constant.LAYER.register()
class AdaptiveMaxPool2d(base.PassThroughLayerParsingMixin, nn.AdaptiveMaxPool2d):
    pass


@constant.LAYER.register()
class AdaptiveMaxPool3d(base.PassThroughLayerParsingMixin, nn.AdaptiveMaxPool3d):
    pass


@constant.LAYER.register()
class AdaptivePool2d(base.PassThroughLayerParsingMixin, nn.Module):
    """Selectable global pooling layer with dynamic input kernel size."""

    def __init__(
        self,
        output_size: int  = 1,
        pool_type  : str  = "fast",
        flatten    : bool = False,
        *args, **kwargs
    ):
        from mon.coreml.layer.common import linear, mutating
        
        super().__init__()
        self.pool_type = pool_type or ""
        
        self.flatten   = mutating.Flatten(1) \
            if flatten else linear.Identity()
        if pool_type == "":
            self.pool = linear.Identity()  # pass through
        elif pool_type == "fast":
            if output_size != 1:
                raise ValueError()
            self.pool    = FastAdaptiveAvgPool2d(flatten)
            self.flatten = linear.Identity()
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.pool(x)
        y = self.flatten(y)
        return y

    def feat_mult(self):
        if self.pool_type == "cat_avg_max":
            return 2
        else:
            return 1


@constant.LAYER.register()
class FastAdaptiveAvgPool2d(base.PassThroughLayerParsingMixin, nn.Module):

    def __init__(self, flatten: bool = False):
        super().__init__()
        self.flatten = flatten

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = x.mean((2, 3), keepdim=not self.flatten)
        return y

# endregion


# region Average Pool

def avg_pool2d_same(
    input            : torch.Tensor,
    kernel_size      : Int2T,
    stride           : Int2T,
    padding          : Int2T = 0,
    ceil_mode        : bool  = False,
    count_include_pad: bool  = True,
    *args, **kwargs
) -> torch.Tensor:
    x = input
    y = pad.pad_same(
        input       = x,
        kernel_size = kernel_size,
        stride      = stride,
    )
    y = functional.avg_pool2d(
        input             = y,
        kernel_size       = kernel_size,
        stride            = stride,
        padding           = padding,
        ceil_mode         = ceil_mode,
        count_include_pad = count_include_pad,
    )
    return y


@constant.LAYER.register()
class AvgPool2dSame(base.PassThroughLayerParsingMixin, nn.AvgPool2d):
    """Tensorflow like 'same' wrapper for 2D average pooling."""

    def __init__(
        self,
        kernel_size      : Int2T,
        stride           : Int2T | None = None,
        padding          : Int2T        = 0,
        ceil_mode        : bool         = False,
        count_include_pad: bool         = True,
    ):
        kernel_size = foundation.to_2tuple(kernel_size)
        stride      = foundation.to_2tuple(stride)
        super().__init__(
            kernel_size       = kernel_size,
            stride            = stride,
            padding           = padding,
            ceil_mode         = ceil_mode,
            count_include_pad = count_include_pad
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = pad.pad_same(
            input       = x,
            kernel_size = self.kernel_size,
            stride      = self.stride
        )
        y = functional.avg_pool2d(
            input             = y,
            kernel_size       = self.kernel_size,
            stride            = self.stride,
            padding           = self.padding,
            ceil_mode         = self.ceil_mode,
            count_include_pad = self.count_include_pad
        )
        return y


@constant.LAYER.register()
class AvgPool1d(base.PassThroughLayerParsingMixin, nn.AvgPool1d):
    pass


@constant.LAYER.register()
class AvgPool2d(base.PassThroughLayerParsingMixin, nn.AvgPool2d):
    pass


@constant.LAYER.register()
class AvgPool3d(base.PassThroughLayerParsingMixin, nn.AvgPool3d):
    pass

# endregion


# region Channel Pool

@constant.LAYER.register()
class ChannelPool(base.PassThroughLayerParsingMixin, nn.Module):
    """Global Channel Pool used in CBAM Module proposed by the paper: "CBAM:
    Convolutional Block Attention Module". """
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.cat(
            tensors = (
                torch.max(x, 1)[0].unsqueeze(1),
                torch.mean(x, 1).unsqueeze(1)
            ),
            dim = 1,
        )
        return y

# endregion


# region LP Pool

@constant.LAYER.register()
class LPPool1d(base.PassThroughLayerParsingMixin, nn.LPPool1d):
    pass


@constant.LAYER.register()
class LPPool2d(base.PassThroughLayerParsingMixin, nn.LPPool2d):
    pass

# endregion


# region LSE Pool

def lse_pool2d(input: torch.Tensor) -> torch.Tensor:
    """The LogSumExp (LSE) Pool (also called RealSoftMax or multivariable
    softplus) function. It is defined as the logarithm of the sum of the
    exponential.
    """
    x        = input
    x_flat   = x.view(x.size(0), x.size(1), -1)
    x_max, _ = torch.max(x_flat, dim=2, keepdim=True)
    y        = x_flat - x_max
    y        = x_max + y.exp().sum(dim=2, keepdim=True).log()
    return y

# endregion


# region Max Pool

def max_pool2d_same(
    input      : torch.Tensor,
    kernel_size: Int2T,
    stride     : Int2T,
    padding    : Int2T = 0,
    dilation   : Int2T = 1,
    ceil_mode  : bool  = False,
    *args, **kwargs
) -> torch.Tensor:
    x = input
    y = pad.pad_same(
        input       = x,
        kernel_size = kernel_size,
        stride      = stride,
        value       = -float("inf"),
    )
    y = functional.max_pool2d(
        input       = y,
        kernel_size = kernel_size,
        stride      = stride,
        padding     = padding,
        dilation    = dilation,
        ceil_mode   = ceil_mode,
    )
    return y


@constant.LAYER.register()
class FractionalMaxPool2d(base.PassThroughLayerParsingMixin, nn.FractionalMaxPool2d):
    pass


@constant.LAYER.register()
class FractionalMaxPool3d(base.PassThroughLayerParsingMixin, nn.FractionalMaxPool3d):
    pass


@constant.LAYER.register()
class MaxPool1d(base.PassThroughLayerParsingMixin, nn.MaxPool1d):
    pass


@constant.LAYER.register()
class MaxPool2d(base.PassThroughLayerParsingMixin, nn.MaxPool2d):
    pass


@constant.LAYER.register()
class MaxPool2dSame(base.PassThroughLayerParsingMixin, nn.MaxPool2d):
    """Tensorflow like `same` wrapper for 2D max pooling."""

    def __init__(
        self,
        kernel_size: Int2T,
        stride     : Int2T | None = None,
        padding    : Int2T | None = (0, 0),
        dilation   : Int2T        = (1, 1),
        ceil_mode  : bool         = False,
    ):
        kernel_size = foundation.to_2tuple(kernel_size)
        stride      = foundation.to_2tuple(stride)
        dilation    = foundation.to_2tuple(dilation)
        super().__init__(
            kernel_size = kernel_size,
            stride      = stride,
            padding     = padding,
            dilation    = dilation,
            ceil_mode   = ceil_mode
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = pad.pad_same(
            input       = x,
            kernel_size = self.kernel_size,
            stride      = self.stride,
            value       = -float("inf")
        )
        y = functional.max_pool2d(
            input             = y,
            kernel_size       = self.kernel_size,
            stride            = self.stride,
            padding           = self.padding,
            ceil_mode         = self.dilation,
            count_include_pad = self.ceil_mode
        )
        return y


@constant.LAYER.register()
class MaxPool3d(base.PassThroughLayerParsingMixin, nn.MaxPool3d):
    pass


@constant.LAYER.register()
class MaxUnpool1d(base.PassThroughLayerParsingMixin, nn.MaxUnpool1d):
    pass


@constant.LAYER.register()
class MaxUnpool2d(base.PassThroughLayerParsingMixin, nn.MaxUnpool2d):
    pass


@constant.LAYER.register()
class MaxUnpool3d(base.PassThroughLayerParsingMixin, nn.MaxUnpool3d):
    pass

# endregion


# region Median Pool

@constant.LAYER.register()
class MedianPool2d(base.PassThroughLayerParsingMixin, nn.Module):
    """Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size : Size of pooling kernel.
         stride: Pool stride, int or 2-tuple
         padding: Pool padding, int or 4-tuple (ll, r, t, b) as in pytorch F.pad.
         same: Override padding and enforce same padding. Defaults to False.
    """

    def __init__(
        self,
        kernel_size: Int2T,
        stride     : Int2T       = (1, 1),
        padding    : Int2T | str = 0,
        same	   : bool	     = False,
        *args, **kwargs
    ):
        super().__init__()
        self.kernel_size = foundation.to_2tuple(kernel_size)
        self.stride 	 = foundation.to_2tuple(stride)
        self.padding 	 = foundation.to_4tuple(padding)  # convert to ll, r, t, b
        self.same	 	 = same

    def _padding(self, input: torch.Tensor):
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = functional.pad(x, self._padding(x), mode="reflect")
        y = y.unfold(2, self.k[0], self.stride[0])
        y = y.unfold(3, self.k[1], self.stride[1])
        y = y.contiguous().view(y.size()[:4] + (-1,)).median(dim=-1)[0]
        return y
    
# endregion
