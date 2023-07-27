#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements upsampling and downsampling layers."""

from __future__ import annotations

__all__ = [
    "Downsample", "Interpolate", "Scale", "Upsample", "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

import torch
from torch import nn
from torch.nn import functional

from mon.core import math
from mon.globals import LAYERS
from mon.nn.layer import base
from mon.nn.typing import _ratio_2_t, _size_2_t


# region Downsampling

@LAYERS.register()
class Downsample(base.PassThroughLayerParsingMixin, nn.Module):
    """Downsample a given multi-channel 1D (temporal), 2D (spatial) or 3D
    (volumetric) data.

    The input data is assumed to be of the form `minibatch x channels x
    [optional depth] x [optional height] x width`. Hence, for spatial inputs, we
    expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input Tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size`
    to calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size: Output spatial sizes
        scale_factor: Multiplier for spatial size. Has to match input size if
            it is a tuple.
        mode: The upsampling algorithm. One of ['nearest', 'linear', 'bilinear',
            'bicubic', 'trilinear']. Default: 'nearest'.
        align_corners: If True, the corner pixels of the input and output
            tensors are aligned, and thus preserving the values at those pixels.
            This only has effect when :param:`mode` is 'linear', 'bilinear',
            'bicubic', or 'trilinear'. Default: False.
        recompute_scale_factor: Recompute the scale_factor for use in the
            interpolation calculation.
            - If True, then :param:`scale_factor` must be passed in and
                :param:`scale_factor` is used to compute the output
                :param:`size`. The computed output :param:`size` will be used
                to infer new scales for the interpolation. Note that when
                :param:`scale_factor` is floating-point, it may differ from the
                recomputed :param:`scale_factor` due to rounding and precision
                issues.
            - If False, then :param:`size` or :param:`scale_factor` will be used
                directly for interpolation.
            Default: False.
    """
    
    def __init__(
        self,
        size                  : _size_2_t  | None = None,
        scale_factor          : _ratio_2_t | None = None,
        mode                  : str               = "nearest",
        align_corners         : bool              = False,
        recompute_scale_factor: bool              = False,
    ):
        super().__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(1.0 / factor) for factor in scale_factor)
        else:
            self.scale_factor = float(1.0 / scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.size and self.size == list(x[2:]):
            return x
        if self.scale_factor is not None \
            and (self.scale_factor == 1.0 or all(s == 1.0 for s in self.scale_factor)):
            return x
        y = functional.interpolate(
            input         = x,
            size          = self.size,
            scale_factor  = self.scale_factor,
            mode          = self.mode,
            align_corners = self.align_corners,
            recompute_scale_factor = self.recompute_scale_factor
        )
        return y


# endregion


# region Upsampling

@LAYERS.register()
class Upsample(base.PassThroughLayerParsingMixin, nn.Upsample):
    pass


@LAYERS.register()
class UpsamplingBilinear2d(base.PassThroughLayerParsingMixin, nn.UpsamplingBilinear2d):
    pass


@LAYERS.register()
class UpsamplingNearest2d(base.PassThroughLayerParsingMixin, nn.UpsamplingNearest2d):
    pass


# endregion


@LAYERS.register()
class Scale(base.PassThroughLayerParsingMixin, nn.Module):
    """A learnable scale parameter. This layer scales the input by a learnable
    factor. It multiplies a learnable scale parameter of shape (1,) with input
    of any shape.
    
    Args:
        scale: Initial value of the scale factor. Default: 1.0.
    """
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = x + self.scale
        return y


@LAYERS.register()
class Interpolate(base.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(self, size: _size_2_t):
        super().__init__()
        self.size = math.get_hw(size)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = functional.interpolate(input=x, size=self.size)
        return y
