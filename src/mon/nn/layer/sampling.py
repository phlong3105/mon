#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements upsampling and downsampling layers."""

from __future__ import annotations

__all__ = [
    "CustomDownsample",
    "CustomUpsample",
    "Downsample",
    "DownsampleConv2d",
    "Interpolate",
    "Scale",
    "Upsample",
    "UpsampleConv2d",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

import math
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.upsampling import *

from mon.core import _ratio_2_t, _size_2_t


# region Downsampling

class Downsample(nn.Module):
    """Downsample a given multi-channel 1D (temporal), 2D (spatial) or 3D
    (volumetric) data.

    The input data is assumed to be of the form `minibatch x channels x
    [optional depth] x [optional height] x width`. Hence, for spatial inputs, we
    expect a 4D Tensor and for volumetric inputs, we expect a 5D Tensor.

    The algorithms available for upsampling are the nearest neighbor and linear,
    bilinear, bicubic and trilinear for 3D, 4D and 5D input tensor,
    respectively.

    One can either give a :attr:`scale_factor` or the target output :attr:`size`
    to calculate the output size. (You cannot give both, as it is ambiguous)

    Args:
        size: Output spatial sizes
        scale_factor: Multiplier for spatial size. Has to match input size if
            it is a tuple.
        mode: The upsampling algorithm. One of: ``'nearest'``, ``'linear'``,
            ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``. Default:
            ``'nearest'``.
        align_corners: If ``True``, the corner pixels of the input and output
            tensors are aligned, and thus preserving the values of those pixels.
            This only has effect when :param:`mode` is ``'linear'``,
            ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``. Default:
            ``False``.
        recompute_scale_factor: Recompute the :param:`scale_factor` for use in
            the interpolation calculation.
            - If ``True``, then :param:`scale_factor` must be passed in and
                :param:`scale_factor` is used to compute the output
                :param:`size`. The computed output :param:`size` will be used
                to infer new scales for the interpolation. Note that when
                :param:`scale_factor` is floating-point, it may differ from the
                recomputed :param:`scale_factor` due to rounding and precision
                issues.
            - If ``False``, then :param:`size` or :param:`scale_factor` will be
                used directly for interpolation.
            - Default: ``False``.
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
        if self.scale_factorImageDataset \
            and isinstance(self.scale_factor, tuple) \
            and (self.scale_factor == 1.0 or all(s == 1.0 for s in self.scale_factor)):
            return x
        y = F.interpolate(
            input                  = x,
            size                   = self.size,
            scale_factor           = self.scale_factor,
            mode                   = self.mode,
            align_corners          = self.align_corners,
            recompute_scale_factor = self.recompute_scale_factor
        )
        return y


class DownsampleConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        )
        self.in_channels  = in_channels
        self.out_channels = out_channels
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        b, l, c = x.shape
        h = int(math.sqrt(l))
        w = int(math.sqrt(l))
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return x
    
    def flops(self, h: int, w: int) -> int:
        flops = 0
        # conv
        flops += h / 2 * w / 2 * self.in_channels * self.out_channels * 4 * 4
        # print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


class CustomDownsample(nn.Module):
    """

    References:
        `<http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf>`__
    """

    def __init__(
        self,
        in_channels  : int,
        scale_factor : int,
        kernel_type  : str | Literal["box", "gauss", "gauss12", "gauss1sq2", "lanczos", "lanczos2", "lanczos3"],
        phase        : float        = 0,
        kernel_width : int   | None = None,
        support      : int   | None = None,
        sigma        : float | None = None,
        preserve_size: bool         = False,
    ):
        super().__init__()
        assert phase in [0, 0.5], "``phase`` should be 0 or 0.5"

        if kernel_type == "lanczos2":
            support      = 2
            kernel_width = 4 * scale_factor + 1
            kernel_type_ = "lanczos"
        elif kernel_type == "lanczos3":
            support      = 3
            kernel_width = 6 * scale_factor + 1
            kernel_type_ = "lanczos"
        elif kernel_type == "gauss12":
            kernel_width = 7
            sigma        = 1 / 2
            kernel_type_ = "gauss"
        elif kernel_type == "gauss1sq2":
            kernel_width = 9
            sigma        = 1.0 / np.sqrt(2)
            kernel_type_ = "gauss"
        elif kernel_type in ["lanczos", "gauss", "box"]:
            kernel_type_ = kernel_type
        else:
            assert False, "Wrong name kernel"

        # note that `kernel width` will be different to actual size for ``phase`` = 1/2
        self.kernel = self.get_kernel(
            scale_factor = scale_factor,
            kernel_type  = kernel_type_,
            phase        = phase,
            kernel_width = kernel_width,
            support      = support,
            sigma        = sigma
        )

        downsampler = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = self.kernel.shape,
            stride       = scale_factor,
            padding      = 0,
        )
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:]   = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(in_channels):
            downsampler.weight.data[i, i] = kernel_torch

        self.downsampler_ = downsampler

        if preserve_size:
            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.0)
            else:
                pad = int((self.kernel.shape[0] - scale_factor) / 2.0)
            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.preserve_size:
            x = self.padding(input)
        else:
            x = input
        self.x = x
        return self.downsampler_(x)

    @staticmethod
    def get_kernel(
        scale_factor : int,
        kernel_type  : str   | Literal["box", "gauss", "gauss12", "gauss1sq2", "lanczos", "lanczos2", "lanczos3"],
        phase        : float        = 0,
        kernel_width : int   | None = None,
        support      : int   | None = None,
        sigma        : float | None = None,
    ):
        assert kernel_type in ["lanczos", "gauss", "box"]

        # scale_factor = float(scale_factor)
        if phase == 0.5 and kernel_type != "box":
            kernel = np.zeros([kernel_width - 1, kernel_width - 1])
        else:
            kernel = np.zeros([kernel_width, kernel_width])

        if kernel_type == "box":
            assert phase == 0.5, "Box filter is always half-phased"
            kernel[:] = 1.0 / (kernel_width * kernel_width)
        elif kernel_type == "gauss":
            assert sigma,        "``sigma`` is not specified."
            assert phase != 0.5, "``phase`` 1/2 for gauss not implemented."

            center   = (kernel_width + 1.0) / 2.0
            sigma_sq = sigma * sigma

            for i in range(1, kernel.shape[0] + 1):
                for j in range(1, kernel.shape[1] + 1):
                    di = (i - center) / 2.0
                    dj = (j - center) / 2.0
                    kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                    kernel[i - 1][j - 1] = kernel[i - 1][j - 1] / (2.0 * np.pi * sigma_sq)
        elif kernel_type == "lanczos":
            assert support, "``support`` is not specified"
            center = (kernel_width + 1) / 2.0

            for i in range(1, kernel.shape[0] + 1):
                for j in range(1, kernel.shape[1] + 1):
                    if phase == 0.5:
                        di = abs(i + 0.5 - center) / scale_factor
                        dj = abs(j + 0.5 - center) / scale_factor
                    else:
                        di = abs(i - center) / scale_factor
                        dj = abs(j - center) / scale_factor

                    pi_sq = np.pi * np.pi

                    val = 1
                    if di != 0:
                        val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                        val = val / (np.pi * np.pi * di * di)
                    if dj != 0:
                        val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                        val = val / (np.pi * np.pi * dj * dj)

                    kernel[i - 1][j - 1] = val
        else:
            assert False, "Wrong method name"

        kernel /= kernel.sum()
        return kernel

# endregion


# region Upsampling

class UpsampleConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        )
        self.in_channels  = in_channels
        self.out_channels = out_channels
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        b, l, c = x.shape
        h = int(math.sqrt(l))
        w = int(math.sqrt(l))
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        x = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return x
    
    def flops(self, h: int, w: int) -> int:
        flops = 0
        # conv
        flops += h * 2 * w * 2 * self.in_channels * self.out_channels * 2 * 2
        # print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


class CustomUpsample(nn.Module):

    def __init__(self, output_shape: list | tuple, scale_factor: int):
        super().__init__()
        assert output_shape[0] % scale_factor == 0
        assert output_shape[1] % scale_factor == 0
        seed = np.ones((1, 1, output_shape[0] // scale_factor, output_shape[1] // scale_factor)) * 0.5
        self.output_shape = output_shape
        self.sigmoid      = nn.Sigmoid()
        self.seed         = nn.Parameter(data=torch.cuda.FloatTensor(seed), requires_grad=True)

    def forward(self):
        return nn.functional.interpolate(self.sigmoid(self.seed), size=self.output_shape, mode="bilinear")

# endregion


# region Misc

class Scale(nn.Module):
    """A learnable scale parameter. This layer scales the input by a learnable
    factor. It multiplies a learnable scale parameter of shape :math:`(1,)` with
    input of any shape.
    
    Args:
        scale: Initial value of the scale factor. Default: ``1.0``.
    """
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = x + self.scale
        return y


class Interpolate(nn.Module):
    
    def __init__(self, size: _size_2_t):
        super().__init__()
        self.size = math.parse_hw(size)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = F.interpolate(input=x, size=self.size)
        return y

# endregion
