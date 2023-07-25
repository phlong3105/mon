#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image filtering functions.

Common layers are:
    - bilateral
    - blur
    - canny
    - dexined
    - filter
    - gaussian
    - guided
    - kernel
    - laplacian
    - median
    - motion
    - sobel
    - unsharp
"""

from __future__ import annotations

__all__ = [
    "AdaptiveUSM2d",
    "LoG2d",
    "USM2d",
    "USMBase2d",
    "log2d_kernel",
    "log2d_kernel_np",
]

import numpy as np
import torch
from torch import nn

from mon.coreml.layer.typing import _size_2_t
from mon.foundation import math
from mon.globals import LAYERS


# region Filter

@LAYERS.register()
class BoxFilter(nn.Module):
    """Box Filter layer from: "`DeepGuidedFilter
    <https://github.com/wuhuikai/DeepGuidedFilter>`__"
    
    Args:
        kernel_size: Blurring kernel size.
        radius: kernel_size = 2 * radius + 1.
    """
    
    def __init__(
        self,
        kernel_size: int | None = None,
        radius     : int | None = None,
    ):
        super().__init__()
        assert kernel_size is not None and radius is not None
        if kernel_size is not None:
            self.kernel_size = kernel_size
            self.radius      = int((kernel_size - 1) / 2)
        else:
            self.kernel_size = radius * 2 + 1
            self.radius      = radius
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4
        x = input
        return self.diff_y(self.diff_x(x.cumsum(dim=2)).cumsum(dim=3))
    
    def diff_x(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4
        x      = input
        r      = self.radius
        left   = x[:, :,         r:2 * r + 1]
        middle = x[:, :, 2 * r + 1:         ] - x[:, :,           :-2 * r - 1]
        right  = x[:, :,        -1:         ] - x[:, :, -2 * r - 1:    -r - 1]
        y      = torch.cat([left, middle, right], dim=2)
        return y
    
    def diff_y(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4
        x      = input
        r      = self.radius
        left   = x[:, :, :,         r:2 * r + 1]
        middle = x[:, :, :, 2 * r + 1:         ] - x[:, :, :,           :-2 * r - 1]
        right  = x[:, :, :,        -1:         ] - x[:, :, :, -2 * r - 1:    -r - 1]
        y      = torch.cat([left, middle, right], dim=3)
        return y

# endregion


# region Guided Filter

@LAYERS.register()
class GuidedFilter(nn.Module):
    """Guided Filter layer from: "`DeepGuidedFilter
    <https://github.com/wuhuikai/DeepGuidedFilter>`__"
    
    Args:
        kernel_size: Blurring kernel size.
        radius: kernel_size = 2 * radius + 1.
    """
    
    def __init__(
        self,
        kernel_size: int | None = None,
        radius     : int | None = None,
        eps        : float      = 1e-8,
    ):
        super().__init__()
        assert kernel_size is not None and radius is not None
        if kernel_size is not None:
            self.kernel_size = kernel_size
            self.radius      = int((kernel_size - 1) / 2)
        else:
            self.kernel_size = radius * 2 + 1
            self.radius      = radius
        self.eps        = eps
        self.box_filter = BoxFilter(kernel_size=kernel_size, radius=radius)
    
    def forward(self, input: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """
        
        Args:
            input: Input image.
            guide: Guidance image.

        Returns:
            y: Filtering output.
        """
        x = input
        y = guide
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # n
        n = self.boxfilter(torch.autograd.Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))
        # mean_x
        mean_x = self.boxfilter(x) / n
        # mean_y
        mean_y = self.boxfilter(y) / n
        # cov_xy
        cov_xy = self.boxfilter(x * y) / n - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / n - mean_x * mean_x
        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x
        # mean_A ; mean_b
        mean_A = self.boxfilter(A) / n
        mean_b = self.boxfilter(b) / n
        y      = mean_A * x + mean_b
        
        return y

# endregion


# region Unsharp Masking

def log2d_kernel_np(k: int, sigma: float) -> np.ndarray:
    """Get LoG kernel."""
    ax       = np.round(np.linspace(-np.floor(k / 2), np.floor(k / 2), k))
    x, y     = np.meshgrid(ax, ax)
    x2       = np.power(x, 2)
    y2       = np.power(y, 2)
    s2       = np.power(sigma, 2)
    s4       = np.power(sigma, 4)
    hg       = np.exp(-(x2 + y2) / (2. * s2))
    kernel_t = hg * (x2 + y2 - 2 * s2) / (s4 * np.sum(hg))
    kernel   = kernel_t - np.sum(kernel_t) / np.power(k, 2)
    return kernel


def log2d_kernel(k: int, sigma: torch.Tensor, cuda: bool = False) -> torch.Tensor:
    """Get LoG kernel."""
    if cuda:
        ax = torch.round(torch.linspace(-math.floor(k/2), math.floor(k/2), k), out=torch.FloatTensor())
        ax = ax.cuda()
    else:
        ax = torch.round(torch.linspace(-math.floor(k/2), math.floor(k/2), k), out=torch.FloatTensor())
    y        = ax.view(-1, 1).repeat(1, ax.size(0))
    x        = ax.view(1, -1).repeat(ax.size(0), 1)
    x2       = torch.pow(x, 2)
    y2       = torch.pow(y, 2)
    s2       = torch.pow(sigma, 2)
    s4       = torch.pow(sigma, 4)
    hg       = (-(x2 + y2) / (2. * s2)).exp()
    kernel_t = hg * (1.0 - (x2 + y2 / 2 * s2)) * (1.0 / s4 * hg.sum())
    if cuda:
        kernel = kernel_t - kernel_t.sum() / torch.pow(torch.FloatTensor([k]).cuda(),2)
    else:
        kernel = kernel_t - kernel_t.sum() / torch.pow(torch.FloatTensor([k]),2)
    return kernel


@LAYERS.register()
class LoG2d(nn.Module):
    """LogGd layer from: "`pytorch_usm
    <https://github.com/maeotaku/pytorch_usm/blob/master/LoG.py>`__".
    """
    
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        kernel_size  : _size_2_t,
        stride       : _size_2_t = 1,
        padding      : _size_2_t = 0,
        dilation     : _size_2_t = 1,
        fixed_coeff  : bool      = False,
        sigma        : float     = -1,
        cuda         : bool      = False,
        requires_grad: bool      = True,
    ):
        super().__init__()
        self.fixed_coeff   = fixed_coeff
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.kernel_size   = kernel_size
        self.stride        = stride
        self.padding       = padding
        self.dilation      = dilation
        self.cuda          = cuda
        self.requires_grad = requires_grad
        if not self.fixed_coeff:
            if self.cuda:
                self.sigma = nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=self.requires_grad)
            else:
                self.sigma = nn.Parameter(torch.FloatTensor(1), requires_grad=self.requires_grad)
        else:
            if self.cuda:
                self.sigma = torch.cuda.FloatTensor([sigma])
            else:
                self.sigma = torch.FloatTensor([sigma])
            self.kernel = log2d_kernel(self.kernel_size, self.sigma, self.cuda)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        self.init_weights()
    
    def init_weights(self):
        if not self.fixed_coeff:
            self.sigma.data.uniform_(0.0001, 0.9999)
            
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x          = input
        b, c, h, w = x.size()
        if not self.fixed_coeff:
            self.kernel = log2d_kernel(self.kernel_size, self.sigma, self.cuda)
            self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = self.kernel
        # kernel size is (out_channels, in_channels, h, w)
        output = nn.functional.conv2d(
            input    = x.view(b * self.in_channels, 1, h, w),
            weight   = kernel,
            stride   = self.stride,
            padding  = self.padding,
            dilation = self.dilation,
            groups   = 1,
        )
        output = output.view(b, self.in_channels, h, w)
        return output


class USMBase2d(nn.Module):
    """Unsharp masking layer proposed in the paper: "`Unsharp Masking Layer:
    Injecting Prior Knowledge in Convolutional Networks for Image Classification
    <https://link.springer.com/chapter/10.1007/978-3-030-30508-6_1>`__".
    
    References:
        - https://github.com/maeotaku/pytorch_usm/blob/master/USM.py
    """
    
    def __init__(
        self,
        in_channels  : int,
        kernel_size  : _size_2_t,
        stride       : _size_2_t = 1,
        dilation     : _size_2_t = 1,
        fixed_coeff  : bool      = False,
        sigma        : float     = -1,
        cuda         : bool      = False,
        requires_grad: bool      = True,
    ):
        super().__init__()
        # Padding must be forced so output size is = to input size
        # Thus, in_channels = out_channels
        padding = int(
            (
                stride * (in_channels - 1)
                + ((kernel_size - 1) * (dilation - 1))
                + kernel_size-in_channels
            ) / 2
        )
        self.log2d = LoG2d(
            in_channels   = in_channels,
            out_channels  = in_channels,
            kernel_size   = kernel_size,
            stride        = stride,
            padding       = padding,
            dilation      = dilation,
            fixed_coeff   = fixed_coeff,
            sigma         = sigma,
            cuda          = cuda,
            requires_grad = requires_grad,
        )
        self.alpha = None

    def init_weights(self):
        if self.requires_grad:
            super().init_weights()
            self.alpha.data.uniform_(0, 10)

    def assign_weight(self, alpha):
        if self.cuda:
            self.alpha = torch.cuda.FloatTensor([alpha])
        else:
            self.alpha = torch.FloatTensor([alpha])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x              = input
        brightness     = self.log2d(x)
        unsharp        = x + self.alpha * brightness
        max_brightness = torch.max(torch.abs(brightness))
        max_x          = torch.max(x)
        unsharp        = unsharp * max_x / max_brightness
        return unsharp


@LAYERS.register()
class USM2d(USMBase2d):
    """Unsharp masking layer proposed in the paper: "`Unsharp Masking Layer:
    Injecting Prior Knowledge in Convolutional Networks for Image Classification
    <https://link.springer.com/chapter/10.1007/978-3-030-30508-6_1>`__".
    
    References:
        - https://github.com/maeotaku/pytorch_usm/blob/master/USM.py
    """
    
    def __init__(
        self,
        in_channels  : int,
        kernel_size  : _size_2_t,
        stride       : _size_2_t = 1,
        dilation     : _size_2_t = 1,
        fixed_coeff  : bool      = False,
        sigma        : float     = -1,
        cuda         : bool      = False,
        requires_grad: bool      = True,
    ):
        super().__init__(
            in_channels   = in_channels,
            kernel_size   = kernel_size,
            stride        = stride,
            dilation      = dilation,
            fixed_coeff   = fixed_coeff,
            sigma         = sigma,
            cuda          = cuda,
            requires_grad = requires_grad,
        )
        if self.requires_grad:
            if self.cuda:
                self.alpha = nn.Parameter(torch.cuda.FloatTensor(1), requires_grad=self.requires_grad)
            else:
                self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=self.requires_grad)
            self.init_weights()


@LAYERS.register()
class AdaptiveUSM2d(USMBase2d):
    """Unsharp masking layer proposed in the paper: "`Unsharp Masking Layer:
    Injecting Prior Knowledge in Convolutional Networks for Image Classification
    <https://link.springer.com/chapter/10.1007/978-3-030-30508-6_1>`__".
    
    References:
        - https://github.com/maeotaku/pytorch_usm/blob/master/USM.py
    """
    
    def __init__(
        self,
        in_channels  : int,
        in_side      : int,
        kernel_size  : _size_2_t,
        stride       : _size_2_t = 1,
        dilation     : _size_2_t = 1,
        fixed_coeff  : bool      = False,
        sigma        : float     = -1,
        cuda         : bool      = False,
        requires_grad: bool      = True,
    ):
        super().__init__(
            in_channels   = in_channels,
            kernel_size   = kernel_size,
            stride        = stride,
            dilation      = dilation,
            fixed_coeff   = fixed_coeff,
            sigma         = sigma,
            cuda          = cuda,
            requires_grad = requires_grad,
        )
        if self.requires_grad:
            if self.cuda:
                self.alpha = nn.Parameter(torch.cuda.FloatTensor(in_side, in_side), requires_grad=self.requires_grad)
            else:
                self.alpha = nn.Parameter(torch.FloatTensor(in_side, in_side), requires_grad=self.requires_grad)
            self.init_weights()
            
# endregion


# region Main

def test_usm():
    # CPU
    x   = torch.randn(2, 3, 11, 11)
    usm = USM2d(in_channels=3, kernel_size=3)
    y   = usm(x)
    print(f"USM (CPU): {y}")
    print(y.size())
    
    # CUDA
    x    = torch.randn(2, 3, 11, 11).cuda()
    ausm = AdaptiveUSM2d(in_channels=3, in_side=11, kernel_size=3, cuda=True)
    y    = ausm(x)
    print(f"Adaptive USM (CUDA): {y}")
    print(y.size())


if __name__ == "__main__":
    test_usm()
    
# endregion
