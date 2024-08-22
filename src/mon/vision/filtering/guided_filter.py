#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements guided filter."""

from __future__ import annotations

__all__ = [
    "ConvGuidedFilter",
    "DeepGuidedFilter",
    "DeepGuidedFilterAdvanced",
    "DeepGuidedFilterConvGF",
    "DeepGuidedFilterGuidedMapConvGF",
    "FastGuidedFilter",
    "GuidedFilter",
    "guided_filter",
]

import cv2
import numpy as np
import torch
from cv2 import ximgproc
from plum import dispatch
from torch.autograd import Variable

from mon import core, nn
from mon.nn import functional as F, init
from mon.vision import geometry
from mon.vision.filtering.box_filter import BoxFilter


# region Guided Filter

@dispatch
def guided_filter(
    image : torch.Tensor,
    guide : torch.Tensor,
    radius: int,
    eps   : float = 1e-8,
) -> torch.Tensor:
    """Perform guided filter using :module:`torch`.
    
    Args:
        image: An image in `[B, C, H, W]` format.
        guide: A guidance image with the same shape with :obj:`image`.
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``1e-8``.

    Returns:
        A filtered image.
    
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py>`__
    """
    x = image
    y = guide
    box_filter = BoxFilter(radius=radius)
    n_x, c_x, h_x, w_x = x.size()
    n_y, c_y, h_y, w_y = y.size()
    
    assert n_x == n_y
    assert c_x == 1 or c_x == c_y
    assert h_x == h_y and w_x == w_y
    assert h_x > 2 * radius + 1 and w_x > 2 * radius + 1
    
    # N
    N = box_filter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))
    # mean_x
    mean_x = box_filter(x) / N
    # mean_y
    mean_y = box_filter(y) / N
    # cov_xy
    cov_xy = box_filter(x * y) / N - mean_x * mean_y
    # var_x
    var_x  = box_filter(x * x) / N - mean_x * mean_x
    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x
    # mean_A; mean_b
    mean_A = box_filter(A) / N
    mean_b = box_filter(b) / N
    
    return mean_A * x + mean_b


@dispatch
def guided_filter(
    image : np.ndarray,
    guide : np.ndarray,
    radius: int,
    eps   : float = 1e-8,
) -> np.ndarray:
    """Perform guided filter using :module:`cv2`.

    Args:
        image: An image in `[H, W, C]` format.
        guide: A guidance image with the same shape with :obj:`input`.
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``1e-8``.
    
    Returns:
        A filtered image.
    
    References:
        - `<https://github.com/lisabug/guided-filter/blob/master/core/filter.py>`__
        - `<https://github.com/lisabug/guided-filter/tree/master>`__
        - `<https://github.com/wuhuikai/DeepGuidedFilter>`__
    """
    kernel_size = core.parse_hw(radius * 2 + 1)
    mean_i      = cv2.boxFilter(image,         cv2.CV_64F, kernel_size)
    mean_g      = cv2.boxFilter(guide,         cv2.CV_64F, kernel_size)
    mean_ig     = cv2.boxFilter(image * guide, cv2.CV_64F, kernel_size)
    cov_ig      = mean_ig - mean_i * mean_g
    mean_ii     = cv2.boxFilter(image * image, cv2.CV_64F, kernel_size)
    var_i       = mean_ii - mean_i * mean_i
    a           = cov_ig / (var_i + eps)
    b           = mean_i  - a * mean_i
    mean_a      = cv2.boxFilter(a, cv2.CV_64F, kernel_size)
    mean_b      = cv2.boxFilter(b, cv2.CV_64F, kernel_size)
    output      = mean_a * image + mean_b
    return output


@dispatch
def guided_filter(
    image : np.ndarray,
    guide : np.ndarray,
    radius: int,
    eps   : float = 1e-8,
    **kwargs
) -> np.ndarray:
    """Perform guided filter using :module:`cv2.ximgproc`.
    
    Args:
        image: An image in `[H, W, C]` format.
        guide: A guidance image with the same shape with :obj:`input`.
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``1e-8``.
    
    Returns:
        A filtered image.
    """
    return ximgproc.guidedFilter(guide=guide, src=image, radius=radius, eps=eps)


class GuidedFilter(nn.Module):
    """Guided Filter.
    
    Args:
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``1e-8``.
    
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py>`__
    """
    
    def __init__(self, radius: int, eps: float = 1e-8):
        super().__init__()
        self.radius     = radius
        self.eps        = eps
        self.box_filter = BoxFilter(self.radius)
    
    def forward(self, image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        x = image
        y = guide
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        
        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.radius + 1 and w_x > 2 * self.radius + 1
        
        # N
        N = self.box_filter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))
        # mean_x
        mean_x = self.box_filter(x) / N
        # mean_y
        mean_y = self.box_filter(y) / N
        # cov_xy
        cov_xy = self.box_filter(x * y) / N - mean_x * mean_y
        # var_x
        var_x  = self.box_filter(x * x) / N - mean_x * mean_x
        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x
        # mean_A; mean_b
        mean_A = self.box_filter(A) / N
        mean_b = self.box_filter(b) / N
       
        return mean_A * x + mean_b
    
# endregion


# region Fast Guided Filter

class FastGuidedFilter(nn.Module):
    """Fast Guided Filter.
    
    Args:
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``1e-8``.
        downscale: Downscale factor. Default: ``8``.
    
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py>`__
    """
    
    def __init__(self, radius: int, eps: float = 1e-8, downscale: int = 8):
        super().__init__()
        self.radius     = radius
        self.downscale  = downscale
        self.eps        = eps
        self.box_filter = BoxFilter(radius=self.radius)
    
    @dispatch
    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        n_xlr, c_xlr, h_xlr, w_xlr = x_lr.shape
        n_ylr, c_ylr, h_ylr, w_ylr = y_lr.shape
        n_xhr, c_xhr, h_xhr, w_xhr = x_hr.shape
        
        assert n_xlr == n_ylr and n_ylr == n_xhr
        assert c_xlr == c_xhr and (c_xlr == 1 or c_xlr == c_ylr)
        assert h_xlr == h_ylr and w_xlr == w_ylr
        assert h_xlr > 2 * self.radius + 1 and w_xlr > 2 * self.radius + 1
        
        # N
        N = self.box_filter(Variable(x_lr.data.new().resize_((1, 1, h_xlr, w_xlr)).fill_(1.0)))
        # mean_x
        mean_x = self.box_filter(x_lr) / N
        # mean_y
        mean_y = self.box_filter(y_lr) / N
        # cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        # var_x
        var_x  = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x
        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x
        # mean_A; mean_b
        mean_A = F.interpolate(A, (h_xhr, w_xhr), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_xhr, w_xhr), mode="bilinear", align_corners=True)
        
        return mean_A * x_hr + mean_b
    
    @dispatch
    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        _, _, x_h, x_w = x_lr.size()
        h    = x_h // self.downscale
        w    = x_w // self.downscale
        x_lr = geometry.resize(x_lr, (h, w), "bilinear")
        y_lr = geometry.resize(x_hr, (h, w), "bilinear")
        return self.forward(x_lr, y_lr, x_hr)

# endregion


# region Deep Guided Filter

class ConvGuidedFilter(nn.Module):
    """Convolutional Guided Filter.
    
    Args:
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        norm: Normalization layer. Default: :obj:`nn.BatchNorm2d`.
        downscale: Downscale factor. Default: ``8``.
        
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py>`__
    """
    
    def __init__(self, radius: int = 1, norm: nn.Module = nn.BatchNorm2d, downscale: int = 8):
        super().__init__()
        self.box_filter = nn.Conv2d(3, 3, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=3)
        self.conv_a     = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=False)
        )
        self.box_filter.weight.data[...] = 1.0
        self.downscale = downscale
    
    @dispatch
    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()
        
        N = self.box_filter(x_lr.data.new().resize_((1, 3, h_lrx, w_lrx)).fill_(1.0))
        # mean_x
        mean_x = self.box_filter(x_lr) / N
        # mean_y
        mean_y = self.box_filter(y_lr) / N
        # cov_xy
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        # var_x
        var_x  = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x
        # A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        # b
        b = mean_y - A * mean_x
        # mean_A; mean_b
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        
        return mean_A * x_hr + mean_b
        
    @dispatch
    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        _, _, x_h, x_w = x_lr.size()
        h    = x_h // self.downscale
        w    = x_w // self.downscale
        x_lr = geometry.resize(x_lr, (h, w), "bilinear")
        y_lr = geometry.resize(x_hr, (h, w), "bilinear")
        return self.forward(x_lr, y_lr, x_hr)


def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n_out, n_in, h, w = m.weight.data.size()
        # Last Layer
        if n_out < n_in:
            init.xavier_uniform_(m.weight.data)
            return
        # Except Last Layer
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data,   0.0)
        
        
def build_lr_net(
    in_channels : int       = 3,
    mid_channels: int       = 24,
    layers      : int       = 5,
    relu_slope  : float     = 0.2,
    norm        : nn.Module = nn.AdaptiveBatchNorm2d,
) -> nn.Sequential:
    """Build a low-resolution network.
    
    Args:
        in_channels: Number of input channels. Default: ``3``.
        mid_channels: Number of middle channels. Default: ``24``.
        layers: Number of layers. Default: ``5``.
        relu_slope: Slope of the LeakyReLU. Default: ``0.2``.
        norm: Normalization layer. Default: :obj:`nn.AdaptiveNorm2d`.
    """
    net = [
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(mid_channels),
        nn.LeakyReLU(relu_slope, inplace=True),
    ]
    for l in range(1, layers):
        net += [
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=2**l, dilation=2**l, bias=False),
            norm(mid_channels),
            nn.LeakyReLU(relu_slope, inplace=True)
        ]
    net += [
        nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
        norm(mid_channels),
        nn.LeakyReLU(relu_slope, inplace=True),
        nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1)
    ]
    net = nn.Sequential(*net)
    net.apply(weights_init_identity)
    return net


class DeepGuidedFilter(nn.Module):
    """Deep Guided Filter network.
    
    Args:
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``1e-8``.
        lr_channels: Number of channels for the low-resolution network.
            Default: ``24``.
        lr_layer: Number of layers for the low-resolution network. Default: ``5``.
        lr_relu_slope: Slope of the LeakyReLU for the low-resolution network.
            Default: ``0.2``.
        lr_norm: Normalization layer for the low-resolution network.
            Default: :obj:`nn.AdaptiveNorm2d`.
        
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py>`__
    """
    
    def __init__(
        self,
        radius       : int       = 1,
        eps          : float     = 1e-8,
        lr_channels  : int       = 24,
        lr_layers    : int       = 5,
        lr_relu_slope: float     = 0.2,
        lr_norm      : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        self.lr = build_lr_net(
            in_channels  = 3,
            mid_channels = lr_channels,
            layers       = lr_layers,
            relu_slope   = lr_relu_slope,
            norm         = lr_norm,
        )
        self.gf = FastGuidedFilter(radius=radius, eps=eps)
    
    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)
    
    def load_lr_weights(self, path: str | core.Path):
        self.lr.load_state_dict(torch.load(str(path)))


class DeepGuidedFilterAdvanced(DeepGuidedFilter):
    """Deep Guided Filter network.
    
    Args:
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``1e-4``.
        lr_channels: Number of middle channels for the low-resolution network.
            Default: ``24``.
        lr_layer: Number of layers for the low-resolution network. Default: ``5``.
        lr_relu_slope: Slope of the LeakyReLU for the low-resolution network.
            Default: ``0.2``.
        lr_norm: Normalization layer for the low-resolution network.
            Default: :obj:`nn.AdaptiveNorm2d`.
        
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py>`__
    """
    
    def __init__(
        self,
        radius         : int       = 1,
        eps            : float     = 1e-2,
        guided_channels: int       = 64,
        lr_channels    : int       = 24,
        lr_layers      : int       = 5,
        lr_relu_slope  : float     = 0.2,
        lr_norm        : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__(
            radius        = radius,
            eps           = eps,
            lr_channels   = lr_channels,
            lr_layers     = lr_layers,
            lr_relu_slope = lr_relu_slope,
            lr_norm       = lr_norm,
        )
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, guided_channels, 1, bias=False),
            nn.AdaptiveBatchNorm2d(guided_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(guided_channels, 3, 1)
        )
        self.guided_map.apply(weights_init_identity)

    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr))


class DeepGuidedFilterConvGF(nn.Module):
    """Deep Guided Filter network.
    
    Args:
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        lr_channels: Number of middle channels for the low-resolution network.
            Default: ``24``.
        lr_layer: Number of layers for the low-resolution network. Default: ``5``.
        lr_relu_slope: Slope of the LeakyReLU for the low-resolution network.
            Default: ``0.2``.
        lr_norm: Normalization layer for the low-resolution network.
            Default: :obj:`nn.AdaptiveNorm2d`.
        
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py>`__
    """
    
    def __init__(
        self,
        radius       : int       = 1,
        lr_channels  : int       = 24,
        lr_layers    : int       = 5,
        lr_relu_slope: float     = 0.2,
        lr_norm      : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__()
        self.lr = build_lr_net(
            in_channels  = 3,
            mid_channels = lr_channels,
            layers       = lr_layers,
            relu_slope   = lr_relu_slope,
            norm         = lr_norm,
        )
        self.gf = ConvGuidedFilter(radius=radius, norm=nn.AdaptiveBatchNorm2d)

    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)
    
    def init_lr(self, path: str | core.Path):
        self.lr.load_state_dict(torch.load(str(path)))


class DeepGuidedFilterGuidedMapConvGF(DeepGuidedFilterConvGF):
    
    def __init__(
        self,
        radius         : int       = 1,
        guided_dilation: int       = 0,
        guided_channels: int       = 64,
        lr_channels    : int       = 24,
        lr_layers      : int       = 5,
        lr_relu_slope  : float     = 0.2,
        lr_norm        : nn.Module = nn.AdaptiveBatchNorm2d,
    ):
        super().__init__(
            radius        = radius,
            lr_channels   = lr_channels,
            lr_layers     = lr_layers,
            lr_relu_slope = lr_relu_slope,
            lr_norm       = lr_norm,
        )
        self.guided_map = nn.Sequential(
            nn.Conv2d(3, guided_channels, 1, bias=False) if guided_dilation == 0 else nn.Conv2d(3, guided_channels, 5, padding=guided_dilation, dilation=guided_dilation, bias=False),
            nn.AdaptiveBatchNorm2d(guided_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(guided_channels, 3, 1)
        )

    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        return self.gf(self.guided_map(x_lr), self.lr(x_lr), self.guided_map(x_hr)).clamp(0, 1)
    
# endregion
