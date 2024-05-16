#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements guided filter."""

from __future__ import annotations

__all__ = [
    "ConvGuidedFilter",
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
from mon.nn import functional as F
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
        image: An image in :math:`[B, C, H, W]` format.
        guide: A guidance image with the same shape with :attr:`image`.
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
        image: An image in :math:`[H, W, C]` format.
        guide: A guidance image with the same shape with :attr:`input`.
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
        image: An image in :math:`[H, W, C]` format.
        guide: A guidance image with the same shape with :attr:`input`.
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``1e-8``.
    
    Returns:
        A filtered image.
    """
    return ximgproc.guidedFilter(guide=guide, src=image, radius=radius, eps=eps)


class GuidedFilter(nn.Module):
    """Guided filter.
    
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
    """Fast guided filter.
    
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
    def forward(
        self,
        image_lr: torch.Tensor,
        guide_lr: torch.Tensor,
        image_hr: torch.Tensor,
    ) -> torch.Tensor:
        x_lr = image_lr
        y_lr = guide_lr
        x_hr = image_hr
        n_xlr, c_xlr, h_xlr, w_xlr = x_lr.size()
        n_ylr, c_ylr, h_ylr, w_ylr = y_lr.size()
        n_xhr, c_xhr, h_xhr, w_xhr = x_hr.size()
        
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
    def forward(self, image_hr: torch.Tensor, guide_hr: torch.Tensor) -> torch.Tensor:
        _, _, x_h, x_w = image_hr.size()
        h    = x_h // self.downscale
        w    = x_w // self.downscale
        x_lr = geometry.resize(image_hr, (h, w), "bilinear")
        y_lr = geometry.resize(guide_hr, (h, w), "bilinear")
        x_hr = image_hr
        return self.forward(x_lr, y_lr, x_hr)

# endregion


# region Conv Guided Filter

class ConvGuidedFilter(nn.Module):
    """Convolutional guided filter.
    
    Args:
        radius: Radius of filter a.k.a. kernel_size = radius * 2 + 1.
            Commonly be ``1``, ``2``, ``4``, or ``8``.
        norm: Normalization layer. Default: :class:`nn.BatchNorm2d`.
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
    def forward(
        self,
        image_lr: torch.Tensor,
        guide_lr: torch.Tensor,
        image_hr: torch.Tensor,
    ) -> torch.Tensor:
        x_lr = image_lr
        y_lr = guide_lr
        x_hr = image_hr
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
    def forward(self, image_hr: torch.Tensor, guide_hr: torch.Tensor) -> torch.Tensor:
        _, _, x_h, x_w = image_hr.size()
        h    = x_h // self.downscale
        w    = x_w // self.downscale
        x_lr = geometry.resize(image_hr, size=(h, w), interpolation="bilinear")
        y_lr = geometry.resize(guide_hr, size=(h, w), interpolation="bilinear")
        x_hr = image_hr
        return self.forward(x_lr, y_lr, x_hr)
        
# endregion
