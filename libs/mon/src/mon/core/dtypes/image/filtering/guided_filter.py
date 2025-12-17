#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for guided filters.

This module implements various guided filtering techniques, including standard
guided filtering, fast guided filtering, and convolutional guided filtering.
These filters are useful for edge-preserving smoothing and detail enhancement in
images.
"""

__all__ = [
    "ConvGuidedFilter",
    "FastGuidedFilter",
    "GuidedFilter",
    "guided_filter",
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .box_filter import BoxFilter


# ----- Guided Filter -----
def guided_filter(
    image      : torch.Tensor,
    guide      : torch.Tensor,
    kernel_size: int,
    eps        : float = 1e-8
) -> torch.Tensor:
    """Applies guided filtering to an image.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    
    Args:
        image (torch.Tensor): Input image as torch.Tensor of shape (B, C, H, W)
            with pixel values in [0.0, 1.0].
        guide (torch.Tensor): Guidance image with the same shape, type, and
            format as ``image``.
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
        eps (float): Sharpness control value. Defaults to 1e-8.
    
    Returns:
        torch.Tensor: Filtered image with the same shape, type, and format as
            ``image``.
    
    Raises:
        TypeError: If ``image`` or ``guide`` is not a torch.Tensor.
    """
    if not isinstance(image, torch.Tensor) or not isinstance(guide, torch.Tensor):
        raise TypeError(f"``image`` and ``guide`` must be torch.Tensor, got {type(image)} and {type(guide)}.")
    
    x          = image
    y          = guide
    box_filter = BoxFilter(kernel_size=kernel_size)
    _, _, h, w = x.shape
    N          = box_filter(Variable(x.data.new().resize_((1, 1, h, w)).fill_(1.0)))
    mean_x     = box_filter(x) / N
    mean_y     = box_filter(y) / N
    cov_xy     = box_filter(x * y) / N - mean_x * mean_y
    var_x      = box_filter(x * x) / N - mean_x * mean_x
    A          = cov_xy / (var_x + eps)
    b          = mean_y - A * mean_x
    mean_A     = box_filter(A) / N
    mean_b     = box_filter(b) / N
    return mean_A * x + mean_b


class GuidedFilter(nn.Module):
    """A class that applies guided filtering to an image.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    
    Attributes:
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
        eps (float): Sharpness control value.
        box_filter (BoxFilter): Box filter instance.
    """

    def __init__(self, kernel_size: int, eps: float = 1e-8):
        """Initializes the GuidedFilter instance.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
            eps: Sharpness control value. Defaults to 1e-8.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.eps         = eps
        self.box_filter  = BoxFilter(kernel_size=kernel_size)

    def forward(self, image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Filters an image using a guide.
        
        Args:
            image: Input image as torch.Tensor of shape (B, C, H, W) with pixel
                values in [0.0, 1.0].
            guide: Guidance image with the same shape, type, and format as
                ``image``.
        
        Returns:
            Filtered image with the same shape, type, and format as ``image``.
        """
        x          = image
        y          = guide
        _, _, h, w = x.shape
        N          = self.box_filter(torch.ones(1, 1, h, w, device=x.device))
        mean_x     = self.box_filter(x) / N
        mean_y     = self.box_filter(y) / N
        cov_xy     = self.box_filter(x * y) / N - mean_x * mean_y
        var_x      = self.box_filter(x * x) / N - mean_x * mean_x
        A          = cov_xy / (var_x + self.eps)
        b          = mean_y - A * mean_x
        mean_A     = self.box_filter(A) / N
        mean_b     = self.box_filter(b) / N
        return mean_A * x + mean_b


class FastGuidedFilter(nn.Module):
    """A class that applies fast guided filtering to an image.

    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    
    Attributes:
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
        eps (float): Sharpness control value.
        box_filter (BoxFilter): Box filter instance.
    """

    def __init__(self, kernel_size: int, eps: float = 1e-8):
        """Initializes the GuidedFilter instance.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
            eps: Sharpness control value. Defaults to 1e-8.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.eps         = eps
        self.box_filter  = BoxFilter(kernel_size=kernel_size)

    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filters a high-resolution image using low-resolution image and guide.

        Args:
            x_lr (torch.Tensor): Low-res input image as torch.Tensor of shape
                (B, C, H, W) in [0.0, 1.0].
            y_lr (torch.Tensor): Low-res guidance image with the same type and
                format as ``x_lr``.
            x_hr (torch.Tensor): High-res input image with the same type and
                format as ``x_lr``, but larger in size.
        
        Returns:
            torch.Tensor: Filtered high-resolution image with the same type and
                format as ``x_hr``.
        """
        _, _, h_xlr, w_xlr = x_lr.shape
        _, _, h_xhr, w_xhr = x_hr.shape
        N      = self.box_filter(torch.ones(1, 1, h_xlr, w_xlr, device=x_lr.device))
        mean_x = self.box_filter(x_lr) / N
        mean_y = self.box_filter(y_lr) / N
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        var_x  = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x
        A      = cov_xy / (var_x + self.eps)
        b      = mean_y - A * mean_x
        mean_A = F.interpolate(A, (h_xhr, w_xhr), mode="bicubic", align_corners=True)
        mean_b = F.interpolate(b, (h_xhr, w_xhr), mode="bicubic", align_corners=True)
        return mean_A * x_hr + mean_b


class ConvGuidedFilter(nn.Module):
    """Applies convolutional guided filtering to an image.
 
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    
    Attributes:
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
        box_filter (nn.Conv2d): Box filter implemented as a convolutional layer.
        conv_a (nn.Sequential): Convolutional layers to compute the linear
            coefficients.
    """

    def __init__(self, kernel_size: int, norm: nn.Module = nn.BatchNorm2d):
        """Initializes the ConvGuidedFilter instance.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
            norm: Normalization layer to use. Defaults to nn.BatchNorm2d.
        """
        super().__init__()
        radius = int((kernel_size - 1) / 2)
        self.box_filter = nn.Conv2d(3, 3, 3, padding=radius, dilation=radius, bias=False, groups=3)
        self.conv_a     = nn.Sequential(
            nn.Conv2d(6, 32, 1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, bias=False)
        )
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filters a high-resolution image using low-resolution image and guide.

        Args:
            x_lr (torch.Tensor): Low-res input image as torch.Tensor of shape
                (B, C, H, W) in [0.0, 1.0].
            y_lr (torch.Tensor): Low-res guidance image with the same type and
                format as ``x_lr``.
            x_hr (torch.Tensor): High-res input image with the same type and
                format as ``x_lr``, but larger in size.
        
        Returns:
            torch.Tensor: Filtered high-resolution image with the same type and
                format as ``x_hr``.
        """
        _, _, h_lrx, w_lrx = x_lr.shape
        _, _, h_hrx, w_hrx = x_hr.shape
        N      = self.box_filter(torch.ones(1, 3, h_lrx, w_lrx, device=x_lr.device))
        mean_x = self.box_filter(x_lr) / N
        mean_y = self.box_filter(y_lr) / N
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        var_x  = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x
        A      = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b      = mean_y - A * mean_x
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bicubic", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bicubic", align_corners=True)
        return mean_A * x_hr + mean_b
