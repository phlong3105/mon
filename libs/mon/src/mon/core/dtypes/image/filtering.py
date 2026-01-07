#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image filtering operations.

This package implements various image filtering techniques, including box filter,
guided filter, and Sobel filter. These filters are commonly used in image
processing tasks such as smoothing, edge detection, and detail enhancement.
"""

__all__ = [
    "BoxFilter",
    "ConvGuidedFilter",
    "FastGuidedFilter",
    "GuidedFilter",
    "sobel_filter",
]

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import is_color


# ==============================================================================
# SPATIAL FILTERS
# ==============================================================================

# --- Linear Smoothing ---
class BoxFilter(nn.Module):
    """A module that performs box filtering on an image.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    
    Attributes:
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, kernel_size: int):
        """Initialize a new instance.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
        """
        super().__init__()
        self.kernel_size = kernel_size
    
    # --- Callable & Context Manager ---
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Perform box filtering on an image.
        
        Args:
            image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
                values in the range [0, 1].
        """
        if image.ndim != 4:
            raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
        return self._diff_y(self._diff_x(image.cumsum(dim=2)).cumsum(dim=3))
    
    def _diff_x(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the differences along the x-axis of an image.
        
        Args:
            image: An RGB image, formatted as a torch.Tensor with dimensions
                (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
        
        Returns:
            The x-axis differences.
        
        Raises:
            ValueError: If ``image`` does not have 4 dimensions.
        """
        if image.ndim != 4:
            raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
        radius = int((self.kernel_size - 1) / 2)
        left   = image[:, :, radius        : 2 * radius + 1]
        middle = image[:, :, 2 * radius + 1:               ] - image[: , : ,                : -2 * radius - 1]
        right  = image[:, :, -1            :               ] - image[: , : , -2 * radius - 1:     -radius - 1]
        diff_x = torch.cat([left, middle, right], dim=2)
        return diff_x
    
    def _diff_y(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the differences along the y-axis of an image.
    
        Args:
            image: An RGB image, formatted as a torch.Tensor with dimensions
                (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
        
        Returns:
            The y-axis differences.
        
        Raises:
            ValueError: If ``image`` does not have 4 dimensions.
        """
        if image.ndim != 4:
            raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
        radius = int((self.kernel_size - 1) / 2)
        left   = image[:, :, :,         radius:2 * radius + 1]
        middle = image[:, :, :, 2 * radius + 1:              ] - image[:, :, :,                :-2 * radius - 1]
        right  = image[:, :, :,             -1:              ] - image[:, :, :, -2 * radius - 1:    -radius - 1]
        diff_y = torch.cat([left, middle, right], dim=3)
        return diff_y


# --- Edge Detection ---
def sobel_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Sobel filter to detect edges in an image.

    Args:
        image: An RGB or grayscale image, formatted as a numpy.ndarray with
            dimensions (H, W, C) and pixel values ranging from 0 to 255.
        kernel_size: Sobel kernel size. Must be odd and greater than 1. Defaults to 3.
        
    Returns:
        The image after applying Sobel filter.
    
    Raises:
        TypeError: If ``image`` is not a numpy.ndarray with 2 or 3 dimensions.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in [2, 3]:
        raise TypeError(f"``image`` must be a numpy.ndarray with 2 or 3 dimensions, "
                        f"got {type(image)} with shape {image.shape}.")
    
    if is_color(image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    return sobel_combined


# --- Edge-Preserving & Joint Filters ---
class GuidedFilter(nn.Module):
    """A class that applies guided filtering to an image.
    
    Attributes:
        box_filter (BoxFilter): Box filter instance.
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
        eps (float): Sharpness control value.
        
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, kernel_size: int, eps: float = 1e-8):
        """Initialize a new instance.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
            eps: Sharpness control value. Defaults to 1e-8.
        """
        super().__init__()
        self.box_filter  = BoxFilter(kernel_size=kernel_size)
        self.kernel_size = kernel_size
        self.eps         = eps
    
    # --- Callable & Context Manager ---
    def forward(self, image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Filter an image using a guidance image.
        
        Args:
            image: An RGB image, formatted as a torch.Tensor with dimensions
                (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
            guide: A guidance image with the same shape, type, and format as
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
    """A module that applies fast-guided filtering to an image.
  
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    
    Attributes:
        box_filter (BoxFilter): Box filter instance.
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
        eps (float): Sharpness control value.
     """

    # --- Lifecycle & Initialization ---
    def __init__(self, kernel_size: int, eps: float = 1e-8):
        """Initialize a new instance.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
            eps: Sharpness control value. Defaults to 1e-8.
        """
        super().__init__()
        self.box_filter  = BoxFilter(kernel_size=kernel_size)
        self.kernel_size = kernel_size
        self.eps         = eps
        
    # --- Callable & Context Manager ---
    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filter a high-resolution image using a low-resolution image and guide.
        
        Args:
            x_lr: A low-resolution RGB image, formatted as a torch.Tensor with
                dimensions (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
            y_lr: A low-resolution guidance image with the same shape, type, and
                format as ``x_lr``.
            x_hr: The high-resolution version of ``x_lr``.
        
        Returns:
            Filtered image with the same shape, type, and format as ``x_hr``.
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
    """A module that applies convolutional guided filtering to an image.
 
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    
    Attributes:
        box_filter (nn.Conv2d): Box filter implemented as a convolutional layer.
        conv_a (nn.Sequential): Convolutional layers to compute the linear
            coefficients.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, kernel_size: int, norm: nn.Module = nn.BatchNorm2d):
        """Initialize a new instance.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
            norm: Normalization layer to use. Defaults to nn.BatchNorm2d.
        """
        super().__init__()
        radius          = int((kernel_size - 1) / 2)
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

    # --- Callable & Context Manager ---
    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filter a high-resolution image using a low-resolution image and guide.

        Args:
            x_lr: A low-resolution RGB image, formatted as a torch.Tensor with
                dimensions (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
            y_lr: A low-resolution guidance image with the same shape, type, and
                format as ``x_lr``.
            x_hr: The high-resolution version of ``x_lr``.
        
        Returns:
            Filtered image with the same shape, type, and format as ``x_hr``.
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
