#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image filtering operations.

This package implements various image filtering techniques, including box filter,
guided filter, and Sobel filter. These filters are commonly used in image
processing tasks such as smoothing, edge detection, and detail enhancement.
"""

from __future__ import annotations

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


# ==============================================================================
# region SPATIAL FILTERS
# ==============================================================================

# --- Linear Smoothing ---
class BoxFilter(nn.Module):
    """A module that performs box filtering on an image.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    
    Attributes:
        k (int): Kernel size (e.g., 3, 5, 7, 9).
        r (int): Radius of the box filter.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, kernel_size: int):
        """Initialize a new instance.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
        """
        super().__init__()
        self.k = kernel_size
        self.r = kernel_size // 2
    
    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform box filtering on an image.
        
        Args:
            x: An RGB image as a torch.Tensor of shape (B, C, H, W) and
                pixel values ranging from 0.0 to 1.0.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 'x' to have 4 dimensions, but got {x.ndim}.")
        
        # Pad to handle boundaries (Replicate padding matches your diff logic)
        x_padded = F.pad(x, (self.r + 1, self.r, self.r + 1, self.r), mode="replicate")

        # Integrate along Y axis
        # Formula: Sum = Image[y+r] - Image[y-r-1]
        sum_y  = x_padded.cumsum(dim=2)
        diff_y = sum_y[:, :, self.k:, :] - sum_y[:, :, :-self.k, :]

        # Integrate along X axis
        # Formula: Sum = Image[x+r] - Image[x-r-1]
        sum_x  = diff_y.cumsum(dim=3)
        diff_x = sum_x[:, :, :, self.k:] - sum_x[:, :, :, :-self.k]

        # Normalize to get the mean (The "Box" average)
        return diff_x / (self.k ** 2)
        
        # TODO: Delete later
        """
        if image.ndim != 4:
            raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
        return self._diff_y(self._diff_x(image.cumsum(dim=2)).cumsum(dim=3))
        """
    
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
        image: An RGB or grayscale image, formatted as a numpy.ndarray of shape
            (H, W, C) and pixel values ranging from 0 to 255.
        kernel_size: Sobel kernel size. Must be odd and greater than 1. Defaults to 3.
        
    Returns:
        The image after applying Sobel filter.
    
    Raises:
        TypeError: If ``image`` is not a numpy.ndarray with 2 or 3 dimensions.
    """
    if image.ndim not in [2, 3]:
        raise TypeError(f"Expected 'image' to have 2 or 3 dimensions, but got {image.ndim}.")
    
    # Automatic Color Handling
    # If the image is (H, W, 3), convert to gray.
    if image.ndim == 3 and image.shape[-1] in [3, 4]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute Gradients
    # cv2.CV_64F prevents truncation of negative values
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # Combine Magnitudes
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Final Normalize & Cast
    # Using cv2.normalize allows for better control over contrast
    output = cv2.convertScaleAbs(magnitude)
    
    return output
    

# --- Edge-Preserving & Joint Filters ---
class GuidedFilter(nn.Module):
    """A class that applies guided filtering to an image.
    
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
    def forward(self, image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Filter an image using a guidance image.
        
        Args:
            image: An RGB image, formatted as a torch.Tensor of shape
                (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
            guide: A guidance image with the same shape, type, and format as
                ``image``.
        
        Returns:
            Filtered image with the same shape, type, and format as ``image``.
        """
        # p is image to be filtered, I is guidance
        p = image
        I = guide
        
        # Handle the normalization factor N (counts pixels in local windows)
        # This correctly accounts for edges where the window is smaller
        ones = torch.ones(1, 1, p.shape[2], p.shape[3], device=p.device, dtype=p.dtype)
        N    = self.box_filter(ones)
    
        # Compute means
        mean_I  = self.box_filter(I) / N
        mean_p  = self.box_filter(p) / N
        mean_Ip = self.box_filter(I * p) / N
        mean_II = self.box_filter(I * I) / N
    
        # Compute covariance and variance
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I  = mean_II - mean_I * mean_I
    
        # Calculate linear coefficients A and b
        # A = Cov(I, p) / (Var(I) + eps)
        A = cov_Ip / (var_I + self.eps)
        b = mean_p - A * mean_I
    
        # Average the coefficients over the windows
        mean_A = self.box_filter(A) / N
        mean_b = self.box_filter(b) / N
    
        # Final output reconstruction
        return mean_A * I + mean_b


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
            x_lr: A low-resolution RGB image, formatted as a torch.Tensor of
                shape (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
            y_lr: A low-resolution guidance image with the same shape, type, and
                format as ``x_lr``.
            x_hr: The high-resolution version of ``x_lr``.
        
        Returns:
            Filtered image with the same shape, type, and format as ``x_hr``.
        """
        # Compute stats in low-resolution space
        ones = torch.ones(1, 1, x_lr.shape[2], x_lr.shape[3], device=x_lr.device, dtype=x_lr.dtype)
        N    = self.box_filter(ones)
    
        mean_x  = self.box_filter(x_lr) / N
        mean_y  = self.box_filter(y_lr) / N
        mean_xx = self.box_filter(x_lr * x_lr) / N
        mean_xy = self.box_filter(x_lr * y_lr) / N
    
        # Compute coefficients A and b at low-res
        var_x  = mean_xx - mean_x * mean_x
        cov_xy = mean_xy - mean_x * mean_y
        
        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x
    
        # Upsample coefficients to high-res
        # mode="bilinear" is generally safer to avoid overshoot artifacts
        h_hr, w_hr = x_hr.shape[2:]
        mean_A = F.interpolate(A, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
        mean_b = F.interpolate(b, size=(h_hr, w_hr), mode="bilinear", align_corners=False)
    
        # High-res reconstruction
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
        self.radius = kernel_size // 2
        
        # Use AvgPool2d for a true, efficient box filter (local mean)
        # count_include_pad=False correctly handles image boundaries
        self.box_filter = nn.AvgPool2d(
            kernel_size       = kernel_size,
            stride            = 1,
            padding           = self.radius,
            count_include_pad = False
        )
        
        # Learnable regression for A (Gain)
        # Input: cat([cov_xy, var_x]) -> 6 channels
        self.conv_a = nn.Sequential(
            nn.Conv2d(6, 32, 1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, bias=False)
        )
        
    # --- Callable & Context Manager ---
    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filter a high-resolution image using a low-resolution image and guide.

        Args:
            x_lr: A low-resolution RGB image, formatted as a torch.Tensor of
                shape (B, 3, H, W) and pixel values ranging from 0.0 to 1.0.
            y_lr: A low-resolution guidance image with the same shape, type, and
                format as ``x_lr``.
            x_hr: The high-resolution version of ``x_lr``.
        
        Returns:
            Filtered image with the same shape, type, and format as ``x_hr``.
        """
        # Compute stats in low-res space using AvgPool2d
        # This replaces the conv2d + ones normalization logic
        mean_x = self.box_filter(x_lr)
        mean_y = self.box_filter(y_lr)
        cov_xy = self.box_filter(x_lr * y_lr) - mean_x * mean_y
        var_x  = self.box_filter(x_lr * x_lr) - mean_x * mean_x
        
        # Predict adaptive coefficients A
        # Features are concatenated along the channel dimension (B, 6, H, W)
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        
        # Compute bias b
        b = mean_y - A * mean_x
        
        # Upsample coefficients to high-res
        # mode='bilinear' is often preferred for coefficients to prevent ringing
        hr_size = x_hr.shape[-2:]
        mean_A  = F.interpolate(A, size=hr_size, mode="bilinear", align_corners=False)
        mean_b  = F.interpolate(b, size=hr_size, mode="bilinear", align_corners=False)
        
        # Apply linear model at high-res
        return mean_A * x_hr + mean_b

# endregion
