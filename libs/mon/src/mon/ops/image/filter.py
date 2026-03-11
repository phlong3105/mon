#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Filters.

This module provides image filters.
"""

from __future__ import annotations

__all__ = [
    "BoxFilter",
    "FastGuidedFilter",
    "sobel_filter",
]

import cv2
import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F


# ==============================================================================
# region UTILS
# ==============================================================================

def _diff_x(image: Tensor, r: int) -> Tensor:
    """Compute the horizontal difference of an image.

    Args:
        image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.
        r (int): Radius of the filter.

    Returns:
        Tensor: The horizontal difference of the image.
    """
    # Validate inputs
    if image.ndim != 4:
        raise ValueError(
            f"Expected 'image' to have 4 dimensions, but got {image.ndim}."
        )

    left = image[:, :, r:2 * r + 1]
    middle = image[:, :, 2 * r + 1:] - image[:, :, :-2 * r - 1]
    right = image[:, :, -1:] - image[:, :, -2 * r - 1:    -r - 1]
    diff_x = torch.cat([left, middle, right], dim=2)
    return diff_x


def _diff_y(image: Tensor, r: int) -> Tensor:
    """Compute the vertical difference of an image.

    Args:
        image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.
        r (int): Radius of the filter.

    Returns:
        Tensor: The vertical difference of the image.
    """
    # Validate inputs
    if image.ndim != 4:
        raise ValueError(
            f"Expected 'image' to have 4 dimensions, but got {image.ndim}."
        )

    left = image[:, :, :, r:2 * r + 1]
    middle = image[:, :, :, 2 * r + 1:] - image[:, :, :, :-2 * r - 1]
    right = image[:, :, :, -1:] - image[:, :, :, -2 * r - 1:    -r - 1]
    diff_y = torch.cat([left, middle, right], dim=3)
    return diff_y

# endregion


# ==============================================================================
# region BOX FILTER
# ==============================================================================

class BoxFilter(nn.Module):
    """Box filter.

    Performs a box filter on an image. This is equivalent to a Gaussian filter
    with a uniform kernel.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, r: int):
        """Initialize a new instance.

        Args:
            r (int): Radius of the filter.
        """
        super().__init__()
        # Assign attributes
        self.r = r

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Apply the box filter to an image.

        Args:
            x (Tensor): Image tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: The filtered image.
        """
        return _diff_y(_diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)

# endregion


# ==============================================================================
# region SOBEL FILTER
# ==============================================================================

def sobel_filter(image: ndarray, kernel_size: int = 3) -> ndarray:
    """Apply Sobel filter to detect edges in an image.

    Args:
        image (ndarray): An RGB or grayscale image, formatted as an array of
            shape (H, W, C) and pixel values ranging from 0 to 255.
        kernel_size (int, optional): Sobel kernel size. Must be odd and greater
            than 1. Defaults to 3.

    Returns:
        ndarray: A single-channel image array of shape (H, W) with values
            ranging from 0 to 255, where higher values indicate stronger edges.

    Raises:
        TypeError: If ``image`` is not a numpy.ndarray with 2 or 3 dimensions.
    """
    if image.ndim not in [2, 3]:
        raise TypeError(
            f"Expected 'image' to have 2 or 3 dimensions, but got {image.ndim}."
        )

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

# endregion


# ==============================================================================
# region GUIDED FILTER
# ==============================================================================

class FastGuidedFilter(nn.Module):
    """Fast-guided filter.

    Perform a fast-guided filter on three images: low-resolution guidance image,
    low-resolution input image, and high-resolution guidance image. The output
    is a high-resolution filtered image.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, r: int, eps: float = 1e-8):
        """Initialize a new instance.

        Args:
            r (int): Radius of the filter.
            eps (float, optional): Regularization parameter to avoid division
                by zero. Defaults to 1e-8.
        """
        super().__init__()
        # Assign attributes
        self.r = r
        self.eps = eps
        self.box_filter = BoxFilter(r)

    # --- Callable & Context Manager ---
    def forward(self, lr_x: Tensor, lr_y: Tensor, hr_x: Tensor) -> Tensor:
        """Apply the guided filter to an image pair.

        Args:
            lr_x (Tensor): Low-resolution guidance image tensor of shape
                (B, C, H0, W0) and values ranging from 0.0 to 1.0.
            lr_y (Tensor): Low-resolution input image tensor of shape
                (B, C, H0, W0) and values ranging from 0.0 to 1.0.
            hr_x (Tensor): High-resolution guidance image tensor of shape
                (B, C, H1, W1) and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: The filtered high-resolution image tensor of shape
                (B, C, H1, W1) and values ranging from 0.0 to 1.0.
        """
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2 * self.r + 1 and w_lrx > 2 * self.r + 1

        N = self.box_filter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))
        mean_x = self.box_filter(lr_x) / N
        mean_y = self.box_filter(lr_y) / N
        cov_xy = self.box_filter(lr_x * lr_y) / N - mean_x * mean_y
        var_x  = self.box_filter(lr_x * lr_x) / N - mean_x * mean_x

        A = cov_xy / (var_x + self.eps)
        b = mean_y - A * mean_x

        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bilinear", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bilinear", align_corners=True)

        return mean_A * hr_x + mean_b

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
