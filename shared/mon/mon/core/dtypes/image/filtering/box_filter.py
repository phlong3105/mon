#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for box filtering.

This module implements box filtering techniques for image processing, including
a function, and a class for applying box filters to images. Box filtering is
commonly used for smoothing and noise reduction in images.
"""

__all__ = [
    "BoxFilter",
    "box_filter",
]

import torch
import torch.nn as nn


# ----- Utils -----
def _diff_x(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Computes difference along the x-axis of an image.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    
    Args:
        image (torch.Tensor): Image as torch.Tensor of shape (B, C, H, W) in
            [0.0, 1.0].
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        torch.Tensor: x-axis differences.
    
    Raises:
        ValueError: If image does not have 4 dimensions.
    """
    if image.ndim != 4:
        raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
    radius = int((kernel_size - 1) / 2)
    left   = image[:, :, radius        : 2 * radius + 1]
    middle = image[:, :, 2 * radius + 1:               ] - image[: , : ,                : -2 * radius - 1]
    right  = image[:, :, -1            :               ] - image[: , : , -2 * radius - 1:     -radius - 1]
    output = torch.cat([left, middle, right], dim=2)
    return output


def _diff_y(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Computes difference along the y-axis of an image.

    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    
    Args:
        image (torch.Tensor): Image as torch.Tensor of shape (B, C, H, W) in
            [0.0, 1.0].
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        torch.Tensor: y-axis differences.
    """
    if image.ndim != 4:
        raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
    radius = int((kernel_size - 1) / 2)
    left   = image[:, :, :,         radius:2 * radius + 1]
    middle = image[:, :, :, 2 * radius + 1:              ] - image[:, :, :,                :-2 * radius - 1]
    right  = image[:, :, :,             -1:              ] - image[:, :, :, -2 * radius - 1:    -radius - 1]
    output = torch.cat([left, middle, right], dim=3)
    return output


# ----- Box Filter -----
def box_filter(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Performs box filtering on an image.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    
    Args:
        image (torch.Tensor): Image as torch.Tensor of shape (B, C, H, W) in
            [0.0, 1.0].
        kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        torch.Tensor: Box filtered image.
    
    Raises:
        ValueError: If image does not have 4 dimensions.
    """
    if image.ndim != 4:
        raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
    return _diff_y(_diff_x(image.cumsum(dim=2), kernel_size).cumsum(dim=3), kernel_size)
    

class BoxFilter(nn.Module):
    """A class for box filtering.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    
    def __init__(self, kernel_size: int):
        """Initializes the BoxFilter module.
        
        Args:
            kernel_size (int): Kernel size (e.g., 3, 5, 7, 9).
        """
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Performs box filtering on an image.
        
        Args:
            image (torch.Tensor): Image as torch.Tensor of shape (B, C, H, W) in
                [0.0, 1.0].
        """
        return box_filter(image, self.kernel_size)
