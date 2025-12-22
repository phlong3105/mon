#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Box filter.

This module implements box filtering techniques for image processing.
"""

__all__ = [
    "BoxFilter",
    "box_filter",
]

import torch
import torch.nn as nn


# --- Utils ---
def _diff_x(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Compute the differences along the x-axis of an image.
    
    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1].
        kernel_size: Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        The x-axis differences.
    
    Raises:
        ValueError: If ``image`` does not have 4 dimensions.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
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
    """Compute the differences along the y-axis of an image.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1].
        kernel_size: Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        The y-axis differences.
    
    Raises:
        ValueError: If ``image`` does not have 4 dimensions.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    if image.ndim != 4:
        raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
    radius = int((kernel_size - 1) / 2)
    left   = image[:, :, :,         radius:2 * radius + 1]
    middle = image[:, :, :, 2 * radius + 1:              ] - image[:, :, :,                :-2 * radius - 1]
    right  = image[:, :, :,             -1:              ] - image[:, :, :, -2 * radius - 1:    -radius - 1]
    output = torch.cat([left, middle, right], dim=3)
    return output


# --- Box Filter ---
def box_filter(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Perform box filtering on an image.
    
    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1].
        kernel_size: Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        torch.Tensor: Box filtered image.
    
    Raises:
        ValueError: If ``image`` does not have 4 dimensions.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
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
        """Initialize the BoxFilter module.
        
        Args:
            kernel_size: Kernel size (e.g., 3, 5, 7, 9).
        """
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Perform box filtering on an image.
        
        Args:
            image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
                values in the range [0, 1].
        """
        return box_filter(image, self.kernel_size)
