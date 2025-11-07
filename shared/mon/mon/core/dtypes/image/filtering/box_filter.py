#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements box filters."""

__all__ = [
    "BoxFilter",
    "box_filter",
]

import torch
import torch.nn as nn


# ----- Utils -----
def _diff_x(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Computes difference along the x-axis of an image.
    
    Args:
        image: Image as ``torch.Tensor`` of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`.
        kernel_size: Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        A ``torch.Tensor`` with x-axis differences.
    
    Raises:
        ValueError: If image does not have 4 dimensions.
    
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
    """Computes difference along the y-axis of an image.

    Args:
        image: Image as ``torch.Tensor`` of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`.
        kernel_size: Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        A ``torch.Tensor`` with y-axis differences.
    
    Raises:
        ValueError: If image does not have 4 dimensions.
    
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


# ----- Box Filter -----
def box_filter(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Performs box filtering on an image.

    Args:
        image: Image as ``torch.Tensor`` of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`.
        kernel_size: Kernel size (e.g., 3, 5, 7, 9).
    
    Returns:
        Filtered image.
    
    Raises:
        ValueError: If neither ``kernel_size`` nor ``radius`` is provided, or
            image dimensions are invalid.
        TypeError: If image type is neither ``torch.Tensor`` nor ``numpy.ndarray``.
    
    References:
        - Code: https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    if image.ndim != 4:
        raise ValueError(f"``image`` must have 4 dimensions, got {image.ndim}.")
    return _diff_y(_diff_x(image.cumsum(dim=2), kernel_size).cumsum(dim=3), kernel_size)
    

class BoxFilter(nn.Module):
    """Applies box filtering to an image.

    Args:
        kernel_size: Size of the kernel (e.g., 3, 5, 7, 9).
    """
    
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return box_filter(image, self.kernel_size)
