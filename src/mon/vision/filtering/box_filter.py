#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Box Filter.

This module implements box filters.
"""

from __future__ import annotations

__all__ = [
    "BoxFilter",
    "box_filter",
    "box_filter_conv",
]

import cv2
import numpy as np
import torch

from mon import core, nn
from mon.nn import functional as F


# region Utils

def diff_x(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Compute the difference of the input along the x-axis.
    
    Args:
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0.0, 1.0]``.
        radius: Radius of the kernel.
        
    References:
        https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    if image.ndim != 4:
        raise ValueError(f"`image` must have 4 dimensions.")
    left   = image[:, :, radius        : 2 * radius + 1]
    middle = image[:, :, 2 * radius + 1:               ] - image[: , : ,                : -2 * radius - 1]
    right  = image[:, :, -1            :               ] - image[: , : , -2 * radius - 1:     -radius - 1]
    output = torch.cat([left, middle, right], dim=2)
    return output


def diff_y(image: torch.Tensor, radius: int) -> torch.Tensor:
    """Compute the difference of the image along the y-axis.
    
    Args:
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0.0, 1.0]``.
        radius: Radius of the kernel.
    
    References:
        https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    if image.ndim != 4:
        raise ValueError(f"`image` must have 4 dimensions.")
    left   = image[:, :, :,         radius:2 * radius + 1]
    middle = image[:, :, :, 2 * radius + 1:              ] - image[:, :, :,                :-2 * radius - 1]
    right  = image[:, :, :,             -1:              ] - image[:, :, :, -2 * radius - 1:    -radius - 1]
    output = torch.cat([left, middle, right], dim=3)
    return output

# endregion


# region Box Filter

def box_filter(
    image      : torch.Tensor,
    kernel_size: int = None,
    radius     : int = None,
    **kwargs
) -> torch.Tensor:
    """Perform box filer on the image.
    
    Args:
        image: An image of type :obj:`torch.Tensor` in ``[B, C, H, W]`` format
            with data in the range ``[0.0, 1.0]``.
        kernel_size: Size of the kernel. Commonly be ``3``, ``5``, ``7``, or
            ``9``.
        radius: Radius of the kernel (kernel_size = radius * 2 + 1).
            Commonly be ``1``, ``2``, ``3``, or ``4``.
    
        kwargs (:obj:`cv2.boxFilter`) includes:
            ddepth: The output image depth. Default: ``-1`` means the same as
                the depth as :obj:`image`.
            anchor: The anchor of the kernel. Default: ``(-1, -1)`` means at
                the center.
            normalize: Whether to normalize the kernel. Default: ``False``.
            borderType: Border mode used to extrapolate pixels outside of the
                image. Default: `cv2.BORDER_DEFAULT`.
                
    Returns:
        A filtered image.
    
    References:
        https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py
    """
    if kernel_size is None and radius is None:
        raise ValueError("Either `kernel_size` or `radius` must be provided.")
    if isinstance(image, torch.Tensor):
        if image.ndim != 4:
            raise ValueError(f"`image` must have 4 dimensions.")
        radius = radius or int((kernel_size - 1) / 2)
        return diff_y(diff_x(image.cumsum(dim=2), radius).cumsum(dim=3), radius)
    elif isinstance(image, np.ndarray):
        ddepth      = kwargs.get("ddepth",     -1)
        anchor      = kwargs.get("anchor",     (-1, -1))
        normalize   = kwargs.get("normalize",  False)
        borderType  = kwargs.get("borderType", cv2.BORDER_DEFAULT)
        kernel_size = kernel_size or 2 * radius + 1
        kernel_size = core.get_image_size(kernel_size)
        return cv2.boxFilter(
            src        = image,
            ddepth     = ddepth,
            ksize      = kernel_size,
            anchor     = anchor,
            normalize  = normalize,
            borderType = borderType,
        )
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    

def box_filter_conv(
    image      : torch.Tensor,
    kernel_size: int = None,
    radius     : int = None,
) -> torch.Tensor:
    """Perform box filer on the image.
    
    Args:
        image: An image in `[B, C, H, W]` format.
        kernel_size: Size of the kernel. Commonly be ``3``, ``5``, ``7``, or ``9``.
        radius: Radius of the kernel (kernel_size = radius * 2 + 1).
            Commonly be ``1``, ``2``, ``3``, or ``4``.
        
    Returns:
        A filtered image.
    """
    if kernel_size is None and radius is None:
        raise ValueError("Either `kernel_size` or `radius` must be provided.")
    kernel_size = kernel_size or 2 * radius + 1
    b, c, h, w  = image.shape
    # Create a 2D box kernel with all values as 1
    kernel  = torch.ones(b, 1, kernel_size, kernel_size, device=image.device)
    # Normalize the kernel
    kernel /= kernel_size ** 2
    # Apply 2D convolution separately to each channel
    output  = [F.conv2d(image[:, i:i + 1, :, :], kernel, padding=kernel_size // 2)
               for i in range(image.size(1))]
    # Concatenate the filtered channels along the channel dimension
    output  = torch.cat(output, dim=1)
    return output


class BoxFilter(nn.Module):
    
    def __init__(
        self,
        kernel_size: int = None,
        radius     : int = None,
    ):
        super().__init__()
        if kernel_size is None and radius is None:
            raise ValueError("Either `kernel_size` or `radius` must be provided.")
        self.kernel_size = kernel_size or 2 * radius + 1
        self.radius      = int((self.kernel_size - 1) / 2)
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return box_filter(image, self.kernel_size, self.radius)
    
# endregion
