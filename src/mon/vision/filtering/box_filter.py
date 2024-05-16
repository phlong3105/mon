#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements box filter."""

from __future__ import annotations

__all__ = [
    "BoxFilter",
    "box_filter",
    "box_filter_conv",
]

import cv2
import numpy as np
import torch
from plum import dispatch

from mon import core, nn
from mon.nn import functional as F


# region Utils

def diff_x(input: torch.Tensor, radius: int) -> torch.Tensor:
    """Compute the difference of the input along the x-axis.
    
    Args:
        input: A tensor with shape :math:`[B, C, H, W]`.
        radius: Radius of the kernel.
        
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py>`__
    """
    assert input.dim() == 4
    left   = input[:, :,         radius:2 * radius + 1]
    middle = input[:, :, 2 * radius + 1:              ] - input[:, :,                :-2 * radius - 1]
    right  = input[:, :,             -1:              ] - input[:, :, -2 * radius - 1:    -radius - 1]
    output = torch.cat([left, middle, right], dim=2)
    return output


def diff_y(input: torch.Tensor, radius: int) -> torch.Tensor:
    """Compute the difference of the input along the y-axis.
    
    Args:
        input: A tensor with shape :math:`[B, C, H, W]`.
        radius: Radius of the kernel.
    
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py>`__
    """
    assert input.dim() == 4
    left   = input[:, :, :,         radius:2 * radius + 1]
    middle = input[:, :, :, 2 * radius + 1:              ] - input[:, :, :,                :-2 * radius - 1]
    right  = input[:, :, :,             -1:              ] - input[:, :, :, -2 * radius - 1:    -radius - 1]
    output = torch.cat([left, middle, right], dim=3)
    return output

# endregion


# region Box Filter

@dispatch
def box_filter(
    image      : torch.Tensor,
    kernel_size: int | None = None,
    radius     : int | None = None,
) -> torch.Tensor:
    """Perform box filer on the image.
    
    Args:
        image: An image in :math:`[B, C, H, W]` format.
        kernel_size: Size of the kernel. Commonly be ``3``, ``5``, ``7``, or ``9``.
        radius: Radius of the kernel (kernel_size = radius * 2 + 1).
            Commonly be ``1``, ``2``, ``3``, or ``4``.
        
    Returns:
        A filtered image.
    
    References:
        `<https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/box_filter.py>`__
    """
    assert kernel_size is not None or radius is not None, \
        "Either :param:`kernel_size` or :param:`radius` must be provided."
    radius = radius or int((kernel_size - 1) / 2)
    assert image.dim() == 4
    return diff_y(diff_x(image.cumsum(dim=2), radius).cumsum(dim=3), radius)
    

@dispatch
def box_filter(
    image      : np.ndarray,
    kernel_size: int | None = None,
    radius     : int | None = None,
    **kwargs
) -> np.ndarray:
    """Perform box filter on the image.
    
    Args:
        image: An image in :math:`[H, W, C]` format.
        kernel_size: Size of the kernel. Commonly be ``3``, ``5``, ``7``, or ``9``.
        radius: Radius of the kernel (kernel_size = radius * 2 + 1).
            Commonly be ``1``, ``2``, ``3``, or ``4``.
    
    kwargs (:func:`cv2.boxFilter`) includes:
        ddepth: The output image depth. Default: ``-1`` means the same as the
            depth as :attr:`image`.
        anchor: The anchor of the kernel. Default: ``(-1, -1)`` means at the center.
        normalize: Whether to normalize the kernel. Default: ``False``.
        borderType: Border mode used to extrapolate pixels outside of the image.
            Default: `cv2.BORDER_DEFAULT`.
        
    Returns:
        A filtered image.
    """
    assert kernel_size is not None or radius is not None, \
        "Either :param:`kernel_size` or :param:`radius` must be provided."
    ddepth      = kwargs.get("ddepth",     -1)
    anchor      = kwargs.get("anchor",     (-1, -1))
    normalize   = kwargs.get("normalize",  False)
    borderType  = kwargs.get("borderType", cv2.BORDER_DEFAULT)
    kernel_size = kernel_size or 2 * radius + 1
    kernel_size = core.parse_hw(kernel_size)
    return cv2.boxFilter(
        src        = image,
        ddepth     = ddepth,
        ksize      = kernel_size,
        anchor     = anchor,
        normalize  = normalize,
        borderType = borderType,
    )


@dispatch
def box_filter_conv(
    image      : torch.Tensor,
    kernel_size: int | None = None,
    radius     : int | None = None,
) -> torch.Tensor:
    """Perform box filer on the image.
    
    Args:
        image: An image in :math:`[B, C, H, W]` format.
        kernel_size: Size of the kernel. Commonly be ``3``, ``5``, ``7``, or ``9``.
        radius: Radius of the kernel (kernel_size = radius * 2 + 1).
            Commonly be ``1``, ``2``, ``3``, or ``4``.
        
    Returns:
        A filtered image.
    """
    assert kernel_size is not None or radius is not None, \
        "Either :param:`kernel_size` or :param:`radius` must be provided."
    kernel_size = kernel_size or 2 * radius + 1
    b, c, h, w  = image.shape
    # Create a 2D box kernel with all values as 1
    kernel  = torch.ones(b, 1, kernel_size, kernel_size, device=image.device)
    # Normalize the kernel
    kernel /= kernel_size ** 2
    # Apply 2D convolution separately to each channel
    output  = [F.conv2d(image[:, i:i + 1, :, :], kernel, padding=kernel_size // 2) for i in range(image.size(1))]
    # Concatenate the filtered channels along the channel dimension
    output  = torch.cat(output, dim=1)
    return output


class BoxFilter(nn.Module):
    """Box filter."""
    
    def __init__(
        self,
        kernel_size: int | None = None,
        radius     : int | None = None,
    ):
        super().__init__()
        assert kernel_size is not None or radius is not None, \
            "Either :param:`kernel_size` or :param:`radius` must be provided."
        self.kernel_size = kernel_size or 2 * radius + 1
        self.radius      = int((self.kernel_size - 1) / 2)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dim() == 4
        return box_filter(input, self.kernel_size, self.radius)
    
# endregion
