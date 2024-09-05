#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Miscellaneous Image Processing Ops.

This module implements miscellaneous image processing operations that are not
specific to any particular category.
"""

from __future__ import annotations

__all__ = [
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "add_weighted",
    "blend_images",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
]

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mon.core.image import utils


# region Combination

def add_weighted(
    image1: torch.Tensor | np.ndarray,
    alpha : float,
    image2: torch.Tensor | np.ndarray,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1: The first image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        alpha: The weight of the :obj:`image1` elements.
        image2: The same as :obj:`image1`.
        beta: The weight of the :obj:`image2` elements.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A weighted image.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"`image1` and `image2` must have the same shape, "
                         f"but got {image1.shape} and {image2.shape}.")
    bound  = 1.0 if utils.is_normalized_image(image1) else 255.0
    output = image1 * alpha + image2 * beta + gamma
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(image1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound)
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(input)}.")
    return output


def blend_images(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blend 2 images together using the formula:
        output = :obj:`image1` * alpha + :obj:`image2` * beta + gamma

    Args:
        image1: A source image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        image2: An overlay image that we want to blend on top of :obj:`image1`.
        alpha: An alpha transparency of the overlay.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A blended image.
    """
    return add_weighted(
        image1 = image2,
        alpha  = alpha,
        image2 = image1,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )

# endregion


# region Gradient

def image_local_mean(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local mean of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return patches.mean(dim=(4, 5))


def image_local_variance(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculate the local variance of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean    = patches.mean(dim=(4, 5))
    return ((patches - mean.unsqueeze(4).unsqueeze(5)) ** 2).mean(dim=(4, 5))


def image_local_stddev(
    image     : torch.Tensor,
    patch_size: int   = 5,
    eps       : float = 1e-9
) -> torch.Tensor:
    """Calculate the local standard deviation of an image using a sliding window.
    
    Args:
        image: The input image tensor of shape ``[B, C, H, W]``.
        patch_size: The size of the sliding window. Default: ``5``.
        eps: A small value to avoid sqrt by zero. Default: ``1e-9``.
    """
    padding        = patch_size // 2
    image          = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches        = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean           = patches.mean(dim=(4, 5), keepdim=True)
    squared_diff   = (patches - mean) ** 2
    local_variance = squared_diff.mean(dim=(4, 5))
    local_stddev   = torch.sqrt(local_variance + eps)
    return local_stddev


class ImageLocalMean(nn.Module):
    """Calculate the local mean of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_mean(image, self.patch_size)


class ImageLocalVariance(nn.Module):
    """Calculate the local variance of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
    """
    
    def __init__(self, patch_size: int = 5):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image):
        return image_local_variance(image, self.patch_size)


class ImageLocalStdDev(nn.Module):
    """Calculate the local standard deviation of an image using a sliding window.
    
    Args:
        patch_size: The size of the sliding window. Default: ``5``.
        eps: A small value to avoid sqrt by zero. Default: ``1e-9``.
    """
    
    def __init__(self, patch_size: int = 5, eps: float = 1e-9):
        super().__init__()
        self.patch_size = patch_size
        self.eps        = eps
    
    def forward(self, image):
        return image_local_stddev(image, self.patch_size, self.eps)
    
# endregion
