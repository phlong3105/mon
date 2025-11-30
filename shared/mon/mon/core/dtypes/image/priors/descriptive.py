#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for descriptive image priors.

This module implements descriptive image priors such as local mean, local
standard deviation, and local variance. These priors are useful in various
image processing and computer vision tasks to capture local statistical
properties of images.
"""

__all__ = [
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
]

import torch
import torch.nn as nn
import torch.nn.functional as F


def image_local_mean(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculates the local mean of an image using a sliding window.
    
    Args:
        image (torch.Tensor): Image as a torch.Tensor of shape (B, C, H, W)
            in [0.0, 1.0].
        patch_size (int): Size of the sliding window. Defaults to 5.
        
    Returns:
        torch.Tensor: Local mean with similar type and format as the input image.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    return patches.mean(dim=(4, 5))


def image_local_variance(image: torch.Tensor, patch_size: int = 5) -> torch.Tensor:
    """Calculates the local variance of an image using a sliding window.

    Args:
        image (torch.Tensor): Image as a torch.Tensor of shape (B, C, H, W)
            in [0.0, 1.0].
        patch_size (int): Size of the sliding window. Defaults to 5.

    Returns:
        torch.Tensor: Local variance with similar type and format as the input
            image.
    """
    padding = patch_size // 2
    image   = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    patches = image.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
    mean    = patches.mean(dim=(4, 5))
    return ((patches - mean.unsqueeze(4).unsqueeze(5)) ** 2).mean(dim=(4, 5))


def image_local_stddev(
    image     : torch.Tensor,
    patch_size: int   = 5,
    eps       : float = 1e-9,
) -> torch.Tensor:
    """Calculates the local standard deviation of an image using a sliding
    window.

    Args:
        image (torch.Tensor): Image as a torch.Tensor of shape (B, C, H, W)
            in [0.0, 1.0].
        patch_size (int): Size of the sliding window. Defaults to 5.
        eps (float): Small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        torch.Tensor: Local standard deviation with similar type and format as
            the input image.
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
    """A class to calculate the local mean of an image using a sliding window.
    
    Attributes:
        patch_size (int): Size of the sliding window. Defaults to 5.
    """
    
    def __init__(self, patch_size: int = 5):
        """Initializes the ImageLocalMean module.
        
        Args:
            patch_size (int): Size of the sliding window. Defaults to 5.
        """
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Calculate the local mean of the input image.
        
        Args:
            image (torch.Tensor): Input image as a torch.Tensor of shape
                (B, C, H, W) in [0.0, 1.0].
                
        Returns:
            torch.Tensor: Local mean with similar type and format as the input
                image.
        """
        return image_local_mean(image, self.patch_size)


class ImageLocalVariance(nn.Module):
    """A class to calculate the local variance of an image using a sliding window.
    
    Attributes:
        patch_size (int): Size of the sliding window. Defaults to 5.
    """
    
    def __init__(self, patch_size: int = 5):
        """Initializes the ImageLocalVariance module.
        
        Args:
            patch_size (int): Size of the sliding window. Defaults to 5.
        """
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Calculate the local variance of the input image.
        
        Args:
            image (torch.Tensor): Input image as a torch.Tensor of shape
                (B, C, H, W) in [0.0, 1.0].
                
        Returns:
            torch.Tensor: Local variance with similar type and format as the
                input image.
        """
        return image_local_variance(image, self.patch_size)


class ImageLocalStdDev(nn.Module):
    """A class to calculate the local standard deviation of an image using a
    sliding window.
    
    Attributes:
        patch_size (int): Size of the sliding window. Defaults to 5.
        eps (float): Small value to avoid division by zero. Defaults to 1e-9.
    """
    
    def __init__(self, patch_size: int = 5, eps: float = 1e-9):
        """Initializes the ImageLocalStdDev module.
        
        Args:
            patch_size (int): Size of the sliding window. Defaults to 5.
            eps (float): Small value to avoid division by zero. Defaults to 1e-9.
        """
        super().__init__()
        self.patch_size = patch_size
        self.eps        = eps
    
    def forward(self, image):
        """Calculate the local standard deviation of the input image.
        
        Args:
            image (torch.Tensor): Input image as a torch.Tensor of shape
                (B, C, H, W) in [0.0, 1.0].
                
        Returns:
            torch.Tensor: Local standard deviation with similar type and format
                as the input image.
        """
        return image_local_stddev(image, self.patch_size, self.eps)
