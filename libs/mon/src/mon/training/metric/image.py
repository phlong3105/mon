#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image quality assessment metrics.

This module provides functions and classes to evaluate the quality of images
based on various criteria such as exposedness, contrast, and saturation.
"""

__all__ = [
    "ImageQualityAssessment",
    "scale_gt_mean",
]

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn


# ==============================================================================
# IMAGE PRE-PROCESSING & NORMALIZATION
# ==============================================================================

# --- Normalization ---
def scale_gt_mean(
    image : torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Scale image to match target's mean intensity.
    
    References:
        - Code: https://github.com/Fediory/HVI-CIDNet/blob/master/measure.py
        
    Args:
        image: Input image, formatted as a torch.Tensor of dimensions
            (B, C, H, W) and values ranging from 0.0 to 1.0; or as a np.ndarray
            of shape (H, W, C) with values ranging from 0 to 255.
        target: Target image, formatted as a torch.Tensor of dimensions
            (B, C, H, W) and values ranging from 0.0 to 1.0; or as a np.ndarray
            of shape (H, W, C) with values ranging from 0 to 255.
    
    Returns:
        Scaled image with mean intensity matching the target.
    """
    if isinstance(image, torch.Tensor) and isinstance(target, torch.Tensor):
        mean_image  = kornia.color.rgb_to_grayscale(image).mean()
        mean_target = kornia.color.rgb_to_grayscale(target).mean()
        image       = torch.clip(image * (mean_target / mean_image), 0, 1)
    elif isinstance(image, np.ndarray) and isinstance(target, np.ndarray):
        mean_image  = cv2.cvtColor(image,  cv2.COLOR_RGB2GRAY).mean()
        mean_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).mean()
        image       = np.clip(image * (mean_target / mean_image), 0, 255)
    else:
        raise TypeError(f"``image`` and ``target`` must be same type, "
                        f"got {type(image).__name__} and {type(target).__name__}.")
    return image


# ==============================================================================
# NON-REFERENCE QUALITY ASSESSMENT
# ==============================================================================

# --- Perceptual Metrics ---
class ImageQualityAssessment(nn.Module):
    """Image Quality Assessment (IQA) metric.

    References:
        - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
    
    Attributes:
        exposed_level (float): Ideal exposure level.
        pool_size (int): Size of the pooling kernel.
        mean_pool (nn.Sequential): Mean pooling layer.
    """

    def __init__(self, exposed_level: float = 0.5, pool_size: int = 25):
        """Initialize a new instance.
        
        Args:
            exposed_level: Ideal exposure level. Defaults to 0.5.
            pool_size: Size of the pooling kernel. Defaults to 25.
        """
        super().__init__()
        self.exposed_level = exposed_level
        self.pool_size     = pool_size
        self.mean_pool     = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(self.pool_size // 2),
            torch.nn.AvgPool2d(self.pool_size, stride=1)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the IQA score for input.
        
        Args:
            input: Input image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            IQA scores, formatted as a torch.Tensor of dimensions (B, 1, 1, 1).
        """
        max_rgb     = torch.max(input, dim=1, keepdim=True)[0]
        min_rgb     = torch.min(input, dim=1, keepdim=True)[0]
        saturation  = (max_rgb - min_rgb + 1 / 255.0) / (max_rgb + 1 / 255.0)
        mean_rgb    = self.mean_pool(input).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + 1 / 255.0
        contrast    = self.mean_pool(input * input).mean(dim=1, keepdim=True) - mean_rgb ** 2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
