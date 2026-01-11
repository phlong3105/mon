#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image quality assessment metrics.

This module provides functions and classes to evaluate the quality of images
based on various criteria such as exposedness, contrast, and saturation.
"""

from __future__ import annotations

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
# region NON-REFERENCE IAQ
# ==============================================================================

# --- Perceptual Metrics ---

class ImageQualityAssessment(nn.Module):
    """Image Quality Assessment (IQA) metric.

    References:
        - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        exposed_level: float = 0.5,
        pool_size    : int   = 25,
        eps          : float = 1e-6
    ):
        """Initialize a new instance.
        
        Args:
            exposed_level: Ideal exposure level. Defaults to 0.5.
            pool_size: Size of the pooling kernel. Defaults to 25.
            eps: Small constant for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        self.exposed_level = exposed_level
        self.eps = eps
        
        # Consolidate pooling to avoid repeated padding operations
        self.pad      = nn.ReflectionPad2d(pool_size // 2)
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1)
        
    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the IQA score for input.
        
        Args:
            x: Input image, formatted as a torch.Tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
        
        Returns:
            IQA scores, formatted as a torch.Tensor of shape (B, 1, 1, 1).
        """
        # Saturation (Varying intensities across channels)
        max_rgb, _ = torch.max(x, dim=1, keepdim=True)
        min_rgb, _ = torch.min(x, dim=1, keepdim=True)
        saturation = (max_rgb - min_rgb + self.eps) / (max_rgb + self.eps)
        
        # Local Statistics (Using shared padded input)
        x_padded = self.pad(x)
        mu       = self.avg_pool(x_padded)     # E[X]
        mu2      = self.avg_pool(x_padded**2)  # E[X^2]
        
        # Average across channels for local illumination/contrast
        mu_mean = mu.mean(dim=1, keepdim=True)
        
        # Exposedness (Distance from target level)
        exposedness = torch.abs(mu_mean - self.exposed_level) + self.eps
        
        # Contrast (Local Variance: Var = E[X^2] - E[X]^2)
        # Using channel-wise mean of variance for structural contrast
        contrast = (mu2 - mu**2).mean(dim=1, keepdim=True)
        
        # Final Score Calculation
        # Reduce spatial dimensions to get a per-image score
        quality_map = (saturation * contrast) / exposedness
        return quality_map.mean(dim=[1, 2, 3], keepdim=True)
        
        # TODO: Delete later
        """
        max_rgb     = torch.max(x, dim=1, keepdim=True)[0]
        min_rgb     = torch.min(x, dim=1, keepdim=True)[0]
        saturation  = (max_rgb - min_rgb + 1 / 255.0) / (max_rgb + 1 / 255.0)
        mean_rgb    = self.mean_pool(x).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + 1 / 255.0
        contrast    = self.mean_pool(x * x).mean(dim=1, keepdim=True) - mean_rgb ** 2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
        """

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

def scale_gt_mean(
    image : torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    eps   : float = 1e-6
) -> torch.Tensor | np.ndarray:
    """Scale image to match target's mean intensity.
    
    References:
        - Code: https://github.com/Fediory/HVI-CIDNet/blob/master/measure.py
        
    Args:
        image: Input image, formatted as a torch.Tensor of shape
            (B, C, H, W) and values ranging from 0.0 to 1.0; or as a np.ndarray
            of shape (H, W, C) with values ranging from 0 to 255.
        target: Target image, formatted as a torch.Tensor of shape
            (B, C, H, W) and values ranging from 0.0 to 1.0; or as a np.ndarray
            of shape (H, W, C) with values ranging from 0 to 255.
        eps: Small constant for numerical stability. Defaults to 1e-6.
        
    Returns:
        Scaled image with mean intensity matching the target.
        
    Raises:
        TypeError: If input types are not torch.Tensor or np.ndarray.
    """
    if isinstance(image, torch.Tensor) and isinstance(target, torch.Tensor):
        mean_image  = kornia.color.rgb_to_grayscale(image).mean()
        mean_target = kornia.color.rgb_to_grayscale(target).mean()
        scale       = (mean_target + eps) / (mean_image + eps)
        return torch.clamp(image * scale, 0, 1)
    elif isinstance(image, np.ndarray) and isinstance(target, np.ndarray):
        mean_image  = cv2.cvtColor(image,  cv2.COLOR_RGB2GRAY).mean()
        mean_target = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).mean()
        scale       = (mean_target + eps) / (mean_image + eps)
        return np.clip(image * scale, 0, 255)
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, but got {type(image)} and {type(target)}.")

# endregion
