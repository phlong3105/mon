#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image quality assessment metrics.

This module provides functions and classes to evaluate the quality of images
based on various criteria such as exposedness, contrast, and saturation.
"""

from __future__ import annotations

__all__ = [
    "ImageQualityAssessment",
]

import torch
from torch import nn, Tensor


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
        pool_size: int = 25,
        eps: float = 1e-6,
    ):
        """Initialize a new instance.

        Args:
            exposed_level (float): Target exposedness level. Defaults to 0.5.
            pool_size (int): Size of the pooling window for local statistics.
                Defaults to 25.
            eps (float): Small constant for numerical stability. Defaults to 1e-6.
        """
        super().__init__()
        # Assign attributes
        self.exposed_level = exposed_level
        self.eps = eps

        # Consolidate pooling to avoid repeated padding operations
        self.pad = nn.ReflectionPad2d(pool_size // 2)
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Compute the IQA score for input.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: IQA tensor of shape (B, 1, 1, 1) with values ranging from
                0.0 to 1.0.
        """
        # Saturation (Varying intensities across channels)
        max_rgb, _ = torch.max(x, dim=1, keepdim=True)
        min_rgb, _ = torch.min(x, dim=1, keepdim=True)
        saturation = (max_rgb - min_rgb + self.eps) / (max_rgb + self.eps)

        # Local Statistics (Using shared padded input)
        x_padded = self.pad(x)
        mu = self.avg_pool(x_padded)  # E[X]
        mu2 = self.avg_pool(x_padded ** 2)  # E[X^2]

        # Average across channels for local illumination/contrast
        mu_mean = mu.mean(dim=1, keepdim=True)

        # Exposedness (Distance from target level)
        exposedness = torch.abs(mu_mean - self.exposed_level) + self.eps

        # Contrast (Local Variance: Var = E[X^2] - E[X]^2)
        # Using channel-wise mean of variance for structural contrast
        contrast = (mu2 - mu ** 2).mean(dim=1, keepdim=True)

        # Final Score Calculation
        # Reduce spatial dimensions to get a per-image score
        quality_map = (saturation * contrast) / exposedness
        return quality_map.mean(dim=[1, 2, 3], keepdim=True)


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
