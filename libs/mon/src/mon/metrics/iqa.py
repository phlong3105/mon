#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Quality Assessment Metrics.

This module provides functions and classes to evaluate the quality of images
based on various criteria such as exposedness, contrast, and saturation.
"""

from __future__ import annotations

__all__ = [
    "GeometricImageQualityScore",
    "ImageQualityAssessment",
]

from pyiqa.archs import _lazy_import_arch
import pyiqa.default_model_configs
import torch
from pyiqa.archs.lpips_arch import LPIPS
from pyiqa.archs.psnr_arch import PSNR
from pyiqa.archs.ssim_arch import SSIM
from pyiqa.utils.registry import ARCH_REGISTRY
from torch import nn, Tensor

from mon.core import METRICS
from .base import Metric


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def __register_metrics():
    """Register all metric classes from the given module and its submodules."""
    for k, v in pyiqa.default_model_configs.DEFAULT_CONFIGS.items():
        try:
            type = v["metric_opts"]["metric_opts"]
            _lazy_import_arch(type)
            func = ARCH_REGISTRY.get(type)
        except Exception as e:
            func = None

        METRICS[k] = {
            "name": k,
            "module": func,
            "metric_opts": v["metric_opts"],
            "metric_mode": v["metric_mode"],
            "lower_better": v.get("lower_better", False),
            "score_range": v["score_range"],
        }


__register_metrics()
del __register_metrics
METRICS.sort()

# endregion


# ==============================================================================
# region NON-REFERENCE IAQ
# ==============================================================================

class ImageQualityAssessment(Metric):
    """Image Quality Assessment (IQA) metric.

    References:
        - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
    """

    metric_opts: dict = {}
    metric_mode: str = "NR"             # ["FR" or "NR"]
    lower_better: bool = False          # True if lower score is better
    score_range: tuple[float, float] = (0.0, 1.0)  # (min, max)

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        exposed_level: float = 0.5,
        pool_size: int = 25,
        eps: float = 1e-6,
        device: torch.device = torch.device("cpu")
    ):
        """Initialize a new instance.

        Args:
            exposed_level (float, optional): Target exposedness level.
                Defaults to 0.5.
            pool_size (int, optional): Size of the pooling window for local
                statistics. Defaults to 25.
            eps (float, optional): Small constant for numerical stability.
                Defaults to 1e-6.
        """
        super().__init__(device=device)

        # Assign attributes
        self.exposed_level = exposed_level
        self.eps = eps

        # Consolidate pooling to avoid repeated padding operations
        self.pad = nn.ReflectionPad2d(pool_size // 2)
        self.avg_pool = nn.AvgPool2d(pool_size, stride=1)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        """Compute the IQA score for input.

        Args:
            input (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: IQA tensor of shape (B, 1, 1, 1) with values ranging from
                0.0 to 1.0.
        """
        # Saturation (Varying intensities across channels)
        max_rgb, _ = torch.max(input, dim=1, keepdim=True)
        min_rgb, _ = torch.min(input, dim=1, keepdim=True)
        saturation = (max_rgb - min_rgb + self.eps) / (max_rgb + self.eps)

        # Local Statistics (Using shared padded input)
        x_padded = self.pad(input)
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
# region COMBINED IAQ
# ==============================================================================

@METRICS.register(name="ciqs")
class CompositeImageQualityScore(Metric):
    """Composite Image Quality Score (CIQS) metric."""

    metric_opts: dict = {}
    metric_mode: str = "FR"             # ["FR" or "NR"]
    lower_better: bool = False          # True if lower score is better
    score_range: str = "0, 1"           # (min, max)

    # --- Lifecycle & Initialization ---
    def __init__(self, device: torch.device = torch.device("cpu")):
        """Initialize a new instance."""
        super().__init__(device=device)

        # Define components
        self.psnr = PSNR().to(device)
        self.ssim = SSIM().to(device)
        self.lpips = LPIPS().to(device)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Calculate the metric between ``input`` and ``target``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            target (Tensor): Target (ground truth) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Metric value.
        """
        # Compute individual metrics
        psnr = self.psnr(input, target)
        ssim = self.ssim(input, target)
        lpips = self.lpips(input, target)

        # Normalize metrics
        epsilon = 1e-6
        psnr_min = torch.zeros_like(psnr)
        psnr_max = self.psnr(target, target)
        psnr = (psnr - psnr_min) / (psnr_max - psnr_min + epsilon)

        lpips = 1.0 - lpips

        # Clamp values to avoid absolute zero
        psnr = torch.clamp(psnr, 0.0, 1.0)
        ssim = torch.clamp(ssim, 0.0, 1.0)
        lpips = torch.clamp(lpips, 0.0, 1.0)

        # Calculate the combined metric
        ciqs = (psnr + ssim + lpips) / 3.0
        return ciqs


@METRICS.register(name="giqs")
class GeometricImageQualityScore(Metric):
    """Geometric Image Quality Score (GIQS) metric."""

    metric_opts: dict = {}
    metric_mode: str = "FR"             # ["FR" or "NR"]
    lower_better: bool = False          # True if lower score is better
    score_range: str = "0, 1"           # (min, max)

    # --- Lifecycle & Initialization ---
    def __init__(self, device: torch.device = torch.device("cpu")):
        """Initialize a new instance."""
        super().__init__(device=device)

        # Define components
        self.psnr = PSNR().to(device)
        self.ssim = SSIM().to(device)
        self.lpips = LPIPS().to(device)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        """Calculate the metric between ``input`` and ``target``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.
            target (Tensor): Target (ground truth) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Metric value.
        """
        # Compute individual metrics
        psnr = self.psnr(input, target)
        ssim = self.ssim(input, target)
        lpips = self.lpips(input, target)

        # Normalize metrics
        epsilon = 1e-6
        psnr_min = torch.zeros_like(psnr)
        psnr_max = self.psnr(target, target)
        psnr = (psnr - psnr_min) / (psnr_max - psnr_min + epsilon)

        lpips = 1.0 - lpips

        # Clamp values to avoid absolute zero
        psnr = torch.clamp(psnr, 0.0, 1.0)
        ssim = torch.clamp(ssim, 0.0, 1.0)
        lpips = torch.clamp(lpips, 0.0, 1.0)

        # Calculate the combined metric
        giqs = torch.pow((psnr * ssim * lpips), (1.0 / 3.0))
        return giqs

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
