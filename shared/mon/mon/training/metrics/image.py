#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image quality assessment metrics."""

__all__ = [
    "ImageQualityAssessment",
    "scale_gt_mean",
]

from typing import Union

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn


def scale_gt_mean(
    image : Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """Scales image to match target's mean intensity.

    Args:
        image: RGB image as a ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in range :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in range :math:`[0, 255]`).
        target: Target image of same type as ``image``.
    
    Returns:
        Scaled image matching target's mean.
    
    Raises:
        TypeError: If ``image`` and ``target`` types differ.
    
    References:
        - Code: https://github.com/Fediory/HVI-CIDNet/blob/master/measure.py
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


class ImageQualityAssessment(nn.Module):
    """Assesses image quality based on exposedness, contrast, and saturation.

    Args:
        exposed_level: Target exposure level. Default: ``0.5``.
        pool_size: Size of pooling window. Default: ``25``.

    References:
        - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
    """

    def __init__(self, exposed_level: float = 0.5, pool_size: int = 25):
        super().__init__()
        self.exposed_level = exposed_level
        self.pool_size     = pool_size
        self.mean_pool     = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(self.pool_size // 2),
            torch.nn.AvgPool2d(self.pool_size, stride=1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        max_rgb     = torch.max(images, dim=1, keepdim=True)[0]
        min_rgb     = torch.min(images, dim=1, keepdim=True)[0]
        saturation  = (max_rgb - min_rgb + 1 / 255.0) / (max_rgb + 1 / 255.0)
        mean_rgb    = self.mean_pool(images).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + 1 / 255.0
        contrast    = self.mean_pool(images * images).mean(dim=1, keepdim=True) - mean_rgb ** 2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
