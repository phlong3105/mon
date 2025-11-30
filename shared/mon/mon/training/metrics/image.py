#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for image quality assessment metrics.

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


def scale_gt_mean(
    image : torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """Scales image to match target's mean intensity.
    
    References:
        - Code: https://github.com/Fediory/HVI-CIDNet/blob/master/measure.py
        
    Args:
        image (torch.Tensor or np.ndarray): Input image to be scaled.
        target (torch.Tensor or np.ndarray): Target image for mean intensity
            reference.
    
    Returns:
        torch.Tensor or np.ndarray: Scaled image with mean intensity matching
            the target.
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
    """A class for Image Quality Assessment (IQA) metric.

    References:
        - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/iqa.py
    
    Attributes:
        exposed_level (float): Ideal exposure level.
        pool_size (int): Size of the pooling kernel.
        mean_pool (torch.nn.Sequential): Mean pooling layer.
    """

    def __init__(self, exposed_level: float = 0.5, pool_size: int = 25):
        """Initializes the ImageQualityAssessment instance.
        
        Args:
            exposed_level (float): Ideal exposure level. Defaults to 0.5.
            pool_size (int): Size of the pooling kernel. Defaults to 25.
        """
        super().__init__()
        self.exposed_level = exposed_level
        self.pool_size     = pool_size
        self.mean_pool     = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(self.pool_size // 2),
            torch.nn.AvgPool2d(self.pool_size, stride=1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Computes the IQA score for input images.
        
        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W) with pixel
                values in the range [0.0, 1.0].
        
        Returns:
            torch.Tensor: IQA scores of shape (B, 1, 1, 1).
        """
        max_rgb     = torch.max(images, dim=1, keepdim=True)[0]
        min_rgb     = torch.min(images, dim=1, keepdim=True)[0]
        saturation  = (max_rgb - min_rgb + 1 / 255.0) / (max_rgb + 1 / 255.0)
        mean_rgb    = self.mean_pool(images).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + 1 / 255.0
        contrast    = self.mean_pool(images * images).mean(dim=1, keepdim=True) - mean_rgb ** 2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)
