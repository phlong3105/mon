#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for attention priors.

This module implements attention-based image priors used in computer vision
and image processing tasks. These priors help in enhancing image quality by
focusing on important regions of the image, such as brightness attention maps
"""

__all__ = [
    "BrightnessAttentionMap",
    "brightness_attention_map",
]

import kornia
import torch
import torch.nn as nn


def brightness_attention_map(
    image      : torch.Tensor,
    gamma      : float = 2.5,
    kernel_size: int   = None
) -> torch.Tensor:
    """Gets the Brightness Attention Map (BAM) prior from an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light
    image, multiplied to convolutional activations of all layers in the
    enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing contrast in
    dark regions effectively.

    Args:
        image (torch.Tensor): Input RGB image as torch.Tensor of shape
            (B, 3, H, W) with pixel values in [0.0, 1.0].
        gamma (float): Parameter controlling the curvature of the map. Defaults
            to 2.5.
        kernel_size (int): Window size for denoising operation. Defaults to None.
        
    Returns:
        torch.Tensor: Brightness Attention Map (BAM) as torch.Tensor of shape
            (B, 1, H, W) with pixel values in [0.0, 1.0].
    """
    if kernel_size:
        image = kornia.filters.median_blur(image, kernel_size)
        # image = kornia.filters.bilateral_blur(image, denoise_ksize, 0.1, (1.5, 1.5))
        
    hsv = kornia.color.rgb_to_hsv(image)
    v   = hsv[:, 2:3, :, :]  # Extract the V-channel (brightness)
    bam = torch.pow((1 - v), gamma)
    return bam


class BrightnessAttentionMap(nn.Module):
    """Gets the Brightness Attention Map (BAM) prior from an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light
    image, multiplied to convolutional activations of all layers in the
    enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing contrast in
    dark regions effectively.
    
    Attributes:
        gamma (float): Parameter controlling the curvature of the map.
        kernel_size (int): Window size for denoising operation.
    """
    
    def __init__(self, gamma: float = 2.5, kernel_size: int = None):
        """Initializes the BrightnessAttentionMap instance.
        
        Args:
            gamma (float): Parameter controlling the curvature of the map.
                Defaults to 2.5.
            kernel_size (int): Window size for denoising operation. Defaults to
                None.
        """
        super().__init__()
        self.gamma       = gamma
        self.kernel_size = kernel_size
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Gets the Brightness Attention Map (BAM) prior from an RGB image.
        
        Args:
            image (torch.Tensor): Input RGB image as torch.Tensor of shape
                (B, 3, H, W) with pixel values in [0.0, 1.0].
                
        Returns:
            torch.Tensor: Brightness Attention Map (BAM) as torch.Tensor of
                shape (B, 1, H, W) with pixel values in [0.0, 1.0].
        """
        return brightness_attention_map(image, self.gamma, self.kernel_size)
