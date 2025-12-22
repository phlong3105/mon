#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention-based image priors.

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


# --- Functions ---
def brightness_attention_map(
    image      : torch.Tensor,
    gamma      : float = 2.5,
    kernel_size: int   = None
) -> torch.Tensor:
    """Get the Brightness Attention Map (BAM) prior to an RGB image.

    This is a self-attention map extracted from the V-channel of a low-light
    image, multiplied to convolutional activations of all layers in the
    enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing contrast in
    dark regions effectively.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, 3, H, W) with pixel
            values in the range [0, 1].
        gamma: Parameter controlling the curvature of the map. Defaults to 2.5.
        kernel_size: Window size for denoising operation. Defaults to None.
        
    Returns:
        The Brightness Attention Map as a torch.Tensor of shape (B, 1, H, W)
        with pixel values in the range [0, 1].
    """
    if kernel_size:
        image = kornia.filters.median_blur(image, kernel_size)
        
    hsv = kornia.color.rgb_to_hsv(image)
    v   = hsv[:, 2:3, :, :]  # Extract the V-channel (brightness)
    bam = torch.pow((1 - v), gamma)
    return bam


# --- Modules ---
class BrightnessAttentionMap(nn.Module):
    """A module that computes the Brightness Attention Map (BAM) prior.

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
        """Initialize the BrightnessAttentionMap instance.
        
        Args:
            gamma: Parameter controlling the curvature of the map. Defaults to 2.5.
            kernel_size: Window size for denoising operation. Defaults to None.
        """
        super().__init__()
        self.gamma       = gamma
        self.kernel_size = kernel_size
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Get the Brightness Attention Map (BAM) prior from an RGB image.
        
        Args:
            image: An RGB image as a torch.Tensor of shape (B, 3, H, W) with
                pixel values in the range [0, 1].
                
        Returns:
            The Brightness Attention Map as a torch.Tensor of shape (B, 1, H, W)
            with pixel values in the range [0, 1].
        """
        return brightness_attention_map(image, self.gamma, self.kernel_size)
