#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image data complex operations.

This module provides higher-level logics that might involve multiple atomic
operations to manipulate the image data.
"""

__all__ = [
    "boundary_aware_prior",
    "brightness_attention_map",
    "pad_square",
    "pair_downsample",
    "split",
]

import kornia
import torch

from .meta import imgsz


# ==============================================================================
# GEOMETRIC OPS
# ==============================================================================

# ==============================================================================
# PRIORS
# ==============================================================================

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
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
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


def boundary_aware_prior(
    image      : torch.Tensor,
    eps        : float = 0.05,
    as_gradient: bool  = False,
    normalized : bool  = False,
) -> torch.Tensor:
    """Get the boundary prior from an RGB or grayscale image.

    Args:
        image: An RGB image as a torch.Tensor of shape (B, C, H, W) with pixel
            values in the range [0, 1].
        eps: Threshold to remove weak edges. Defaults to 0.05.
        as_gradient: If True, returns the gradient image instead of the binary
            boundary. Defaults to False.
        normalized: L1 norm of the kernel is set to 1 if True. Defaults to False.
    
    Returns:
        Boundary prior as binary map or gradient image.
    """
    image    = image.to(torch.float32)
    gradient = kornia.filters.sobel(image, normalized=normalized, eps=1e-6)
    g_max    = torch.max(gradient)
    gradient = gradient / g_max
    boundary = (gradient > eps).float()
    # Return boundary, gradient
    if as_gradient:
        return gradient
    else:
        return boundary
