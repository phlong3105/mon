#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Priors.

This module implements priors used in images.
"""

from __future__ import annotations

__all__ = [
    "BoundaryAwarePrior",
    "BrightnessAttentionMap",
    "atmospheric_prior",
    "blur_spot_prior",
    "boundary_aware_prior",
    "bright_channel_prior",
    "bright_spot_prior",
    "brightness_attention_map",
    "dark_channel_prior",
    "dark_channel_prior_02",
]

import cv2
import kornia
import numpy as np
import torch
from torch import nn
from torch.nn.common_types import _size_2_t

from mon import core


# region Intensity & Gradient Prior

def atmospheric_prior(
    image      : np.ndarray,
    kernel_size: _size_2_t = 15,
    p          : float     = 0.0001
) -> np.ndarray:
    """Get the atmosphere light in RGB image.

    Args:
        image: An RGB image of type :obj:`numpy.ndarray` in ``[H, W, C]``
            format with data in the range ``[0, 255]``.
        kernel_size: Window for the dark channel. Default: ``15``.
        p: Percentage of pixels for estimating the atmosphere light.
            Default: ``0.0001``.
    
    Returns:
        A 3-element array containing atmosphere light ``([0, L-1])`` for each
        channel.
    """
    image      = image.transpose(1, 2, 0)
    # Reference CVPR09, 4.4
    dark       = dark_channel_prior_02(image=image, kernel_size=kernel_size)
    m, n       = dark.shape
    flat_i     = image.reshape(m * n, 3)
    flat_dark  = dark.ravel()
    search_idx = (-flat_dark).argsort()[:int(m * n * p)]  # find top M * N * p indexes
    # Return the highest intensity for each channel
    return np.max(flat_i.take(search_idx, axis=0), axis=0)


def blur_spot_prior(image: np.ndarray, threshold: int = 250) -> bool:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Calculate maximum intensity and variance
    laplacian_var = laplacian.var()
    # Check blur condition based on variance of Laplacian image
    is_blur = True if laplacian_var < threshold else False
    return is_blur


def bright_spot_prior(image: np.ndarray) -> bool:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Calculate maximum intensity and variance
    binary_var = binary.var()
    # Check bright spot condition based on variance of binary image
    is_bright = True if 5000 < binary_var < 8500 else False
    return is_bright


def bright_channel_prior(
    image      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t
) -> torch.Tensor | np.ndarray:
    """Get the bright channel prior from an RGB image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        kernel_size: Window size.

    Returns:
        A bright channel prior.
    """
    kernel_size = core.to_2tuple(kernel_size)
    if isinstance(image, torch.Tensor):
        bright_channel = torch.max(image, dim=1)[0]
        kernel         = torch.ones(kernel_size[0], kernel_size[0])
        bcp            = kornia.morphology.erosion(bright_channel, kernel)
    elif isinstance(image, np.ndarray):
        bright_channel = np.max(image, axis=2)
        kernel_size    = core.to_2tuple(kernel_size)
        kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        bcp            = cv2.erode(bright_channel, kernel)
    else:
        raise ValueError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`,"
                         f" but got {type(image)}.")
    return bcp


def dark_channel_prior(
    image      : torch.Tensor | np.ndarray,
    kernel_size: int
) -> torch.Tensor | np.ndarray:
    """Get the dark channel prior from an RGB image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        kernel_size: Window size.
        
    Returns:
        A dark channel prior.
    """
    kernel_size = core.to_2tuple(kernel_size)
    if isinstance(image, torch.Tensor):
        dark_channel = torch.min(image, dim=1)[0]
        kernel       = torch.ones(kernel_size[0], kernel_size[1])
        dcp          = kornia.morphology.erosion(dark_channel, kernel)
    elif isinstance(image, np.ndarray):
        dark_channel = np.min(image, axis=2)
        kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dcp          = cv2.erode(dark_channel, kernel)
    else:
        raise ValueError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`,"
                         f" but got {type(image)}.")
    return dcp


def dark_channel_prior_02(
    image      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t
) -> torch.Tensor | np.ndarray:
    """Get the dark channel prior from an RGB image.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        kernel_size: Window size.

    Returns:
        A dark channel prior.
    """
    m, n, _ = image.shape
    w       = kernel_size
    padded  = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), "edge")
    dcp     = np.zeros((m, n))
    for i, j in np.ndindex(dcp.shape):
        dcp[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return dcp


def boundary_aware_prior(
    image     : torch.Tensor | np.ndarray,
    eps       : float = 0.05,
    normalized: bool  = False
) -> torch.Tensor | np.ndarray:
    """Get the boundary prior from an RGB or grayscale image.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        eps: Threshold to remove weak edges. Default: ``0.05``.
        normalized: If ``True``, L1 norm of the kernel is set to ``1``.
            Default: ``False``.
        
    Returns:
        A boundary aware prior as a binary image.
    """
    if isinstance(image, torch.Tensor):
        gradient = kornia.filters.sobel(image, normalized=normalized, eps=1e-6)
        g_max    = torch.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
    elif isinstance(image, np.ndarray):
        if core.is_color_image(image):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        from mon.vision.filtering import sobel_filter
        gradient = sobel_filter(image, kernel_size=3)
        g_max    = np.max(gradient)
        gradient = gradient / g_max
        boundary = (gradient > eps).float()
        return boundary
    else:
        raise ValueError(f"Unsupported input type: {type(image)}.")
    return boundary


class BoundaryAwarePrior(nn.Module):
    """Get the boundary prior from an RGB or grayscale image.
    
    Args:
        eps: Threshold weak edges. Default: ``0.05``.
        normalized: If ``True``, L1 norm of the kernel is set to ``1``.
            Default: ``True``.
    """
    
    def __init__(self, eps: float = 0.05, normalized: bool = False):
        super().__init__()
        self.eps        = eps
        self.normalized = normalized
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return boundary_aware_prior(image, self.eps, self.normalized)

# endregion


# region Self-Attention Map

def brightness_attention_map(
    image        : torch.Tensor | np.ndarray,
    gamma        : float     = 2.5,
    denoise_ksize: _size_2_t = None,
) -> torch.Tensor:
    """Get the Brightness Attention Map (BAM) prior from an RGB image.
    
    This is a self-attention map extracted from the V-channel of a low-light
    image. This map is multiplied to convolutional activations of all layers in
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing the contrast
    in the dark regions effectively.
    
    Equation: `I_{attn} = (1 - I_{V})^{\gamma}`, where `\gamma \geq 1`.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[B, C, H, W]`` format with data in
                the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, C]`` format with data in the
                range ``[0, 255]``.
        gamma: A parameter controls the curvature of the map.
        denoise_ksize: Window size for de-noising operation. Default: ``None``.
        
    Returns:
        An :obj:`numpy.ndarray` brightness enhancement map as prior.
    """
    if isinstance(image, torch.Tensor):
        if denoise_ksize:
            image = kornia.filters.median_blur(image, denoise_ksize)
            # image = kornia.filters.bilateral_blur(image, denoise_ksize, 0.1, (1.5, 1.5))
        hsv = kornia.color.rgb_to_hsv(image)
        v   = core.get_image_channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, 2:3, :, :]
        bam = torch.pow((1 - v), gamma)
    elif isinstance(image, np.ndarray):
        if denoise_ksize:
            image = cv2.medianBlur(image, denoise_ksize)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if hsv.dtype != np.float64:
            hsv  = hsv.astype("float64")
            hsv /= 255.0
        v   = core.get_image_channel(image=hsv, index=(2, 3), keep_dim=True)  # hsv[:, :, 2:3]
        bam = np.power((1 - v), gamma)
    else:
        raise ValueError(f"Unsupported input type: {type(image)}.")
    return bam


class BrightnessAttentionMap(nn.Module):
    """Get the Brightness Attention Map (BAM) prior from an RGB image.
    
    This is a self-attention map extracted from the V-channel of a low-light
    image. This map is multiplied to convolutional activations of all layers in
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing the contrast
    in the dark regions effectively.
    
    Equation: `I_{attn} = (1 - I_{V})^{\gamma}`, where `\gamma \geq 1`.
    
    Args:
        gamma: A parameter controls the curvature of the map.
        denoise_ksize: Window size for de-noising operation. Default: ``None``.
    """
    
    def __init__(
        self,
        gamma        : float     = 2.5,
        denoise_ksize: _size_2_t = None
    ):
        super().__init__()
        self.gamma         = gamma
        self.denoise_ksize = denoise_ksize
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return brightness_attention_map(image, self.gamma, self.denoise_ksize)

# endregion
