#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements vision priors for images and videos."""

from __future__ import annotations

__all__ = [
    "get_bright_channel_prior",
    "get_dark_channel_prior",
    "get_guided_brightness_enhancement_map_prior",
]

import cv2
import kornia
import numpy as np
import torch

from mon.vision import core

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Image Feature Prior: Intensity & Gradient

def get_bright_channel_prior(
    input     : np.ndarray,
    patch_size: int | tuple[int, int],
) -> np.ndarray:
    """Get the bright channel prior from an image.
    
    Args:
        input: An image in :math:`[H, W, C]` format.
        patch_size: Window size.
        
    Returns:
        An :class:`numpy.ndarray` bright channel as prior.
    """
    # b, g, r      = cv2.split(input)
    # dark_channel = cv2.max(cv2.min(r, g), b)
    dark_channel = np.max(input, axis=2)
    patch_size   = core.to_2tuple(patch_size)
    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, patch_size)
    dcp          = cv2.erode(dark_channel, kernel)
    return dcp


def get_dark_channel_prior(
    input     : np.ndarray,
    patch_size: int | tuple[int, int],
) -> np.ndarray:
    """Get the dark channel prior from an image.
    
    Args:
        input: An image in :math:`[H, W, C]` format.
        patch_size: Window size.
        
    Returns:
        An :class:`numpy.ndarray` dark channel as prior.
    """
    # b, g, r      = cv2.split(input)
    # dark_channel = cv2.min(cv2.min(r, g), b)
    patch_size   = core.to_2tuple(patch_size)
    dark_channel = np.min(input, axis=2)
    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, patch_size)
    dcp          = cv2.erode(dark_channel, kernel)
    return dcp


def get_guided_brightness_enhancement_map_prior(
    input: torch.Tensor | np.ndarray,
    gamma: float = 2.5,
) -> torch.Tensor | np.ndarray:
    """Get the Guided Brightness Enhancement Map (GBEM) prior from an RGB image.
    
    This is a self-attention map extracted from the V-channel of a low-light
    image. This map is multiplied to convolutional activations of all layers in
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing the contrast
    in the dark regions effectively.
    
    Equation: :math:`I_{attn} = (1 - I_{V})^{\gamma}`, where
    :math:`\gamma \geq 1`.
    
    Args:
        input: An image.
        It can be a :class:`torch.Tensor` or :class:`np.ndarray` and in
            :math:`[N, C, H, W]` or :math:`[H, W, C]` format.
        gamma: A parameter controls the curvature of the map.
        
    Returns:
        An :class:`numpy.ndarray` brightness enhancement map as prior.
    """
    if isinstance(input, torch.Tensor):
        hsv  = kornia.color.rgb_to_hsv(input)
        v    = core.get_channel(input=hsv, index=(2, 3), keep_dim=True)  # hsv[:, 2:3, :, :]
        attn = torch.pow((1 - v), gamma)
    elif isinstance(input, np.ndarray):
        hsv  = cv2.cvtColor(input, cv2.COLOR_RGB2HSV)
        v    = core.get_channel(input=hsv, index=(2, 3), keep_dim=True)  # hsv[:, :, 2:3]
        attn = np.power((1 - v), gamma)
    else:
        raise TypeError
    return attn

# endregion


# region Non-Local Self-Similarity Prior

# endregion


# region Physical Prior

# endregion


# region Temporal Prior

# endregion
