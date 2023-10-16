#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements vision priors for images and videos."""

from __future__ import annotations

__all__ = [
    "get_bright_channel_prior",
    "get_dark_channel_prior",
]

import cv2
import numpy as np

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

# endregion


# region Non-Local Self-Similarity Prior

# endregion


# region Physical Prior

# endregion


# region Temporal Prior

# endregion
