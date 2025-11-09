#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements image statistical priors.

This category encompasses assumptions about image properties based on statistical data
(data-driven).
"""

__all__ = [
    "blur_spot_prior",
    "bright_channel_prior",
    "bright_spot_prior",
    "dark_channel_prior",
    "dark_channel_prior_paper",
]

from typing import Union

import cv2
import kornia
import numpy as np
import torch


def blur_spot_prior(image: np.ndarray, threshold: int = 250) -> bool:
    """Detects blur in an image based on Laplacian variance and bright spot thresholding.

    Args:
        image: Image as a ``numpy.ndarray`` of shape :math:`(H, W, C)`
            in :math:`[0, 255]`.
        threshold: Variance threshold for blur detection. Default: ``250``.

    Returns:
        ``True`` if the image is blurry (Laplacian variance < ``threshold``),
        ``False`` otherwise.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"``image`` must be numpy.ndarray, got {type(image)}.")
    
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
    """Detects bright spots in an image based on variance of a binary thresholded
    grayscale image.

    Args:
        image: Image as a ``numpy.ndarray`` in BGR format with shape [H, W, 3].

    Returns:
        ``True`` if bright spots are detected (variance between 5000 and 8500),
        ``False`` otherwise.

    Raises:
        TypeError: If ``image`` is not a ``numpy.ndarray``.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"``image`` must be numpy.ndarray, got {type(image)}.")
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Calculate maximum intensity and variance
    binary_var = binary.var()
    # Check bright spot condition based on variance of binary image
    is_bright = True if 5000 < binary_var < 8500 else False
    return is_bright


def bright_channel_prior(image: Union[torch.Tensor, np.ndarray], ksize: int) ->  Union[torch.Tensor, np.ndarray]:
    """Gets bright channel prior from an RGB image.

    Args:
        image: An RGB image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
        ksize: Window size.

    Returns:
        Bright channel prior with similar type and format as the input ``image``.
    """
    if isinstance(image, torch.Tensor):
        bright_channel = torch.max(image, dim=1)[0]
        kernel         = torch.ones(ksize, ksize)
        bcp            = kornia.morphology.erosion(bright_channel, kernel)
    elif isinstance(image, np.ndarray):
        bright_channel = np.max(image, axis=2)
        kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        bcp            = cv2.erode(bright_channel, kernel)
    else:
        raise ValueError(f"``image`` must be torch.Tensor or numpy.ndarray, got {type(image)}.")
    return bcp


def dark_channel_prior(image: Union[torch.Tensor, np.ndarray], ksize: int) ->  Union[torch.Tensor, np.ndarray]:
    """Gets dark channel prior from an RGB image.

    Args:
        image: An RGB image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
        ksize: Window size.

    Returns:
        Dark channel prior with similar type and format as the input ``image``.
    """
    if isinstance(image, torch.Tensor):
        dark_channel = torch.min(image, dim=1)[0]
        kernel       = torch.ones(ksize, ksize)
        dcp          = kornia.morphology.erosion(dark_channel, kernel)
    elif isinstance(image, np.ndarray):
        dark_channel = np.min(image, axis=2)
        kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        dcp          = cv2.erode(dark_channel, kernel)
    else:
        raise ValueError(f"``image`` must be torch.Tensor or numpy.ndarray, got {type(image)}.")
    return dcp


def dark_channel_prior_paper(image: Union[torch.Tensor, np.ndarray], ksize: int) ->  Union[torch.Tensor, np.ndarray]:
    """Gets dark channel prior from an RGB image (from paper).

    Args:
        image: An RGB image as a
            ``torch.Tensor`` (i.e., of shape :math:`(B, C, H, W)` in :math:`[0.0, 1.0]`)
            or ``numpy.ndarray`` (i.e., of shape :math:`(H, W, C)` in :math:`[0, 255]`).
        ksize: Window size.

    Returns:
        Dark channel prior with similar type and format as the input ``image``.
    """
    m, n, _ = image.shape
    w       = ksize
    padded  = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), "edge")
    dcp     = np.zeros((m, n))
    for i, j in np.ndindex(dcp.shape):
        dcp[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return dcp
