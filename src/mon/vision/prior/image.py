#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements image priors."""

from __future__ import annotations

__all__ = [
    "detect_blur_spot",
    "detect_bright_spot",
    "get_atmosphere_prior",
    "get_bright_channel_prior",
    "get_dark_channel_prior",
    "get_dark_channel_prior_02",
    "get_guided_brightness_enhancement_map_prior",
]

import cv2
import kornia
import numpy as np
import torch

from mon import core
from mon.core import _size_2_t
from mon.vision import filtering


# region Intensity & Gradient

def detect_blur_spot(
    input    : np.ndarray,
    threshold: int  = 250,
    verbose  : bool = False,
) -> bool:
    # Convert image to grayscale
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Apply Laplacian filter for edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    # Calculate maximum intensity and variance
    laplacian_variance = laplacian.var()
    # Check blur condition based on variance of Laplacian image
    is_blur = False
    if laplacian_variance < threshold:
        is_blur = True
    if verbose:
        text = "Blurry" if is_blur else "Not Blurry"
        cv2.putText(input, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Blur Spot", input)
        cv2.waitKey(0)
    return is_blur


def detect_bright_spot(
    input    : np.ndarray,
    threshold: int  = 250,
    verbose  : bool = False,
) -> bool:
    # Convert image to grayscale
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding for bright spot detection
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # Calculate maximum intensity and variance
    binary_variance = binary.var()
    # Check bright spot condition based on variance of binary image
    is_bright = False
    if 5000 < binary_variance < 8500:
        is_bright = True
    if verbose:
        text = "Bright Spot" if is_bright else "No Bright Spot"
        cv2.putText(input, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        cv2.imshow("Bright Spot", input)
        cv2.waitKey(0)
    return is_bright


def get_atmosphere_prior(
    input      : np.ndarray,
    kernel_size: int   = 15,
    p          : float = 0.0001,
):
    """Get the atmosphere light in the (RGB) image data.

    Args:
        input: An image in :math:`[H, W, C]` format.
        kernel_size: Window for the dark channel.
        p: Percentage of pixels for estimating the atmosphere light.

    Returns:
        A 3-element array containing atmosphere light ([0, L-1]) for each channel.
    """
    input      = input.transpose(1, 2, 0)
    # Reference CVPR09, 4.4
    dark       = get_dark_channel_prior_02(input, kernel_size=kernel_size)
    m, n       = dark.shape
    flat_i     = input.reshape(m * n, 3)
    flat_dark  = dark.ravel()
    search_idx = (-flat_dark).argsort()[:int(m * n * p)]  # find top M * N * p indexes
    # Return the highest intensity for each channel
    return np.max(flat_i.take(search_idx, axis=0), axis=0)


def get_bright_channel_prior(
    input      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t,
) -> torch.Tensor | np.ndarray:
    """Get the bright channel prior from an image.
    
    Args:
        input: A :class:`torch.Tensor` image in :math:`[N, C, H, W]` format, or
            a :class:`np.ndarray` image in :math:`[H, W, C]` format.
        kernel_size: Window size.

    Returns:
        A bright channel prior.
    """
    if isinstance(input, torch.Tensor):
        dark_channel   = torch.max(input, dim=1)[0]
        kernel         = torch.ones(kernel_size, kernel_size)
        bcp            = kornia.morphology.erosion(dark_channel, kernel)
    elif isinstance(input, np.ndarray):
        bright_channel = np.max(input, axis=2)
        kernel_size    = core.to_2tuple(kernel_size)
        kernel         = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        bcp            = cv2.erode(bright_channel, kernel)
    else:
        raise TypeError()
    return bcp


def get_dark_channel_prior(
    input      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t,
) -> torch.Tensor | np.ndarray:
    """Get the dark channel prior from an image.
    
    Args:
        input: A :class:`torch.Tensor` image in :math:`[N, C, H, W]` format, or
            a :class:`np.ndarray` image in :math:`[H, W, C]` format.
        kernel_size: Window size.
        
    Returns:
        A dark channel prior.
    """
    if isinstance(input, torch.Tensor):
        dark_channel = torch.min(input, dim=1)[0]
        kernel       = torch.ones(kernel_size, kernel_size)
        dcp          = kornia.morphology.erosion(dark_channel, kernel)
    elif isinstance(input, np.ndarray):
        dark_channel = np.min(input, axis=2)
        kernel_size  = core.to_2tuple(kernel_size)
        kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dcp          = cv2.erode(dark_channel, kernel)
    else:
        raise TypeError()
    return dcp


def get_dark_channel_prior_02(
    input      : torch.Tensor | np.ndarray,
    kernel_size: _size_2_t,
) -> np.ndarray:
    """Get the dark channel prior from an image.

    Args:
        input: A :class:`torch.Tensor` image in :math:`[N, C, H, W]` format, or
            a :class:`np.ndarray` image in :math:`[H, W, C]` format.
        kernel_size: Window size.

    Returns:
        A dark channel prior.
    """
    if isinstance(input, np.ndarray):
        m, n, _ = input.shape
        w       = kernel_size
        padded  = np.pad(input, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), "edge")
        dcp     = np.zeros((m, n))
        for i, j in np.ndindex(dcp.shape):
            dcp[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    else:
        raise TypeError()
    return dcp


def get_guided_brightness_enhancement_map_prior(
    input        : torch.Tensor | np.ndarray,
    gamma        : float            = 2.5,
    denoise_ksize: _size_2_t | None = None,
) -> torch.Tensor | np.ndarray:
    """Get the Guided Brightness Enhancement Map (G) prior from an RGB image.
    
    This is a self-attention map extracted from the V-channel of a low-light
    image. This map is multiplied to convolutional activations of all layers in
    the enhancement network. Brighter regions are given lower weights to avoid
    over-saturation, while preserving image details and enhancing the contrast
    in the dark regions effectively.
    
    Equation: :math:`I_{attn} = (1 - I_{V})^{\gamma}`, where :math:`\gamma \geq 1`.
    
    Args:
        input: An image.
        It can be a :class:`torch.Tensor` or :class:`np.ndarray` and in
            :math:`[N, C, H, W]` or :math:`[H, W, C]` format.
        gamma: A parameter controls the curvature of the map.
        denoise_ksize: Window size for de-noising operation. Default: ``None``.
        
    Returns:
        An :class:`numpy.ndarray` brightness enhancement map as prior.
    """
    if isinstance(input, torch.Tensor):
        if denoise_ksize is not None:
            # input = filtering.guided_filter(input, input, denoise_ksize)
            input = kornia.filters.median_blur(input, denoise_ksize)
        hsv  = kornia.color.rgb_to_hsv(input)
        v    = core.get_channel(input=hsv, index=(2, 3), keep_dim=True)  # hsv[:, 2:3, :, :]
        attn = torch.pow((1 - v), gamma)
    elif isinstance(input, np.ndarray):
        if denoise_ksize is not None:
            input = cv2.medianBlur(input, denoise_ksize)
        hsv = cv2.cvtColor(input, cv2.COLOR_RGB2HSV)
        if hsv.dtype != np.float64:
            hsv  = hsv.astype("float64")
            hsv /= 255.0
        v    = core.get_channel(input=hsv, index=(2, 3), keep_dim=True)  # hsv[:, :, 2:3]
        attn = np.power((1 - v), gamma)
    else:
        raise TypeError
    return attn

# endregion
