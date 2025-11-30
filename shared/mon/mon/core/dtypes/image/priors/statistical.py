#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for statistical image priors.

This module provides functions to compute various statistical priors for images,
including blur spot prior, bright channel prior, bright spot prior, and dark
channel prior.
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
    """Detects blur spots in an image based on variance of Laplacian filtered
    grayscale image.
    
    Args:
        image: Image as a numpy.ndarray in BGR format with shape (H, W, 3) with
            pixel values in the range [0, 255].
        threshold: Variance threshold to determine blur. Defaults to 250.
        
    Returns:
        bool: True if blur spots are detected (variance < threshold), False
            otherwise.
    
    Raises:
        TypeError: If image is not a numpy.ndarray.
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
    """Detects bright spots in an image based on variance of binary thresholded
    grayscale image.
    
    Args:
        image: Image as a numpy.ndarray in BGR format with shape (H, W, 3) with
            pixel values in the range [0, 255].
    
    Returns:
        bool: True if bright spots are detected (5000 < variance < 8500), False
            otherwise.
    
    Raises:
        TypeError: If image is not a numpy.ndarray.
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


def bright_channel_prior(
    image: torch.Tensor | np.ndarray,
    ksize: int
) -> torch.Tensor | np.ndarray:
    """Gets bright channel prior from an RGB image.
    
    Args:
        image (torch.Tensor or numpy.ndarray): An RGB image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in the range [0.0, 1.0])
            or numpy.ndarray (i.e., of shape (H, W, C) with pixel values in the
            range [0, 255]).
        ksize (int): Window size.
        
    Returns:
        torch.Tensor or numpy.ndarray: Bright channel prior with similar type
            and format as the input image.
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


def dark_channel_prior(
    image: torch.Tensor | np.ndarray,
    ksize: int
) ->  torch.Tensor | np.ndarray:
    """Gets dark channel prior from an RGB image.
    
    Args:
        image (torch.Tensor or numpy.ndarray): An RGB image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in the range [0.0, 1.0])
            or numpy.ndarray (i.e., of shape (H, W, C) with pixel values in the
            range [0, 255]).
        ksize (int): Window size.
        
    Returns:
        torch.Tensor or numpy.ndarray: Dark channel prior with similar type
            and format as the input image.
            
    Raises:
        ValueError: If ``image`` is neither a torch.Tensor nor a numpy.ndarray.
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


def dark_channel_prior_paper(
    image: torch.Tensor | np.ndarray,
    ksize: int
) ->  torch.Tensor | np.ndarray:
    """Gets dark channel prior from an RGB image as per the original paper.
    
    Args:
        image (torch.Tensor or numpy.ndarray): An RGB image as a torch.Tensor
            (i.e., of shape (B, C, H, W) with pixel values in the range [0.0, 1.0])
            or numpy.ndarray (i.e., of shape (H, W, C) with pixel values in the
            range [0, 255]).
        ksize (int): Window size.
        
    Returns:
        torch.Tensor or numpy.ndarray: Dark channel prior with similar type
            and format as the input image.
    """
    m, n, _ = image.shape
    w       = ksize
    padded  = np.pad(image, ((w // 2, w // 2), (w // 2, w // 2), (0, 0)), "edge")
    dcp     = np.zeros((m, n))
    for i, j in np.ndindex(dcp.shape):
        dcp[i, j] = np.min(padded[i:i + w, j:j + w, :])  # CVPR09, eq.5
    return dcp
