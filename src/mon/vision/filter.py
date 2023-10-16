#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements smoothing filter."""

from __future__ import annotations

__all__ = [
    "guided_filter",
]

import cv2
import numpy as np

from mon.vision import core

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Edge

# endregion


# region Frequency

# endregion


# region Histogram

# endregion


# region Morphology

# endregion


# region Sharpening

# endregion


# region Smoothness

def guided_filter(
    input : np.ndarray,
    guide : np.ndarray,
    radius: int,
    eps   : float = 0.01,
) -> np.ndarray:
    """Guided filter implementation using :module:`cv2`.
    
    Guided filter is both effective and efficient in a great variety of
    computer vision and computer graphics applications, including edge-aware
    smoothing, detail enhancement, HDR compression, image matting/feathering,
    dehazing, joint upsampling, etc.
    
    Args:
        input: An image in :math:`[H, W, C]` format.
        guide: A guidance image with the same shape with :attr:`input`.
        radius: Radius of filter a.k.a patch size, window size, kernel size,
            etc. Commonly be ``1``, ``2``, ``4``, or ``8``.
        eps: Value controlling sharpness. Default: ``0.01``.
    
    Returns:
        A filtered image.
        
    References:
        - https://github.com/lisabug/guided-filter/blob/master/core/filter.py
        - https://github.com/lisabug/guided-filter/tree/master
        - https://github.com/wuhuikai/DeepGuidedFilter
    """
    mean_i  = cv2.boxFilter(input, cv2.CV_64F, (radius, radius))
    mean_g  = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
    mean_ig = cv2.boxFilter(input * guide, cv2.CV_64F, (radius, radius))
    cov_ig  = mean_ig - mean_i * mean_g
    mean_ii = cv2.boxFilter(input * input, cv2.CV_64F, (radius, radius))
    var_i   = mean_ii - mean_i * mean_i
    a       = cov_ig / (var_i + eps)
    b       = mean_i - a * mean_i
    mean_a  = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
    mean_b  = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))
    output  = mean_a * input + mean_b
    return output

# endregion
