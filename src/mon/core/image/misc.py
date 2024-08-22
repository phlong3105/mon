#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Miscellaneous Image Processing Ops.

This module implements miscellaneous image processing operations that are not
specific to any particular category.
"""

from __future__ import annotations

__all__ = [
    "add_weighted",
    "blend",
]

import numpy as np
import torch


def add_weighted(
    image1: torch.Tensor | np.ndarray,
    alpha : float,
    image2: torch.Tensor | np.ndarray,
    beta  : float,
    gamma : float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Calculate the weighted sum of two image tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1: The first image.
        alpha: The weight of the :obj:`image1` elements.
        image2: The second image.
        beta: The weight of the :obj:`image2` elements.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A weighted image.
    """
    if image1.shape != image2.shape:
        raise ValueError(f"`image1` and `image2` must have the same shape, "
                         f"but got {image1.shape} and {image2.shape}.")
    bound  = 1.0 if image1.is_floating_point() else 255.0
    output = image1 * alpha + image2 * beta + gamma
    if isinstance(output, torch.Tensor):
        output = output.clamp(0, bound).to(image1.dtype)
    elif isinstance(output, np.ndarray):
        output = np.clip(output, 0, bound)
    else:
        raise TypeError(f"`image` must be a `numpy.ndarray` or `torch.Tensor`, "
                        f"but got {type(input)}.")
    return output


def blend(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    alpha : float,
    gamma : float = 0.0
) -> torch.Tensor | np.ndarray:
    """Blend 2 images together using the formula:
        output = :obj:`image1` * alpha + :obj:`image2` * beta + gamma

    Args:
        image1: A source image.
        image2: A n overlay image that we want to blend on top of :obj:`image1`.
        alpha: An alpha transparency of the overlay.
        gamma: A scalar added to each sum. Default: ``0.0``.

    Returns:
        A blended image.
    """
    return add_weighted(
        image1 = image2,
        alpha  = alpha,
        image2 = image1,
        beta   = 1.0 - alpha,
        gamma  = gamma,
    )
