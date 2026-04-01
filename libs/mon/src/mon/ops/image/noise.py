#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Noise.

This module provides traditional algorithms to handle noisy images.
"""

from __future__ import annotations

__all__ = [
    "anscombe",
    "inverse_anscombe",
]

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from mon.core import TensorOrArray


# ==============================================================================
# region ANSCOMBE TRANSFORM
# ==============================================================================

# --- Anscombe ---

def _anscombe_numpy(x: ndarray, eps: float = 1e-6) -> ndarray:
    """Compute the Anscombe variance stabilizing transform.

    Args:
        x (ndarray): A noisy Poisson-distributed image array of shape
            (H, W, C) and values ranging from 0 to 255.
        eps (float, optional): A small constant added to the input to prevent
            numerical instability when taking the square root. Defaults to 1e-6.

    Returns:
        ndarray: The Anscombe transformed image array of shape (H, W, C) and
            values ranging from 0 to 16 (variance approximately equal to 1).
    """
    return 2.0 * np.sqrt(x + (3.0 / 8.0) + eps)


def _anscombe_torch(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute the Anscombe variance stabilizing transform.

    Args:
        x (Tensor): A noisy Poisson-distributed image tensor of shape
            (B, C, H, W) and values ranging from 0.0 to 1.0.
        eps (float, optional): A small constant added to the input to prevent
            numerical instability when taking the square root. Defaults to 1e-6.

    Returns:
        Tensor: The Anscombe transformed image tensor of shape (B, C, H, W) and
            values ranging from 0.0 to 2.0 (variance approximately equal to 1).
    """
    return 2.0 * torch.sqrt(x + (3.0 / 8.0) + eps)


def anscombe(x: TensorOrArray, eps: float = 1e-6) -> TensorOrArray:
    """Compute the Anscombe variance stabilizing transform.

    References:
        - Paper: Anscombe, F. J. (1948), "The transformation of Poisson,
          binomial and negative-binomial data", Biometrika 35 (3-4): 246-254

    Args:
        x (TensorOrArray): A noisy Poisson-distributed image tensor of shape
            (B, C, H, W) and values ranging from 0.0 to 1.0; or an array of shape
            (H, W, C) and values ranging from 0 to 255.
        eps (float, optional): A small constant added to the input to prevent
            numerical instability when taking the square root. Defaults to 1e-6.

    Returns:
        TensorOrArray: The Anscombe transformed image tensor of shape
            (B, C, H, W) and values ranging from 0.0 to 2.0; or an array of
            shape (H, W, C) and values ranging from 0 to 16 (variance
            approximately equal to 1).
    """
    if isinstance(x, Tensor):
        return _anscombe_torch(x, eps=eps)
    else:
        return _anscombe_numpy(x, eps=eps)


# --- Inverse Anscombe ---

def _inverse_anscombe_numpy(z: ndarray, unbiased: bool = False) -> ndarray:
    """Compute the inverse Anscombe variance stabilizing transform.

    Args:
        z (ndarray): An Anscombe-transformed image array of shape (H, W, C)
            and values ranging from 0 to 16.
        unbiased (bool, optional): If True, compute the inverse transform using
            an approximation of the exact unbiased inverse. Reference:
            "Makitalo, M., & Foi, A. (2011). A closed-form approximation of the
            exact unbiased inverse of the Anscombe variance-stabilizing
            transformation. Image Processing." Defaults to False.

    Returns:
        ndarray: The inverse Anscombe transformed image array of shape (H, W, C)
            and values ranging from 0 to 255.
    """
    if unbiased:
        return (
              1.0 / 4.0 * np.power(z, 2)
            + 1.0 / 4.0 * np.sqrt(3.0 / 2.0) * np.power(z, -1.0)
            - 11.0 / 8.0 * np.power(z, -2.0)
            + 5.0 / 8.0 * np.sqrt(3.0 / 2.0) * np.power(z, -3.0) - 1.0 / 8.0
        )
    else:
        return (z - 3.0 / 8.0) / 2.0 ** 0.5


def _inverse_anscombe_torch(z: Tensor, unbiased: bool = False) -> Tensor:
    """Compute the inverse Anscombe variance stabilizing transform.

    Args:
        z (Tensor): An Anscombe-transformed image tensor of shape (B, C, H, W)
            and values ranging from 0.0 to 2.0.
        unbiased (bool, optional): If True, compute the inverse transform using
            an approximation of the exact unbiased inverse. Reference:
            "Makitalo, M., & Foi, A. (2011). A closed-form approximation of the
            exact unbiased inverse of the Anscombe variance-stabilizing
            transformation. Image Processing." Defaults to False.

    Returns:
        Tensor: The inverse Anscombe transformed image tensor of shape (B, C, H, W)
            and values ranging from 0.0 to 1.0.
    """
    if unbiased:
        return (
              1.0 / 4.0 * torch.pow(z, 2)
            + 1.0 / 4.0 * torch.sqrt(torch.tensor(3.0 / 2.0)) * torch.pow(z, -1.0)
            - 11.0 / 8.0 * torch.pow(z, -2.0)
            + 5.0 / 8.0 * torch.sqrt(torch.tensor(3.0 / 2.0)) * torch.pow(z, -3.0) - 1.0 / 8.0
        )
    else:
        return (z - 3.0 / 8.0) / 2.0 ** 0.5


def inverse_anscombe(z: TensorOrArray, unbiased: bool = False) -> TensorOrArray:
    """Compute the inverse Anscombe variance stabilizing transform.

    References:
        - Paper: Anscombe, F. J. (1948), "The transformation of Poisson,
          binomial and negative-binomial data", Biometrika 35 (3-4): 246-254

    Args:
        z (TensorOrArray): An Anscombe-transformed image tensor of shape
            (B, C, H, W) and values ranging from 0.0 to 2.0; or an array of
            shape (H, W, C) and values ranging from 0 to 16.
        unbiased (bool, optional): If True, compute the inverse transform using
            an approximation of the exact unbiased inverse. Reference:
            "Makitalo, M., & Foi, A. (2011). A closed-form approximation of the
            exact unbiased inverse of the Anscombe variance-stabilizing
            transformation. Image Processing." Defaults to False.

    Returns:
        TensorOrArray: The inverse Anscombe transformed image tensor of shape
            (B, C, H, W) and values ranging from 0.0 to 1.0; or an array of
            shape (H, W, C) and values ranging from 0 to 255.
    """
    if isinstance(z, Tensor):
        return _inverse_anscombe_torch(z, unbiased=unbiased)
    else:
        return _inverse_anscombe_numpy(z, unbiased=unbiased)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
