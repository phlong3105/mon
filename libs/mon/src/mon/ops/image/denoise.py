#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Denoising.

This module provides traditional denoising algorithms.
"""

from __future__ import annotations

__all__ = [
    "tv_denoise",
]

import numpy as np
import torch
from numpy import ndarray
from skimage.restoration import denoise_tv_chambolle
from torch import Tensor

from mon.core import TensorOrArray


# ==============================================================================
# region TV DENOISING
# ==============================================================================

def _tv_denoise_torch(
    image: Tensor,
    weight: float = 0.1,
    num_iter: int = 50,
) -> Tensor:
    """Highly optimized Total Variation Denoising for PyTorch.

    Args:
        image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.
        weight (float, optional): Weight of the denoised image. Defaults to 0.1.
        num_iter (int, optional): Number of iterations. Defaults to 50.

    Returns:
        Tensor: Denoising image.
    """
    # CRITICAL: Disable gradient tracking for pure image processing speed
    with torch.no_grad():
        b, c, h, w = image.shape

        # Dual variable p = (p_x, p_y)
        p = torch.zeros((b, c, 2, h, w), dtype=image.dtype, device=image.device)
        tau = 0.25  # Mathematical step size limit for 2D images

        # Pre-allocate tensors to avoid memory reallocation in the loop
        div_p = torch.zeros_like(image)
        grad_u = torch.zeros_like(p)

        for _ in range(num_iter):
            div_p.zero_()

            # 1. Compute divergence (vectorized slicing)
            div_p[:, :, 1:-1, :] += p[:, :, 0, 1:-1, :] - p[:, :, 0, :-2, :]
            div_p[:, :, 0, :]    += p[:, :, 0, 0, :]
            div_p[:, :, -1, :]   -= p[:, :, 0, -2, :]

            div_p[:, :, :, 1:-1] += p[:, :, 1, :, 1:-1] - p[:, :, 1, :, :-2]
            div_p[:, :, :, 0]    += p[:, :, 1, :, 0]
            div_p[:, :, :, -1]   -= p[:, :, 1, :, -2]

            # 2. Compute gradient of (div_p - x / weight)
            u = div_p - (image / weight)

            grad_u.zero_()
            grad_u[:, :, 0, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
            grad_u[:, :, 1, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]

            # 3. Update dual variable p
            mag = torch.sqrt(grad_u[:, :, 0]**2 + grad_u[:, :, 1]**2 + 1e-8)
            mag = mag.unsqueeze(2) # Shape: (B, C, 1, H, W)

            p = (p + tau * grad_u) / (1.0 + tau * mag)

        # Final Projection
        div_p.zero_()
        div_p[:, :, 1:-1, :] += p[:, :, 0, 1:-1, :] - p[:, :, 0, :-2, :]
        div_p[:, :, 0, :]    += p[:, :, 0, 0, :]
        div_p[:, :, -1, :]   -= p[:, :, 0, -2, :]
        div_p[:, :, :, 1:-1] += p[:, :, 1, :, 1:-1] - p[:, :, 1, :, :-2]
        div_p[:, :, :, 0]    += p[:, :, 1, :, 0]
        div_p[:, :, :, -1]   -= p[:, :, 1, :, -2]

        return image - weight * div_p


def _tv_denoise_numpy(
    image: ndarray,
    weight: float = 0.1,
    num_iter: int = 50,
) -> ndarray:
    """Fastest CPU implementation of Total Variation Denoising utilizing a
    Cython backend.

    Args:
        image (ndarray): Image array of shape (H, W, C) or (C, H, W) and values
            ranging from 0 to 255.
        weight (float, optional): Weight of the denoised image. Defaults to 0.1.
        num_iter (int, optional): Number of iterations. Defaults to 50.

    Returns:
        ndarray: Denoising image.
    """
    # Auto-detect channel-first (PyTorch standard) vs channel-last
    is_channel_first = image.ndim == 3 and image.shape[0] <= 4

    if is_channel_first:
        image = np.transpose(image, (1, 2, 0)) # Convert to (H, W, C) for skimage

    out = denoise_tv_chambolle(
        image,
        weight=weight,
        max_num_iter=num_iter,
        channel_axis=-1 if image.ndim == 3 else None
    )

    if is_channel_first:
        out = np.transpose(out, (2, 0, 1)) # Revert back to (C, H, W)

    # Ensure it returns the same float precision it received
    return out.astype(image.dtype)


def tv_denoise(
    image: TensorOrArray,
    weight: float = 0.1,
    num_iter: int = 50
) -> TensorOrArray:
    """Applies high-performance Total Variation Denoising.

    Automatically routes to the optimal GPU or CPU backend.

    Args:
        image (Tensor | ndarray): Image tensor of shape (B, C, H, W) and values
            ranging from 0.0 to 1.0; or an array of shape (H, W, C) or (C, H, W)
            and values ranging from 0 to 255.
        weight (float, optional): Weight of the denoised image. Defaults to 0.1.
        num_iter (int, optional): Number of iterations. Defaults to 50.

    Returns:
        Tensor | ndarray: Denoising image.
    """
    if isinstance(image, torch.Tensor):
        # Optional: Add batch dimension if missing (C, H, W) -> (1, C, H, W)
        has_batch = image.ndim == 4
        if not has_batch:
            image = image.unsqueeze(0)
        out = _tv_denoise_torch(image, weight, num_iter)
        return out.squeeze(0) if not has_batch else out
    elif isinstance(image, np.ndarray):
        return _tv_denoise_numpy(image, weight, num_iter)
    else:
        raise TypeError(
            f"Expected 'image' to be a tensor or an array, "
            f"but got {type(image).__name__}."
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
