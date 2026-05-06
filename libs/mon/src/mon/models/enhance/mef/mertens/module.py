#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the Mertens et al.
Exposure Fusion model.
"""

from __future__ import annotations

__all__ = [
    "mertens",
    "mertens_numpy",
]

from typing import Sequence

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import functional as F

from mon.ops import to_image_array


# ==============================================================================
# region MODULES
# ==============================================================================

# --- Pyramid ---

def _pad_stage_to_downsample(stage: Tensor) -> Tensor:
    _, _, h, w = stage.shape
    if h % 2 == 0:
        pad_vertical = (1, 0)
    else:
        pad_vertical = (1, 1)
    if w % 2 == 0:
        pad_horizontal = (1, 0)
    else:
        pad_horizontal = (1, 1)
    pad = (*pad_horizontal, *pad_vertical)
    padded = F.pad(stage, pad=pad, mode="replicate")
    return padded


def _expand_stage(x: Tensor, target_shape: tuple[int, int] | None = None):
    """Given a tensor, this function does a 2x upsampling by interpolating
    between samples. If ``target_shape`` is given, the function may pad a
    remaining row or column using ``padding=replicate``.

    Args:
        x (Tensor): The tensor to expand.
        target_shape (tuple[int, int], optional): The target shape. If None,
            meaning that the target shape is ``(2h-1. 2w-1)``. If the target is
            bigger, the output interpolation will be padded.
    """
    _, _, h, w = x.shape
    H = 2 * h - 1
    W = 2 * w - 1
    out = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
    if target_shape is not None:
        pad = (0, target_shape[1] - W, 0, target_shape[0] - H)
        out = F.pad(out, pad=pad, mode="replicate")
    return out


def _compute_gaussian_pyramid(x: Tensor, n_levels: int = 4) -> list[Tensor]:
    b, c, _, _ = x.shape
    if c not in [1, 3]:
        raise ValueError(f"expected 1 or 3 channels, got {c}.")

    downsample_kernel = torch.tensor(
        data=[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        device=x.device,
        dtype=x.dtype
    ).unsqueeze(0).unsqueeze(0) / 16

    # Treat color channels as batch dimension
    if c == 3:
        x = torch.cat([x[:, 0], x[:, 1], x[:, 2]], dim=0).unsqueeze(1)

    pyramid = [x]
    for lvl in range(1, n_levels):
        # Manually enforce padding = same on top and left sides.
        # Bottoms and right sides only when necessary.
        padded = _pad_stage_to_downsample(pyramid[-1])
        pyramid.append(F.conv2d(padded, downsample_kernel, padding="valid", stride=2))

    if c == 3:
    # Unpack color channels
        for lvl, stage in enumerate(pyramid):
            pyramid[lvl] = torch.cat([stage[:b], stage[b:2 * b], stage[2 * b:3 * b]], dim=1)

    return pyramid


def _compute_laplacian_pyramid(gaussian_pyramid: list[Tensor]) -> list[Tensor]:
    n_levels = len(gaussian_pyramid)
    laplacian_pyramid = []
    for lvl in range(n_levels - 2, -1, -1):
        _, _, *target_shape = gaussian_pyramid[lvl].shape
        expanded_g = _expand_stage(gaussian_pyramid[lvl + 1], target_shape=target_shape)
        laplacian_pyramid.append(gaussian_pyramid[lvl] - expanded_g)
    laplacian_pyramid.append(gaussian_pyramid[-1])   # Coarsest level is just gaussian coarse.
    return laplacian_pyramid


def _merge_laplacian_pyramid(
    image_pyramid: list[Tensor],
    weight_pyramid: list[Tensor]
) -> list[Tensor]:
    n_stages = len(image_pyramid)
    if n_stages != len(weight_pyramid):
        raise ValueError(f"expected {n_stages} stages, got {len(weight_pyramid)}.")

    merged_pyramid = []
    for lvl, (weight, img) in enumerate(zip(weight_pyramid, image_pyramid)):
        merged_pyramid.append(torch.sum(weight * img, dim=0, keepdim=True))
    return merged_pyramid


def _collapse_pyramid(laplacian_pyramid: list[Tensor]) -> Tensor:
    curr = laplacian_pyramid[-1]
    for stage in laplacian_pyramid[-2::-1]:  # Reverse order, starting from the penultimate
        _, _, *target_shape = stage.size()
        curr = _expand_stage(curr, target_shape=target_shape)
        curr = curr + stage
    return curr


# --- Functional ---

def _compute_contrast(gray_images: Tensor) -> Tensor:
    k_laplacian = torch.tensor(
        data=[[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        device=gray_images.device,
        dtype=gray_images.dtype
    ).unsqueeze(0).unsqueeze(0)
    contrast = torch.abs(F.conv2d(gray_images, k_laplacian, padding="same"))
    return contrast


def _compute_saturation(images: Tensor, gray_images: Tensor) -> Tensor:
    sat = torch.sqrt(torch.mean((images - gray_images) ** 2, dim=1, keepdim=True))
    return sat


def _compute_well_exposedness(images: Tensor) -> Tensor:
    sigma = 0.2
    well_exposedness = torch.exp(-torch.sum((images - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma))
    return well_exposedness


def _mertens_torch(
    images: Tensor | list[Tensor],
    w_sat: float = 1.0,
    w_cont: float = 1.0,
    w_exp: float = 1.0,
    n_levels: int = 4
) -> Tensor:
    if isinstance(images, Sequence):
        images = torch.stack(images, dim=0)

    gray = torch.mean(images, dim=1, keepdim=True)
    cont = _compute_contrast(gray_images=gray)
    sat = _compute_saturation(images=images, gray_images=gray)
    exp = _compute_well_exposedness(images=images)

    weights = (cont ** w_cont) * (sat ** w_sat) * (exp ** w_exp)
    # Normalize weights
    weights = weights / weights.sum(dim=0, keepdim=True)
    # Normalization will give Nan if all frames have 0 weight at 1 pixel.
    # In this case, all of them get the same weight
    weights = weights.nan_to_num(nan =1 / images.size(0))

    # Get gaussian pyramid for weights and images
    gaussian_pyramid = _compute_gaussian_pyramid(x=images, n_levels=n_levels)
    laplacian_pyramid = _compute_laplacian_pyramid(gaussian_pyramid=gaussian_pyramid)
    weight_gaussian_pyramid = _compute_gaussian_pyramid(x=weights, n_levels=n_levels)
    fused_laplacian_pyramid = _merge_laplacian_pyramid(laplacian_pyramid, weight_gaussian_pyramid)
    fusion = _collapse_pyramid(fused_laplacian_pyramid)

    return fusion


def mertens_numpy(
    images: list[ndarray] | list[Tensor] | Tensor,
    w_sat: float = 1.0,
    w_cont: float = 1.0,
    w_exp: float = 1.0,
    n_levels: int = 4
) -> ndarray:
    if isinstance(images, Tensor):
        images = torch.split(images, 1, dim=0)
    if isinstance(images, Sequence) and all(isinstance(i, Tensor) for i in images):
        images = [to_image_array(i) for i in images]

    align_mtb = cv2.createAlignMTB()
    align_mtb.process(images, images)
    mertens = cv2.createMergeMertens()
    mertens.setContrastWeight(w_cont)
    mertens.setSaturationWeight(w_sat)
    mertens.setExposureWeight(w_exp)
    fusion = mertens.process(images)
    fusion = np.clip(fusion * 255, 0, 255).astype(np.uint8)
    return fusion


def mertens(
    images: Tensor | list[Tensor] | list[ndarray],
    w_sat: float = 1.0,
    w_cont: float = 1.0,
    w_exp: float = 1.0,
    n_levels: int = 4
):
    """Apply Mertens exposure fusion algorithm to a sequence of images.

    Combine a burst of images with different exposures into a single image with
    compressed dynamic range.

    Args:
        images (Tensor | list[Tensor] | list[ndarray]): Input tensor of shape
            (B, C, H, W); or a list of images of shape (C, H, W).
        w_sat (float, optional): The saturation importance weight. Defaults to 1.0.
        w_cont (float, optional): The contrast importance weight. Defaults to 1.0.
        w_exp (float, optional): The well-exposed importance weight. Defaults to 1.0.
        n_levels (int, optional): The number of levels in the pyramids. Defaults to 4.

    Returns:
        The fused image of shape (1, C, H, W) or (H, W, C).
    """
    if isinstance(images, Tensor) or all(isinstance(i, Tensor) for i in images):
        return _mertens_torch(
            images=images,
            w_sat=w_sat,
            w_cont=w_cont,
            w_exp=w_exp,
            n_levels=n_levels
        )
    elif all(isinstance(i, ndarray) for i in images):
        return mertens_numpy(
            images=images,
            w_sat=w_sat,
            w_cont=w_cont,
            w_exp=w_exp,
            n_levels=n_levels
        )
    else:
        raise TypeError(f"expected all images to be either tensor or array, "
                        f"got {type(images).__name__}.")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
