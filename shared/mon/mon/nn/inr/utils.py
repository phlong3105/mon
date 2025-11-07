#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements utility functions for Implicit Neural Representations (INR).
"""

__all__ = [
    "create_coords",
    "create_depth_aware_patches",
    "create_noisy_coords",
    "create_patches",
    "ff_embedding",
    "filter_up",
    "interpolate_image",
    "pair_downsampler",
]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mon.core.dtypes import image as I


# ----- Coordinate -----
def create_coords(size: int) -> torch.Tensor:
    """Creates a coordinate grid for INF.

    Args:
        size: Size of the square coordinates grid.

    Returns:
        A ``torch.Tensor`` of shape :math:`(size, size, 2)` with normalized
        coords.
    """
    h, w   = size, size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    return torch.from_numpy(coords).float()


def create_noisy_coords(size: int, sigma: float = 0.5, lamda: float = 1.0) -> torch.Tensor:
    """Creates a coordinates grid with Gaussian noise added.

    Args:
        size: The size of the coordinates grid.
        sigma: Standard deviation of the Gaussian noise. Default: ``0.5``.
        lamda: Lambda parameter for Poisson noise. Default: ``1.0``.
    """
    h, w   = size, size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    coords = torch.from_numpy(coords).float()

    # Add Gaussian noise
    gaussian_noise = torch.normal(mean=0.0, std=sigma, size=coords.shape).to(coords.device)
    noisy_coords   = coords + gaussian_noise
    
    # Add Poisson noise
    poisson_noise  = torch.poisson(torch.full(coords.shape, lamda)).to(coords.device) - lamda
    noisy_coords   = noisy_coords + poisson_noise * 0.1  # Scale noise
    
    # Clip to ensure coordinates stay within [0, 1]
    noisy_coords = torch.clamp(noisy_coords, 0.0, 1.0)

    return noisy_coords


def ff_embedding(p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Applies Fourier feature embedding to input tensor.

    Args:
        p: Input tensor to embed.
        B: Projection matrix as a ``torch.Tensor``. Default: ``None``.

    Returns:
        An embedded ``torch.Tensor`` with sine and cosine features.
    """
    if B is None:
        return p
    else:
        x_proj    = (2 * np.pi * p) @ B.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embedding


# ----- Context/Patch -----
def create_patches(image: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Creates a tensor where the channel contains patch information.

    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(1, C, H, W)` in
            range :math:`[0, 1]`.
        kernel_size: Size of square patches. Default: ``7``.

    Returns:
        A ``torch.Tensor`` of patches in channels of shape :math:`(H, W, kernel_size^2)`.
    """
    if image.ndim != 4:
        raise ValueError(f"``image`` must be a torch.Tensor of shape (1, C, H, W), got {image.shape}.")
    
    b, c, h, w = image.shape
    kernel     = torch.zeros((kernel_size ** 2, c, kernel_size, kernel_size)).to(image.device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i + j * kernel_size, :, i, j] = 1

    pad          = nn.ReflectionPad2d(kernel_size // 2)
    image_padded = pad(image)
    patches      = F.conv2d(image_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(patches, 0, -1)


def create_depth_aware_patches(
    image      : torch.Tensor,
    depth      : torch.Tensor,
    kernel_size: int   = 7,
    alpha      : float = 8.3
) -> torch.Tensor:
    """Creates a tensor where the channel contains weighted patch information
    based on depth.
    
    Args:
        image: Image as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in
            range :math:`[0, 1]`.
        depth: Depth map as a ``torch.Tensor`` of shape :math:`(B, 1, H, W)` in
            range :math:`[0, 1]`.
        kernel_size: Size of square patches. Default: ``1``.
        alpha: Controls depth similarity sensitivity (larger = stricter decay).
            Default: ``8.3``.
        
    Returns:
        A ``torch.Tensor`` with patches in channels of shape :math:`(B, H', W', K^2)`.
    """
    b, c, h, w = image.shape
    kernel = torch.zeros((kernel_size ** 2, c, kernel_size, kernel_size)).to(image.device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i + j * kernel_size, 0, i, j] = 1

    pad           = nn.ReflectionPad2d(kernel_size // 2)
    image_padded  = pad(image)
    image_patches = F.conv2d(image_padded, kernel, padding=0).squeeze(0)
    depth_padded  = pad(depth)
    depth_patches = F.conv2d(depth_padded, kernel, padding=0).squeeze(0)

    # Compute center index in patch
    center_idx   = (kernel_size ** 2) // 2
    depth_center = depth_patches[center_idx, :, :].unsqueeze(0).repeat(kernel_size ** 2, 1, 1)
    
    # FD = exp(-alpha * |depth_center - depth_neighbor|)
    depth_diff = torch.abs(depth_center - depth_patches)
    fd         = torch.exp(-alpha * depth_diff)  # Shape for multiplication
    
    # Weight the image patches and normalize
    patches     = image_patches * fd
    weights_sum = fd.sum(dim=0, keepdim=True) + 1e-6  # Avoid division by zero
    patches     = patches / weights_sum

    return torch.movedim(patches, 0, -1)


# ----- Scale -----
def pair_downsampler(image: torch.Tensor) -> torch.Tensor:
    c       = image.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5],[0.5, 0]]]]).to(image.device)
    filter1 = filter1.repeat(c,1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0],[0, 0.5]]]]).to(image.device)
    filter2 = filter2.repeat(c,1, 1, 1)
    output1 = F.conv2d(image, filter1, stride=2, groups=c)
    output2 = F.conv2d(image, filter2, stride=2, groups=c)
    return output1, output2


def interpolate_image(image: torch.Tensor, size: int) -> torch.Tensor:
    """Reshapes the image based on new resolution."""
    # return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
    return F.interpolate(image, size=(size, size), mode="area")


def filter_up(
    x_lr       : torch.Tensor,
    y_lr       : torch.Tensor,
    x_hr       : torch.Tensor,
    kernel_size: int = 7
) -> torch.Tensor:
    """Applies the guided filter to upscale the predicted image. """
    gf   = I.FastGuidedFilter(kernel_size)
    y_hr = gf(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr
