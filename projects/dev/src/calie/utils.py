#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SALEO utilities.

This module provides various utilities for SALEO.
"""

from __future__ import annotations

__all__ = [
    "RgbToHsv",
    "RgbToHvi",
    "filter_up",
    "get_coords",
    "get_patches",
    "get_v_component",
    "interpolate_image",
    "replace_v_component",
]

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mon.core import Size, SizeLike
from mon.cv.ops import FastGuidedFilter


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Color ---

class RgbToHsv(nn.Module):
    """A convenience class to convert RGB images to HSV color space and back."""

    # --- Callable & Context Manager ---
    def from_rgb(self, rgb: Tensor) -> Tensor:
        """Convert an RGB image to HSV color space.

        Args:
            rgb (Tensor): An RGB image tensor of shape (B, 3, H, W) and pixel
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: An HSV image tensor of shape (B, 3, H, W) and pixel values
                ranging from 0.0 to 1.0.
        """
        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin = torch.min(rgb, dim=1, keepdim=True)[0]
        delta = cmax - cmin
        hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
        cmax_idx[delta == 0] = 3
        hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
        hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
        hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
        hsv_h[cmax_idx == 3] = 0.0
        hsv_h /= 6.0
        hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
        hsv_v = cmax
        return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    def to_rgb(self, hsv: Tensor) -> Tensor:
        """Convert an HSV image to RGB color space.

        Args:
            hsv (Tensor): An HSV image tensor of shape (B, 3, H, W) and pixel
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: An RGB image tensor of shape (B, 3, H, W) and pixel values
                ranging from 0.0 to 1.0.
        """
        hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
        _c = hsv_l * hsv_s
        _x = _c * (- torch.abs(hsv_h * 6. % 2.0 - 1) + 1.)
        _m = hsv_l - _c
        _o = torch.zeros_like(_c)
        idx = (hsv_h * 6.0).type(torch.uint8)
        idx = (idx % 6).expand(-1, 3, -1, -1)
        rgb = torch.empty_like(hsv)
        rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
        rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
        rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
        rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
        rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
        rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
        rgb += _m
        return rgb


class RgbToHvi(nn.Module):
    """A module for converting RGB images to HVI color space and back.

    References:
        - Code: https://github.com/Fediory/HVI-CIDNet/blob/master/net/HVI_transform.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-8, requires_grad: bool = False):
        """Initialize a new instance.

        Args:
            eps (float): Epsilon value to avoid division by zero. Defaults to 1e-8.
            requires_grad (bool): If True, allows gradient computation for
                ``density_k``. Defaults to False.
        """
        super().__init__()
        # Assign attributes
        self.pi = 3.141592653589793
        self.eps = eps
        # Learnable 'k' controls the color gamut sensitivity relative to intensity
        self.density_k = nn.Parameter(
            torch.full([1], 0.1),
            requires_grad=requires_grad
        )

    # --- Callable & Context Manager ---
    def from_rgb(self, rgb: Tensor) -> Tensor:
        """Convert an RGB image to HVI color space.

        Args:
            rgb (Tensor): An RGB image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: The corresponding HVI image tensor of shape (B, 3, H, W)
                with H and V values ranging from -1.0 to 1.0, and I values
                ranging from 0.0 to 1.0.
        """
        r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

        max_val, _ = rgb.max(1)
        min_val, _ = rgb.min(1)
        diff = max_val - min_val + self.eps

        # Standard Hue calculation (0-6 range)
        hue = torch.zeros_like(max_val)
        hue = torch.where(max_val == r, (g - b) / diff % 6, hue)
        hue = torch.where(max_val == g, (b - r) / diff + 2, hue)
        hue = torch.where(max_val == b, (r - g) / diff + 4, hue)
        hue = torch.where(diff < self.eps, torch.zeros_like(hue), hue)
        hue = hue / 6.0  # Normalized to [0, 1]

        # Saturation and Intensity
        # S = (max - min) / max
        s = diff / (max_val + self.eps)
        i = max_val # Intensity (Value)

        # Cartesian Mapping (The HVI specific logic)
        # Sensitivity curves the color response based on Intensity
        sensitivity = ((i * 0.5 * self.pi).sin() + self.eps).pow(self.density_k)

        # Mapping Hue/Saturation (Polar) -> H/V (Cartesian)
        h_coord = sensitivity * s * torch.cos(2.0 * self.pi * hue)
        v_coord = sensitivity * s * torch.sin(2.0 * self.pi * hue)

        return torch.stack([h_coord, v_coord, i], dim=1)

    def to_rgb(self, hvi: Tensor) -> Tensor:
        """Convert an HVI image to RGB color space.

        Args:
            hvi (Tensor): A HVI image tensor of shape (B, 3, H, W) with H and V
                values ranging from -1.0 to 1.0, and I values ranging from
                0.0 to 1.0.

        Returns:
            Tensor: The corresponding RGB image tensor of shape (B, 3, H, W)
                and values ranging from 0.0 to 1.0.
        """
        h_coord, v_coord, i = hvi[:, 0, :, :], hvi[:, 1, :, :], hvi[:, 2, :, :]

        sensitivity = ((i * 0.5 * self.pi).sin() + self.eps).pow(self.density_k)

        # Reconstruct Hue (angle) and Saturation (magnitude)
        h = torch.atan2(v_coord, h_coord) / (2.0 * self.pi) % 1.0
        s = torch.sqrt(h_coord ** 2 + v_coord ** 2 + self.eps) / (sensitivity + self.eps)
        s = torch.clamp(s, 0, 1)

        # HSV to RGB Conversion (Simplified Vectorized)
        hi = (h * 6.0).floor()
        f = h * 6.0 - hi
        p = i * (1.0 - s)
        q = i * (1.0 - (f * s))
        t = i * (1.0 - ((1.0 - f) * s))

        # Reconstruct RGB channels based on Hue sector
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        # Hi indices: 0: R,Gt,Bp | 1: Rq,G,Bp | 2: Rp,G,Bt | 3: Rp,Gq,B | 4: Rt,Gp,B | 5: R,Gp,Bq
        mask = (hi == 0); r[mask], g[mask], b[mask] = i[mask], t[mask], p[mask]
        mask = (hi == 1); r[mask], g[mask], b[mask] = q[mask], i[mask], p[mask]
        mask = (hi == 2); r[mask], g[mask], b[mask] = p[mask], i[mask], t[mask]
        mask = (hi == 3); r[mask], g[mask], b[mask] = p[mask], q[mask], i[mask]
        mask = (hi == 4); r[mask], g[mask], b[mask] = t[mask], p[mask], i[mask]
        mask = (hi == 5); r[mask], g[mask], b[mask] = i[mask], p[mask], q[mask]

        return torch.stack([r, g, b], dim=1)


def get_v_component(image_hsv: Tensor) -> Tensor:
    """Assumes (1, 3, H, W) HSV image."""
    return image_hsv[:, -1].unsqueeze(0)


def replace_v_component(image_hsv: Tensor, v_new: Tensor) -> Tensor:
    """Replaces the V component of an HSV image (1, 3, H, W)."""
    image_hsv[:, -1] = v_new
    return image_hsv


# --- Features ---

def get_coords(size: SizeLike) -> Tensor:
    """Create a normalized square coordinates grid.

    Args:
        size (SizeLike): Size of the grid.

    Returns:
        Tensor: Coordinates tensor of shape (1, H, W, 2) and values ranging from
            -1.0 to 1.0.
    """
    size = Size.from_value(size)
    h, w = size.hw
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    coords = torch.from_numpy(coords).float()
    return coords


def get_patches(image: Tensor, kernel_size: int = 7) -> Tensor:
    """Create a tensor where the channel contains patch information.

    Args:
        image (Tensor): Image, formatted as a tensor of shape (1, C, H, W) and
            pixel values ranging from 0.0 to 1.0.
        kernel_size (int, optional): Size of square patches. Defaults to 7.

    Returns:
        Tensor: A tensor of shape (1, H', W', K^2) where H' and W' are the
            height and width after patch extraction, and K is the ``kernel_size``.

    Raises:
        ValueError: If the input ``image`` does not have 4 dimensions.
    """
    kernel = torch.zeros((kernel_size ** 2, 1, kernel_size, kernel_size)).to(image.device)

    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1

    pad = nn.ReflectionPad2d(kernel_size // 2)
    im_padded = pad(image)
    extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(extracted, 0, -1)


# --- Resize ---

def interpolate_image(image: Tensor, size: SizeLike) -> Tensor:
    """Reshapes the image based on new resolution."""
    size = Size.from_value(size)
    return F.interpolate(image, size=size.hw)


def filter_up(x_lr: Tensor, y_lr: Tensor, x_hr: Tensor, r: int = 1):
    """Applies the guided filter to upscale the predicted image."""
    guided_filter = FastGuidedFilter(r=r)
    y_hr = guided_filter(x_lr, y_lr, x_hr)
    y_hr = torch.clip(y_hr, 0, 1)
    return y_hr

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
