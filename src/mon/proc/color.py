#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements basic color functions."""

from __future__ import annotations

__all__ = [
    "hsl2rgb_torch",
    "hsv2rgb_torch",
    "rgb2hsl_torch",
    "rgb2hsv_torch",
    # YCbCr
    "rgb_to_y",
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
]

import multipledispatch
import torch


# region HSV/HSL

def rgb2hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin  = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsl_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsl_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsl_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsl_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsl_h[cmax_idx == 3] = 0.0
    hsl_h /= 6.0
    
    hsl_l = (cmax + cmin) / 2.0
    hsl_s = torch.empty_like(hsl_h)
    hsl_s[hsl_l == 0] = 0
    hsl_s[hsl_l == 1] = 0
    hsl_l_ma          = torch.bitwise_and(hsl_l > 0, hsl_l < 1)
    hsl_l_s0_5        = torch.bitwise_and(hsl_l_ma, hsl_l <= 0.5)
    hsl_l_l0_5        = torch.bitwise_and(hsl_l_ma, hsl_l  > 0.5)
    hsl_s[hsl_l_s0_5] = ((cmax - cmin) / (hsl_l * 2.0))[hsl_l_s0_5]
    hsl_s[hsl_l_l0_5] = ((cmax - cmin) / (- hsl_l * 2.0 + 2.0))[hsl_l_l0_5]
    return torch.cat([hsl_h, hsl_s, hsl_l], dim=1)


def rgb2hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin  = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s  = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v  = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def hsv2rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]
    _c  = hsv_l * hsv_s
    _x  = _c * (- torch.abs(hsv_h * 6. % 2.0 - 1) + 1.0)
    _m  = hsv_l - _c
    _o  = torch.zeros_like(_c)
    idx = (hsv_h * 6.).type(torch.uint8)
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


def hsl2rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
    hsl_h, hsl_s, hsl_l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]
    _c  = (-torch.abs(hsl_l * 2.0 - 1.0) + 1) * hsl_s
    _x  = _c * (-torch.abs(hsl_h * 6.0 % 2.0 - 1) + 1.0)
    _m  = hsl_l - _c / 2.0
    idx = (hsl_h * 6.0).type(torch.uint8)
    idx = (idx % 6).expand(-1, 3, -1, -1)
    rgb = torch.empty_like(hsl)
    _o  = torch.zeros_like(_c)
    rgb[idx == 0] = torch.cat([_c, _x, _o], dim=1)[idx == 0]
    rgb[idx == 1] = torch.cat([_x, _c, _o], dim=1)[idx == 1]
    rgb[idx == 2] = torch.cat([_o, _c, _x], dim=1)[idx == 2]
    rgb[idx == 3] = torch.cat([_o, _x, _c], dim=1)[idx == 3]
    rgb[idx == 4] = torch.cat([_x, _o, _c], dim=1)[idx == 4]
    rgb[idx == 5] = torch.cat([_c, _o, _x], dim=1)[idx == 5]
    rgb += _m
    return rgb

# endregion


# region YCbCr

def _rgb_to_y(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


@multipledispatch.dispatch(torch.Tensor)
def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image to YCbCr.
    
    Args:
        image: RGB Image to be converted to YCbCr with shape :math:`[*, 3, H, W]`.

    Returns:
        YCbCr version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    delta = 0.5
    y     = _rgb_to_y(r, g, b)
    cb    = (b - y) * 0.564 + delta
    cr    = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], -3)


@multipledispatch.dispatch(torch.Tensor)
def rgb_to_y(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to Y.
    
    Args:
        image: RGB Image to be converted to Y with shape :math:`[*, 3, H, W]`.

    Returns:
        Y version of the image with shape :math:`[*, 1, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]
    y = _rgb_to_y(r, g, b)
    return y


@multipledispatch.dispatch(torch.Tensor)
def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YCbCr image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: YCbCr Image to be converted to RGB with shape :math:`[*, 3, H, W]`.

    Returns:
        RGB version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    y  = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta: float = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r = y + 1.403 * cr_shifted
    g = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b = y + 1.773 * cb_shifted
    return torch.stack([r, g, b], -3).clamp(0, 1)

# endregion
