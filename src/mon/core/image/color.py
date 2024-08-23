#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Colorspace Conversion.

This module implements basic color space conversion functions. We use :obj:`cv2`
to handle :obj:`numpy.ndarray` and :obj:`kornia` to handle :obj:`torch.Tensor`.
"""

from __future__ import annotations

__all__ = [
    "RGBToHVI",
    "bgr_to_grayscale",
    "bgr_to_rgb",
    "bgr_to_rgba",
    "bgr_to_xyz",
    "bgr_to_y",
    "bgr_to_ycbcr",
    "bgr_to_yuv",
    "grayscale_to_rgb",
    "hls_to_rgb",
    "hsv_to_rgb",
    "lab_to_rgb",
    "luv_to_rgb",
    "rgb_to_bgr",
    "rgb_to_grayscale",
    "rgb_to_hls",
    "rgb_to_hsv",
    "rgb_to_lab",
    "rgb_to_linear_rgb",
    "rgb_to_luv",
    "rgb_to_rgba",
    "rgb_to_sepia",
    "rgb_to_xyz",
    "rgb_to_y",
    "rgb_to_ycbcr",
    "rgb_to_yuv",
    "rgba_to_bgr",
    "rgba_to_rgb",
    "xyz_to_rgb",
    "ycbcr_to_rgb",
    "yuv_to_bgr",
    "yuv_to_rgb",
]

import cv2
import kornia
import numpy as np
import torch
from torch import nn


# region Grayscale

def grayscale_to_rgb(
    image: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """Convert a grayscale image to RGB.
    
    Args:
        image: A grayscale image of type:
            - :obj:`torch.Tensor` in ``[*, 1, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.grayscale_to_rgb(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    

def rgb_to_grayscale(
    image      : torch.Tensor | np.ndarray,
    rgb_weights: torch.Tensor = None
) -> torch.Tensor:
    """Convert an RGB image to grayscale.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        rgb_weights: Weights that will be applied on each channel (RGB). The
            sum of the weights should add up to one. Defaults: ``None``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_grayscale(image, rgb_weights)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def bgr_to_grayscale(
    image: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """Convert a BGR image to grayscale. First flips to RGB, then converts.
    
    Args:
        image: A BGR image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_grayscale(rgb)
    
# endregion


# region HSL

def rgb_to_hls(
    image: torch.Tensor | np.ndarray,
    eps  : float = 1e-8
) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to HLS.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-8``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_hls(image, eps)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    

def hls_to_rgb(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a HLS image to RGB.

    Args:
        image: An HLS image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.hls_to_rgb(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")

# endregion


# region HSV

def rgb_to_hsv(
    image: torch.Tensor | np.ndarray,
    eps  : float = 1e-8
) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to HSV.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        eps: Scalar to enforce numerical stability. Defaults: ``1e-8``.
    
    Returns:
        The `H` channel values are in the range ``[0, 2pi]``.
        The `S` and `V` channels are in the range ``[0.0, 1.0]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_hsv(image, eps)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def hsv_to_rgb(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an HSV image to RGB.
    
    Args:
        image: An HSV image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format. The `H` channel
                values are in the range ``[0, 2pi]``. The `S` and `V` channels
                are in the range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.hsv_to_rgb(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    
# endregion


# region HVI

class RGBToHVI(nn.Module):
    """Convert an RGB image to HVI.
    
    Args:
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-8``.
    
    References:
        https://github.com/Fediory/HVI-CIDNet/blob/master/net/HVI_transform.py
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps       = eps
        self.density_k = nn.Parameter(torch.full([1], 0.2), requires_grad=True)  # k is reciprocal to the paper mentioned
        self.gated     = False
        self.gated2    = False
        self.alpha     = 1.0
        self.this_k    = 0
    
    def rgb_to_hvi(self, image: torch.Tensor) -> torch.Tensor:
        pi      = 3.141592653589793
        device  = image.device
        dtypes  = image.dtype
        hue     = torch.Tensor(image.shape[0], image.shape[2], image.shape[3]).to(device).to(dtypes)
        value   = image.max(1)[0].to(dtypes)
        img_min = image.min(1)[0].to(dtypes)
        hue[image[:, 2] == value] =  4.0 + ((image[:, 0] - image[:, 1]) / (value - img_min + self.eps))[image[:, 2] == value]
        hue[image[:, 1] == value] =  2.0 + ((image[:, 2] - image[:, 0]) / (value - img_min + self.eps))[image[:, 1] == value]
        hue[image[:, 0] == value] = (0.0 + ((image[:, 1] - image[:, 2]) / (value - img_min + self.eps))[image[:, 0] == value]) % 6
        
        hue[image.min(1)[0] == value] = 0.0
        hue = hue / 6.0
        
        saturation = (value - img_min) / (value + self.eps)
        saturation[value == 0] = 0
        
        hue         = hue.unsqueeze(1)
        saturation  = saturation.unsqueeze(1)
        value       = value.unsqueeze(1)
        
        self.this_k     = self.density_k.item()
        color_sensitive = ((value * 0.5 * pi).sin() + self.eps).pow(self.density_k)
        cx   = (2.0 * pi * hue).cos()
        cy   = (2.0 * pi * hue).sin()
        X   = color_sensitive * saturation * cx
        Y   = color_sensitive * saturation * cy
        Z   = value
        xyz = torch.cat([X, Y, Z], dim=1)
        return xyz
    
    def hvi_to_rgb(self, image: torch.Tensor) -> torch.Tensor:
        pi      = 3.141592653589793
        H, V, I = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        
        # clip
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        I = torch.clamp(I, 0, 1)
        
        v = I
        color_sensitive = ((v * 0.5 * pi).sin() + self.eps).pow(self.this_k)
        H = H / (color_sensitive + self.eps)
        V = V / (color_sensitive + self.eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V, H) / (2 * pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2)
        
        if self.gated:
            s = s * 1.3
        
        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)
        
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))
        
        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5
        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]
        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]
        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]
        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]
        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]
        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

# endregion


# region Lab

def rgb_to_lab(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to Lab. Lab color is computed using the D65
    illuminant and Observer 2.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    
    Returns:
        The `L` channel values are in the range ``[0, 100]``.
        The `a` and `b` channels are in the range ``[-128, 127]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_lab(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def lab_to_rgb(
    image: torch.Tensor | np.ndarray,
    clip : bool = True
) -> torch.Tensor | np.ndarray:
    """Convert a Lab image to RGB.
    
    Args:
        image: A Lab image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``. The `L` channel values are in the range
                ``[0, 100]``. The `a` and `b` channels are in the range
                ``[-128, 127]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        clip: Whether to apply clipping to insure output RGB values in range
            ``[0.0, 1.0]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.lab_to_rgb(image, clip)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")

# endregion


# region Luv

def rgb_to_luv(
    image: torch.Tensor | np.ndarray,
    eps  : float = 1e-12
) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to Luv. Luv color is computed using the D65
    illuminant and Observer 2.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-12``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_luv(image, eps)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def luv_to_rgb(
    image: torch.Tensor | np.ndarray,
    eps  : float = 1e-12
) -> torch.Tensor | np.ndarray:
    """Convert a Luv image to RGB.
    
    Args:
        image: A Luv image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-12``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.luv_to_rgb(image, eps)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_LUV2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")

# endregion


# region RGB

def rgb_to_bgr(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to BGR.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    return bgr_to_rgb(image)


def bgr_to_rgb(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a BGR image to RGB.

    Args:
        image: A BGR image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.bgr_to_rgb(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def rgb_to_rgba(
    image    : torch.Tensor | np.ndarray,
    alpha_val: float | torch.Tensor
) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to RGBA.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        alpha_val: A :obj:`float` number for the alpha value, or a
            :obj:`torch.Tensor` of shape ``[*, 1, H, W]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_rgba(image, alpha_val)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def bgr_to_rgba(
    image    : torch.Tensor | np.ndarray,
    alpha_val: float | torch.Tensor
) -> torch.Tensor | np.ndarray:
    """Convert a BGR image to RGBA. First convert to RGB, then add alpha channel.

    Args:
        image: A BGR image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        alpha_val: A :obj:`float` number for the alpha value, or a
            :obj:`torch.Tensor` of shape ``[*, 1, H, W]``.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_rgba(rgb, alpha_val)


def rgba_to_rgb(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an RGBA image to RGB.

    Args:
        image: An RGBA image of type:
            - :obj:`torch.Tensor` in ``[*, 4, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 4]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgba_to_rgb(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def rgba_to_bgr(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an RGBA image to BGR. Convert to RGB first, then to BGR.

    Args:
        image: An RGBA image of type:
            - :obj:`torch.Tensor` in ``[*, 4, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 4]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = rgba_to_rgb(image)
    return rgb_to_bgr(rgb)


def rgb_to_linear_rgb(
    image: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to linear RGB.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_linear_rgb(image)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    

def linear_rgb_to_rgb(
    image: torch.Tensor | np.ndarray
) -> torch.Tensor | np.ndarray:
    """Convert a linear RGB image to RGB.

    Args:
        image: A linear RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.linear_rgb_to_rgb(image)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")

# endregion


# region Sepia

def rgb_to_sepia(
    image  : torch.Tensor | np.ndarray,
    rescale: bool  = True,
    eps    : float = 1e-6
) -> torch.Tensor:
    """Convert an RGB image to sepia.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
        rescale: If ``True``, the output tensor will be rescaled (max values be
            ``1.0`` or ``255``).
        eps: Epsilon value to avoid division by zero. Defaults: ``1e-6``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.sepia_from_rgb(image, rescale, eps)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")
    
# endregion


# region XYZ

def rgb_to_xyz(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to XYZ.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_xyz(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def xyz_to_rgb(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an XYZ image to RGB.

    Args:
        image: An XYZ image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.xyz_to_rgb(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_XYZ2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def bgr_to_xyz(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a BGR image to XYZ.

    Args:
        image: A BGR image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_xyz(rgb)


def xyz_to_bgr(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an XYZ image to BGR.

    Args:
        image: An XYZ image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = xyz_to_rgb(image)
    return rgb_to_bgr(rgb)

# endregion


# region YCbCr

def rgb_to_ycbcr(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to YCbCr.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_ycbcr(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def bgr_to_ycbcr(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a BGR image to YCbCr.
    
    Args:
        image: A BGR image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_ycbcr(rgb)


def rgb_to_y(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to Y.
    
    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        ycbcr = kornia.color.rgb_to_ycbcr(image)
        return ycbcr[:, 0, :, :]
    elif isinstance(image, np.ndarray):
        ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        return ycbcr[:, :, 0]
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def bgr_to_y(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a BGR image to Y.
    
    Args:
        image: A BGR image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_y(rgb)
    

def ycbcr_to_rgb(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an YCbCr image to RGB.
    
    Args:
        image: An YCbCr image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.ycbcr_to_rgb(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_YCrCb2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def ycbcr_to_bgr(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an YCbCr image to BGR.
    
    Args:
        image: An YCbCr image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = ycbcr_to_rgb(image)
    return rgb_to_bgr(rgb)
    
# endregion


# region YUV

def rgb_to_yuv(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an RGB image to YUV.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.rgb_to_yuv(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def bgr_to_yuv(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a BGR image to YUV.

    Args:
        image: A BGR image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = bgr_to_rgb(image)
    return rgb_to_yuv(rgb)
    

def yuv_to_rgb(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert an YUV image to RGB.

    Args:
        image: An RGB image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    if isinstance(image, torch.Tensor):
        return kornia.color.yuv_to_rgb(image)
    elif isinstance(image, np.ndarray):
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    else:
        raise TypeError(f"`image` must be a `torch.Tensor` or `numpy.ndarray`, "
                        f"but got {type(image)}.")


def yuv_to_bgr(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Convert a BGR image to RGB.

    Args:
        image: A BGR image of type:
            - :obj:`torch.Tensor` in ``[*, 3, H, W]`` format with data in the
                range ``[0.0, 1.0]``.
            - :obj:`numpy.ndarray` in ``[H, W, 3]`` format with data in the
                range ``[0, 255]``.
    """
    rgb = yuv_to_rgb(image)
    return rgb_to_bgr(rgb)
    
# endregion
