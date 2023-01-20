#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements functions for handling colors in images. Here we focus
on two main color formats: RGB (default) and BGR (cv2). """

from __future__ import annotations

__all__ = [
    "bgr_to_grayscale", "bgr_to_hsv", "bgr_to_lab", "bgr_to_luv", "bgr_to_rgb",
    "bgr_to_rgba", "bgr_to_xyz", "bgr_to_ycrcb", "bgr_to_yuv",
    "grayscale_to_rgb", "hsv_to_bgr", "hsv_to_rgb", "lab_to_bgr", "lab_to_rgb",
    "linear_rgb_to_rgb", "luv_to_bgr", "luv_to_rgb", "rgb_to_bgr",
    "rgb_to_grayscale", "rgb_to_hsv", "rgb_to_lab", "rgb_to_linear_rgb",
    "rgb_to_luv", "rgb_to_rgba", "rgb_to_xyz", "rgb_to_ycrcb", "rgb_to_yuv",
    "rgb_to_yuv420", "rgb_to_yuv422", "rgba_to_bgr", "rgba_to_rgb",
    "xyz_to_bgr", "xyz_to_rgb", "ycrcb_to_bgr", "ycrcb_to_rgb", "yuv420_to_rgb",
    "yuv422_to_rgb", "yuv_to_bgr", "yuv_to_rgb",
]

from typing import cast, Sequence

import torch
from torch.nn import functional

from mon.coreimage import util
from mon.coreimage.typing import Float3T
from mon.foundation import math


# region BGR

def bgr_to_grayscale(
    image      : torch.Tensor,
    rgb_weights: Float3T | torch.Tensor | None = (0.299, 0.587, 0.114),
) -> torch.Tensor:
    """Convert an image from BGR to grayscale.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed.
        rgb_weights: The weights applying on each channel (RGB). Sum of
            the weights should add up to 1.0 ([0.299, 0.587, 0.114] (normalized)
            or 255 ([76, 150, 29]). Defaults to (0.299, 0.587, 0.114).
        
    Returns:
        A grayscale image of shape [..., 1, H, W].
    """
    rgb       = bgr_to_rgb(image=image)
    grayscale = rgb_to_grayscale(image=rgb, rgb_weights=rgb_weights)
    return grayscale


def bgr_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert an image from BGR to HSV.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0]..
        eps: A scalar to enforce numerical stability. Defaults to 1e-8.

    Returns:
        A HSV image of shape [..., 3, H, W]. H channel values are in the range
        [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    rgb = bgr_to_rgb(image=image)
    hsv = rgb_to_hsv(image=rgb, eps=eps)
    return hsv


def bgr_to_lab(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from BGR to Lab. Lab color is computed using the D65
    illuminant and Observer 2.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A Lab image of shape [..., 3, H, W]. L channel values are in the range
        [0, 100]. a and b are in the range [-127, 127].
    """
    rgb = bgr_to_rgb(image=image)
    lab = rgb_to_lab(image=rgb)
    return lab


def bgr_to_luv(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert an image from BGR to Luv. Luv color is computed using the
    D65 illuminant and Observer 2.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        eps: A scalar to enforce numerical stability. Defaults to 1e-12.

    Returns:
        A Luv image of shape [..., 3, H, W].
    """
    rgb = bgr_to_rgb(image=image)
    luv = rgb_to_luv(image=rgb, eps=eps)
    return luv


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from BGR to RGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed, The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) \
           and image.ndim >= 3 and image.shape[-3] == 3
    rgb = image.flip(-3)
    return rgb


def bgr_to_rgba(
    image    : torch.Tensor,
    alpha_val: float | torch.Tensor
) -> torch.Tensor:
    """Convert an image from BGR to RGBA. Convert first to RGB, then add an
    alpha channel.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        alpha_val: A float number or tensor for the alpha value.

    Returns:
        A RGBA image of shape [..., 3, H, W].
    """
    rgb  = bgr_to_rgb(image=image)
    rgba = rgb_to_rgba(image=rgb, alpha_val=alpha_val)
    return rgba


def bgr_to_xyz(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from BGR to XYZ.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A XYZ image of shape [..., 3, H, W].
    """
    rgb = bgr_to_rgb(image=image)
    xyz = bgr_to_xyz(image=rgb)
    return xyz


def bgr_to_ycrcb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from BGR to YCrCb.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        An YCrCb image of shape [..., 3, H, W].
    """
    rgb   = bgr_to_rgb(image=image)
    ycrcb = rgb_to_ycrcb(image=rgb)
    return ycrcb


def bgr_to_yuv(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from BGR to YUV.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        An YUV image of shape [..., 3, H, W].
    """
    rgb = bgr_to_rgb(image=image)
    yuv = rgb_to_yuv(image=rgb)
    return yuv

# endregion


# region Grayscale

def grayscale_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from grayscale to RGB.
    
    Args:
        image: An image of shape [..., 1, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        
    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) \
           and image.ndim >= 3 and image.shape[-3] == 1
    rgb = torch.cat([image, image, image], dim=-3)
    return rgb

# endregion


# region HSV

def hsv_to_bgr(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from HSV to BGR.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0]. H channel values are in the
            range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Returns:
        A BGR image of shape [..., 3, H, W].
    """
    rgb = hsv_to_rgb(image=image)
    bgr = rgb_to_bgr(image=rgb)
    return bgr


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from HSV to RGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0]. H channel values are to be in
            the range [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    h       = image[..., 0, :, :] / (2 * math.pi)
    s       = image[..., 1, :, :]
    v       = image[..., 2, :, :]
    hi      = torch.floor(h * 6) % 6
    f       = ((h * 6) % 6) - hi
    one     = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p       = v * (one - s)
    q       = v * (one - f * s)
    t       = v * (one - (one - f) * s)
    hi      = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    rgb     = torch.stack((
        v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q
    ), dim=-3)
    rgb     = torch.gather(rgb, -3, indices)
    return rgb

# endregion


# region LinearRGB

def linear_rgb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from linear RGB to sRGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A sRGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    threshold = 0.0031308
    rgb       = torch.where(
        image > threshold,
        1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055,
        12.92 * image
    )
    return rgb

# endregion


# region Lab

def lab_to_bgr(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    """Convert an image from Lab to BGR.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        clip: Whether to apply clipping to insure output BGR values in the range
            of [0.0 1.0]. Defaults to True.

    Returns:
        A BGR image of shape [..., 3, H, W].
    """
    rgb = lab_to_rgb(image=image, clip=clip)
    bgr = rgb_to_bgr(image=rgb)
    return bgr
    

def lab_to_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    """Convert an image from  Lab to RGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        clip: Whether to apply clipping to insure output RGB values in the range
            [0.0 1.0]. Defaults to True.

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    L         = image[..., 0, :, :]
    a         = image[..., 1, :, :]
    _b        = image[..., 2, :, :]
    fy        = (L + 16.0) / 116.0
    fx        = (a / 500.0) + fy
    fz        = fy - (_b / 200.0)
    # If color data out of range: Z < 0
    fz        = fz.clamp(min=0.0)
    fxyz      = torch.stack([fx, fy, fz], dim=-3)
    # Convert from Lab to XYZ
    power     = torch.pow(fxyz, 3.0)
    scale     = (fxyz - 4.0 / 29.0) / 7.787
    xyz_image = torch.where(fxyz > 0.2068966, power, scale)
    # For D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883],
        device = xyz_image.device,
        dtype  = xyz_image.dtype
    )[..., :, None, None]
    xyz_im  = xyz_image * xyz_ref_white
    rgbs_im = xyz_to_rgb(xyz_im)
    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    # rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)
    # Convert from RGB Linear to sRGB
    rgb = linear_rgb_to_rgb(rgbs_im)
    # Clip to [0.0, 1.0] https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb = torch.clamp(rgb, min=0.0, max=1.0)
    return rgb

# endregion


# region Luv

def luv_to_bgr(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert an image from Luv to BGR.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        eps: A scalar to enforce numerical stability. Defaults to 1e-12.

    Returns:
        A BGR image of shape [..., 3, H, W].
    """
    rgb = luv_to_rgb(image=image, eps=eps)
    bgr = rgb_to_bgr(image=rgb)
    return bgr


def luv_to_rgb(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert an image from Luv to RGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        eps: A scalar to enforce numerical stability. Defaults to 1e-12.

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.torch.Tensor) and image.shape[-3] == 3
    L       = image[..., 0, :, :]
    u       = image[..., 1, :, :]
    v       = image[..., 2, :, :]
    # Convert from Luv to XYZ
    y       = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0), L / 903.3)
    # Compute white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w     = ((4 * xyz_ref_white[0]) /
               (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    v_w     = ((9 * xyz_ref_white[1]) /
               (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    a       = u_w + u / (13 * L + eps)
    d       = v_w + v / (13 * L + eps)
    c       = 3 * y * (5 * d - 3)
    z       = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x       = -(c / (d + eps) + 3.0 * z)
    xyz_im  = torch.stack([x, y, z], -3)
    rgbs_im = xyz_to_rgb(xyz_im)
    # Convert from RGB Linear to sRGB
    rgb     = linear_rgb_to_rgb(rgbs_im)
    return rgb

# endregion


# region RGB

def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from RGB to BGR.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A BGR image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) \
           and image.ndim >= 3 and image.shape[-3] == 3
    bgr = image.flip(-3)
    return bgr


def rgb_to_grayscale(
    image      : torch.Tensor,
    rgb_weights: Float3T | torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert an image from RGB to grayscale.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        rgb_weights: The weights applying on each channel (RGB). Sum of
            the weights should add up to 1.0 ([0.299, 0.587, 0.114] (normalized)
            or 255 ([76, 150, 29]). Defaults to (0.299, 0.587, 0.114).
        
    Returns:
        A grayscale image of shape [..., 1, H, W].
    """
    assert isinstance(image, torch.Tensor) \
           and image.ndim >= 3 and image.shape[-3] in [1, 3]
    
    if util.get_num_channels(image) == 1:
        return image

    if rgb_weights is None:
        # 8-bit images
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor(
                [76, 150, 29], device=image.device, dtype=torch.uint8
            )
        # Floating-point images
        elif image.dtype in [torch.float16, torch.float32, torch.float64]:
            rgb_weights = torch.tensor(
                [0.299, 0.587, 0.114], device=image.device, dtype=image.dtype
            )
        else:
            raise TypeError(
                f":param:`image` must have value type: `torch.uint8`, "
                f"`torch.float16`, `torch.float32`, or `torch.float64`. "
                f"But got: {image.dtype}."
            )
    elif isinstance(rgb_weights, list | tuple):
        rgb_weights = torch.FloatTensor(rgb_weights)
    if isinstance(rgb_weights, torch.Tensor):
        # We make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # Unpack the color image channels with RGB order
    r             = image[..., 0:1, :, :]
    g             = image[..., 1:2, :, :]
    b             = image[..., 2:3, :, :]
    w_r, w_g, w_b = rgb_weights.unbind()
    grayscale     =  w_r * r + w_g * g + w_b * b
    return grayscale


def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert an image from RGB to HSV.
    
    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        eps: Scalar to enforce numerical stability. Defaults to 1e-8.

    Returns:
        A HSV image of shape [..., 3, H, W]. H channel values are in the range
        [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac              = max_rgb - min_rgb
    v           = max_rgb
    s           = deltac / (max_rgb + eps)
    deltac      = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc  = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)
    h1          = (bc - gc)
    h2          = (rc - bc) + 2.0 * deltac
    h3          = (gc - rc) + 4.0 * deltac
    h           = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h           = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h           = (h / 6.0) % 1.0
    h          *= 2.0 * math.pi  # We return 0/2pi output
    hsv         = torch.stack((h, s, v), dim=-3)
    return hsv


def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from RGB to Lab. Lab color is computed using the
    D65 illuminant and Observer 2.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A Lab image of shape [..., 3, H, W]. L channel values are in the range
        [0, 100]. a and b are in the range [-127, 127].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    
    # Convert from sRGB to Linear RGB
    linear_rgb = rgb_to_linear_rgb(image=image)
    xyz_im     = rgb_to_xyz(image=linear_rgb)

    # Normalize for D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883],
        device = xyz_im.device,
        dtype  = xyz_im.dtype
    )[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    threshold = 0.008856
    power     = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale     = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int   = torch.where(xyz_normalized > threshold, power, scale)
    x         = xyz_int[..., 0, :, :]
    y         = xyz_int[..., 1, :, :]
    z         = xyz_int[..., 2, :, :]
    L         = (116.0 * y) - 16.0
    a         = 500.0 * (x - y)
    _b        = 200.0 * (y - z)
    lab       = torch.stack([L, a, _b], dim=-3)
    return lab


def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from RGB to linear RGB.

    Args:
       image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A Linear RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    linear_rgb = torch.where(
        image > 0.04045,
        torch.pow(((image + 0.055) / 1.055), 2.4),
        image / 12.92
    )
    return linear_rgb


def rgb_to_luv(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert an image from RGB to Luv. Luv color is computed using the
    D65 illuminant and Observer 2.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        eps: Scalar to enforce numerical stability. Defaults to 1e-12.

    Returns:
        A Luv image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    
    # Convert from sRGB to Linear RGB
    linear_rgb = rgb_to_linear_rgb(image=image)
    xyz_im     = rgb_to_xyz(image=linear_rgb)
    x          = xyz_im[..., 0, :, :]
    y          = xyz_im[..., 1, :, :]
    z          = xyz_im[..., 2, :, :]
    threshold  = 0.008856
    L = torch.where(
        y > threshold,
        116.0 * torch.pow(y.clamp(min=threshold), 1.0 / 3.0) - 16.0,
        903.3 * y
    )

    # Compute reference white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w = ((4 * xyz_ref_white[0]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    v_w = ((9 * xyz_ref_white[1]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    u_p = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p = (9 * y) / (x + 15 * y + 3 * z + eps)
    u   = 13 * L * (u_p - u_w)
    v   = 13 * L * (v_p - v_w)
    luv = torch.stack([L, u, v], dim=-3)
    return luv


def rgb_to_rgba(
    image    : torch.Tensor,
    alpha_val: float | torch.Tensor,
) -> torch.Tensor:
    """Convert an image from RGB to RGBA.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].
        alpha_val: A float number or tensor for the alpha value.

    Returns:
        A RGBA image of shape [..., 4, H, W]
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    if not isinstance(alpha_val, float | torch.Tensor):
        raise TypeError(
            f":param:`alpha_val` must be :class:`float` or :class:`torch.Tensor`. "
            f"But got: {type(alpha_val)}."
        )
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)
    a       = cast(torch.Tensor, alpha_val)
    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))
    rgba  = torch.cat([r, g, b, a], dim=-3)
    return rgba


def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from RGB to XYZ.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A XYZ image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    r   = image[..., 0, :, :]
    g   = image[..., 1, :, :]
    b   = image[..., 2, :, :]
    x   = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y   = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z   = 0.019334 * r + 0.119193 * g + 0.950227 * b
    xyz = torch.stack([x, y, z], -3)
    return xyz


def rgb_to_ycrcb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from RGB to YCrCb.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        An YCrCb image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    r     = image[..., 0, :, :]
    g     = image[..., 1, :, :]
    b     = image[..., 2, :, :]
    delta = 0.5
    y     = 0.299 * r + 0.587 * g + 0.114 * b
    cb    = (b - y) * 0.564 + delta
    cr    = (r - y) * 0.713 + delta
    ycrcb = torch.stack([y, cr, cb], -3)
    return ycrcb


def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from RGB to  YUV.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        An YUV image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    r   = image[..., 0, :, :]
    g   = image[..., 1, :, :]
    b   = image[..., 2, :, :]
    y   =  0.299 * r + 0.587 * g + 0.114 * b
    u   = -0.147 * r - 0.289 * g + 0.436 * b
    v   =  0.615 * r - 0.515 * g - 0.100 * b
    yuv = torch.stack([y, u, v], -3)
    return yuv


def rgb_to_yuv420(image: torch.Tensor) -> Sequence[torch.Tensor]:
    """Convert an image from RGB to YUV 420 (sub-sampled). The Input needs to
    be padded to be evenly divisible by 2 horizontal and vertical. This function
    will output chroma siting [0.5, 0.5].

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A Tensor containing the Y plane with shape [..., 1, H, W]
        A Tensor containing the UV planes with shape [..., 2, H/2, W/2]
    """
    assert isinstance(image, torch.Tensor) \
           and image.ndim >= 3 \
           and image.shape[-3] == 3 \
           and image.shape[-2] % 2 == 1 \
           and image.shape[-1] % 2 == 1
    yuv420 = rgb_to_yuv(image)
    return (
        yuv420[..., :1, :, :],
        functional.avg_pool2d(yuv420[..., 1:3, :, :], (2, 2))
    )


def rgb_to_yuv422(image: torch.Tensor) -> Sequence[torch.Tensor]:
    """Convert an image from RGB to YUV 422 (sub-sampled). The input needs to
    be padded to be evenly divisible by vertical 2. This function will output
    chroma siting (0.5).

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
       A tensor containing the Y plane with shape [..., 1, H, W].
       A tensor containing the UV planes with shape [..., 2, H, W/2].
    """
    assert isinstance(image, torch.Tensor) \
           and image.ndim >= 3 \
           and image.shape[-3] == 3 \
           and image.shape[-2] % 2 == 1 \
           and image.shape[-1] % 2 == 1
    yuv420 = rgb_to_yuv(image)
    return (
        yuv420[..., :1, :, :],
        functional.avg_pool2d(yuv420[..., 1:3, :, :], (1, 2))
    )

# endregion


# region RGBA

def rgba_to_bgr(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from RGBA to BGR. Convert to RGB first, then to BGR.

    Args:
        image: An image of shape [..., 4, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A BGR image of shape [..., 3, H, W].
    """
    rgb = rgba_to_rgb(image=image)
    bgr = rgb_to_bgr(image=rgb)
    return bgr


def rgba_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from RGBA to RGB.

    Args:
        image: An image of shape [..., 4, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 4
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)
    a_one      = torch.tensor(1.0) - a
    r_new      = a_one * r + a * r
    g_new      = a_one * g + a * g
    b_new      = a_one * b + a * b
    rgb        = torch.cat([r_new, g_new, b_new], dim=-3)
    return rgb

# endregion


# region XYZ

def xyz_to_bgr(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from XYZ to BGR.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A BGR image of shape [..., 3, H, W].
    """
    rgb = xyz_to_rgb(image=image)
    bgr = rgb_to_bgr(image=rgb)
    return bgr


def xyz_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from XYZ to RGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    x   = image[..., 0, :, :]
    y   = image[..., 1, :, :]
    z   = image[..., 2, :, :]
    r   = ( 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z)
    g   = (-0.9692549499965682 * x +  1.8759900014898907 * y +  0.0415559265582928 * z)
    b   = ( 0.0556466391351772 * x + -0.2040413383665112 * y +  1.0573110696453443 * z)
    rgb = torch.stack([r, g, b], dim=-3)
    return rgb

# endregion


# region YCrCb

def ycrcb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from YCrCb to BGR.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A BGR image of shape [..., 3, H, W].
    """
    rgb = ycrcb_to_rgb(image=image)
    bgr = rgb_to_bgr(image=rgb)
    return bgr


def ycrcb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from YCrCb to RGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are in the range of [0.0, 1.0].

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.Tensor) and image.shape[-3] == 3
    y          = image[..., 0, :, :]
    cb         = image[..., 1, :, :]
    cr         = image[..., 2, :, :]
    delta      = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta
    r          = y + 1.403 * cr_shifted
    g          = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b          = y + 1.773 * cb_shifted
    rgb        = torch.stack([r, g, b], -3)
    return rgb

# endregion


# region YUV

def yuv_to_bgr(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from YUV to RGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are assumed to be in the range of [0.0, 1.0] for luma and
            [-0.5, 0.5] for chroma.

    Returns:
        A BGR image of shape [..., 3, H, W].
    """
    rgb = yuv_to_rgb(image=image)
    bgr = rgb_to_bgr(image=rgb)
    return bgr


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert an image from YUV to RGB.

    Args:
        image: An image of shape [..., 3, H, W] to be transformed. The image
            pixels are assumed to be in the range of [0.0, 1.0] for luma and
            [-0.5, 0.5] for chroma.

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image, torch.torch.Tensor) and image.shape[-3] == 3
    y   = image[..., 0, :, :]
    u   = image[..., 1, :, :]
    v   = image[..., 2, :, :]
    r   = y + 1.14 * v  # coefficient for g is 0
    g   = y + -0.396 * u - 0.581 * v
    b   = y + 2.029 * u  # coefficient for b is 0
    rgb = torch.stack([r, g, b], -3)
    return rgb


def yuv420_to_rgb(image_y: torch.Tensor, image_uv: torch.Tensor) -> torch.Tensor:
    """Convert an image from YUV420 to RGB. The image data is assumed to be
    in the range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Input needs
    to be padded to be evenly divisible by 2 horizontal and vertical. This
    function assumed chroma siting is [0.5, 0.5].

    Args:
        image_y: Y (luma) image plane of shape [..., 1, H, W] to be converted to
            RGB.
        image_uv: UV (chroma) image planes of shape [..., 2, H/2, W/2] to be
            converted to RGB.

    Returns:
        A RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image_y,  torch.torch.Tensor) and image_y.shape[-3]  == 1
    assert isinstance(image_uv, torch.torch.Tensor) and image_uv.shape[-3] == 2
    if (len(image_y.shape) < 2 or
        image_y.shape[-2] % 2 == 1 or
        image_y.shape[-1] % 2 == 1):
        raise ValueError(
            f"`image_y` H, W must be evenly divisible by 2. "
            f"But got: {image_y.shape}."
        )
    if (len(image_uv.shape) < 2 or
        len(image_y.shape) < 2 or
        image_y.shape[-2] / image_uv.shape[-2] != 2 or
        image_y.shape[-1] / image_uv.shape[-1] != 2):
        raise ValueError(
            f"`image_uv` H, W must be half the size of the luma "
            f"plane. But got: {image_y.shape} and {image_uv.shape}."
        )
    # First upsample
    yuv444image = torch.cat([
        image_y, image_uv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)
    ], dim=-3)
    # Then convert the yuv444 image
    return yuv_to_rgb(yuv444image)


def yuv422_to_rgb(image_y: torch.Tensor, image_uv: torch.Tensor) -> torch.Tensor:
    """Convert an image from YUV422 to RGB. Image data is assumed to be
    in the range of [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Input need
    to be padded to be evenly divisible by 2 vertical. This function assumed
    chroma siting is (0.5).

    Args:
        image_y: Y (luma) image plane of shape [..., 1, H, W] to be converted to
            RGB, where ... means it can have an arbitrary number of leading
            dimensions.
        image_uv: UV (luma) image planes of shape [..., 2, H/2, W/2] to be
            converted to RGB, where ... means it can have an arbitrary number of
            leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert isinstance(image_y,  torch.torch.Tensor) and image_y.shape[-3]  == 1
    assert isinstance(image_uv, torch.torch.Tensor) and image_uv.shape[-3] == 2
    if (len(image_y.shape) < 2 or
        image_y.shape[-2] % 2 == 1 or
        image_y.shape[-1] % 2 == 1):
        raise ValueError(
            f"`image_y` H, W must be evenly divisible by 2. "
            f"But got: {image_y.shape}."
        )
    if (len(image_uv.shape) < 2 or
        len(image_y.shape) < 2 or
        image_y.shape[-1] / image_uv.shape[-1] != 2):
        raise ValueError(
            f"`image_uv` W must be half the size of the luma "
            f"plane. But got: {image_y.shape} and {image_uv.shape}"
        )
    # First upsample
    yuv444image = torch.cat([image_y, image_uv.repeat_interleave(2, dim=-1)], dim=-3)
    # Then convert the yuv444 image
    return yuv_to_rgb(yuv444image)

# endregion
