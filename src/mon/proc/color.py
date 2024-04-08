#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements basic color functions."""

from __future__ import annotations

__all__ = [
    # Grayscale
    "bgr_to_grayscale",
    "grayscale_to_rgb",
    "rgb_to_grayscale",
    # HLS
    "hls_to_rgb",
    "rgb_to_hls",
    # HSV
    "hsv_to_rgb",
    "rgb_to_hsv",
    # Lab
    "lab_to_rgb",
    "rgb_to_lab",
    # LUV
    "luv_to_rgb",
    "rgb_to_luv",
    # RGB
    "bgr_to_rgba",
    "rgb_to_bgr",
    "rgb_to_linear_rgb",
    "rgb_to_rgba",
    "rgba_to_bgr",
    "rgba_to_rgb",
    # Sepia
    "rgb_to_sepia",
    # XYZ
    "rgb_to_xyz",
    "xyz_to_rgb",
    # YCbCr
    "rgb_to_y",
    "rgb_to_ycbcr",
    "ycbcr_to_rgb",
    # YUV
    "rgb_to_yuv",
    "rgb_to_yuv420",
    "rgb_to_yuv422",
    "yuv420_to_rgb",
    "yuv422_to_rgb",
    "yuv_to_rgb",
]

import math

import torch


# region Grayscale

def grayscale_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a grayscale image to RGB version of image.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: grayscale image tensor to be converted to RGB with shape :math:`[*,1,H,W]`.

    Returns:
        RGB version of the image with shape :math:`[*,3,H,W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 1, H, W]`, but got {image.shape}.")
    
    return torch.cat([image, image, image], -3)


def rgb_to_grayscale(image: torch.Tensor, rgb_weights: torch.Tensor | None = None) -> torch.Tensor:
    r"""Convert a RGB image to grayscale version of image.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB image to be converted to grayscale with shape :math:`[*, 3, H, W]`.
        rgb_weights: Weights that will be applied on each channel (RGB).
            The sum of the weights should add up to one.
            
    Returns:
        grayscale version of the image with shape :math:`[*, 1, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    if rgb_weights is None:
        # 8 bit images
        if image.dtype == torch.uint8:
            rgb_weights = torch.tensor([76, 150, 29], device=image.device, dtype=torch.uint8)
        # floating point images
        elif image.dtype in (torch.float16, torch.float32, torch.float64):
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
        else:
            raise TypeError(f"Unknown data type: {image.dtype}")
    else:
        # is tensor that we make sure is in the same device/dtype
        rgb_weights = rgb_weights.to(image)

    # unpack the color image channels with RGB order
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    w_r, w_g, w_b = rgb_weights.unbind()
    return w_r * r + w_g * g + w_b * b


def bgr_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to grayscale.

    The image data is assumed to be in the range of (0, 1). First flips to RGB, then converts.

    Args:
        image: BGR image to be converted to grayscale with shape :math:`[*, 3, H, W]`.

    Returns:
        grayscale version of the image with shape :math:`[*, 1, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 1:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 1, H, W]`, but got {image.shape}.")
    
    image_rgb = bgr_to_rgb(image)
    return rgb_to_grayscale(image_rgb)

# endregion


# region HSL

def rgb_to_hls(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an RGB image to HLS.

    The image data is assumed to be in the range of (0, 1).

    NOTE: this method cannot be compiled with JIT in pytohrch < 1.7.0

    Args:
        image: RGB image to be converted to HLS with shape :math:`[*, 3, H, W]`.
        eps: epsilon value to avoid div by zero.

    Returns:
        HLS version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    _RGB2HSL_IDX = torch.tensor([[[0.0]], [[1.0]], [[2.0]]], device=image.device, dtype=image.dtype)  # 3x1x1
    
    _img_max = image.max(-3)
    maxc     = _img_max[0]
    imax     = _img_max[1]
    minc     = image.min(-3)[0]

    if image.requires_grad:
        l_ = maxc + minc
        s  = maxc - minc
        # weird behaviour with undefined vars in JIT...
        # scripting requires image_hls be defined even if it is not used :S
        h = l_  # assign to any tensor...
        image_hls = l_  # assign to any tensor...
    else:
        # define the resulting image to avoid the torch.stack([h, l, s])
        # so, h, l and s require inplace operations
        # NOTE: stack() increases in a 10% the cost in colab
        image_hls = torch.empty_like(image)
        h, l_, s  = image_hls[..., 0, :, :], image_hls[..., 1, :, :], image_hls[..., 2, :, :]
        torch.add(maxc, minc, out=l_)  # l = max + min
        torch.sub(maxc, minc, out=s)  # s = max - min

    # precompute image / (max - min)
    im = image / (s + eps).unsqueeze(-3)

    # epsilon cannot be inside the torch.where to avoid precision issues
    s  /= torch.where(l_ < 1.0, l_, 2.0 - l_) + eps  # saturation
    l_ /= 2  # luminance

    # note that r,g and b were previously div by (max - min)
    r, g, b = im[..., 0, :, :], im[..., 1, :, :], im[..., 2, :, :]
    # h[imax == 0] = (((g - b) / (max - min)) % 6)[imax == 0]
    # h[imax == 1] = (((b - r) / (max - min)) + 2)[imax == 1]
    # h[imax == 2] = (((r - g) / (max - min)) + 4)[imax == 2]
    cond = imax[..., None, :, :] == _RGB2HSL_IDX
    if image.requires_grad:
        h = ((g - b) % 6) * cond[..., 0, :, :]
    else:
        # replacing `torch.mul` with `out=h` with python * operator gives wrong results
        torch.mul((g - b) % 6, cond[..., 0, :, :], out=h)
    h += (b - r + 2) * cond[..., 1, :, :]
    h += (r - g + 4) * cond[..., 2, :, :]
    # h = 2.0 * math.pi * (60.0 * h) / 360.0
    h *= math.pi / 3.0  # hue [0, 2*pi]

    if image.requires_grad:
        return torch.stack([h, l_, s], -3)
    return image_hls


def hls_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a HLS image to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: HLS image to be converted to RGB with shape :math:`[*, 3, H, W]`.

    Returns:
        RGB version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    _HLS2RGB = torch.tensor([[[0.0]], [[8.0]], [[4.0]]], device=image.device, dtype=image.dtype)  # 3x1x1
    
    im   = image.unsqueeze(-4)
    h_ch = im[..., 0, :, :]
    l_ch = im[..., 1, :, :]
    s_ch = im[..., 2, :, :]
    h_ch = h_ch * (6 / math.pi)  # h * 360 / (2 * math.pi) / 30
    a    = s_ch * torch.min(l_ch, 1.0 - l_ch)

    # kr = (0 + h) % 12
    # kg = (8 + h) % 12
    # kb = (4 + h) % 12
    k = (h_ch + _HLS2RGB) % 12

    # l - a * max(min(min(k - 3.0, 9.0 - k), 1), -1)
    mink = torch.min(k - 3.0, 9.0 - k)
    return torch.addcmul(l_ch, a, mink.clamp_(min=-1.0, max=1.0), value=-1)

# endregion


# region HSV

def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`[*, 3, H, W]`.
        eps: scalar to enforce numerical stability.

    Returns:
        HSV version of the image with shape of :math:`[*, 3, H, W]`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2.0 * math.pi * h  # we return 0/2pi output
    
    return torch.stack((h, s, v), dim=-3)


def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`[*, 3, H, W]`.

    Returns:
        RGB version of the image with shape of :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    h = image[..., 0, :, :] / (2 * math.pi)
    s = image[..., 1, :, :]
    v = image[..., 2, :, :]

    hi  = torch.floor(h * 6) % 6
    f   = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p   = v * (one - s)
    q   = v * (one - f * s)
    t   = v * (one - (one - f) * s)

    hi      = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out     = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out     = torch.gather(out, -3, indices)

    return out

# endregion


# region Lab

def rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to Lab.

    The input RGB image is assumed to be in the range of :math:`[0, 1]`. Lab
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Lab with shape :math:`[*, 3, H, W]`.

    Returns:
        Lab version of the image with shape :math:`[*, 3, H, W]`.
        The L channel values are in the range 0..100. a and b are in the range -128..127.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)
    
    xyz_im  = rgb_to_xyz(lin_rgb)
    
    # normalize for D65 white point
    xyz_ref_white  = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None]
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)
    
    threshold = 0.008856
    power     = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale     = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int   = torch.where(xyz_normalized > threshold, power, scale)
    
    x = xyz_int[..., 0, :, :]
    y = xyz_int[..., 1, :, :]
    z = xyz_int[..., 2, :, :]
    
    L  = (116.0 * y) - 16.0
    a  = 500.0 * (x - y)
    _b = 200.0 * (y - z)
    
    out = torch.stack([L, a, _b], dim=-3)
    return out


def lab_to_rgb(image: torch.Tensor, clip: bool = True) -> torch.Tensor:
    r"""Convert a Lab image to RGB.

    The L channel is assumed to be in the range of :math:`[0, 100]`.
    a and b channels are in the range of :math:`[-128, 127]`.

    Args:
        image: Lab image to be converted to RGB with shape :math:`[*, 3, H, W]`.
        clip: Whether to apply clipping to insure output RGB values in range :math:`[0, 1]`.

    Returns:
        Lab version of the image with shape :math:`[*, 3, H, W]`.
        The output RGB image are in the range of :math:`[0, 1]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    L  = image[..., 0, :, :]
    a  = image[..., 1, :, :]
    _b = image[..., 2, :, :]
    
    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)
    
    # if color data out of range: Z < 0
    fz   = fz.clamp(min=0.0)
    
    fxyz = torch.stack([fx, fy, fz], dim=-3)
    
    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz   = torch.where(fxyz > 0.2068966, power, scale)
    
    # For D65 white point
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype)[..., :, None, None]
    xyz_im        = xyz * xyz_ref_white
    
    rgbs_im = xyz_to_rgb(xyz_im)
    
    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)
    
    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)
    
    # Clip to 0,1 https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)
    
    return rgb_im

# endregion


# region Luv

def rgb_to_luv(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Convert an RGB image to Luv.

    The image data is assumed to be in the range of :math:`[0, 1]`. Luv
    color is computed using the D65 illuminant and Observer 2.

    Args:
        image: RGB Image to be converted to Luv with shape :math:`[*, 3, H, W]`.
        eps: for numerically stability when dividing.

    Returns:
        Luv version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)

    xyz_im  = rgb_to_xyz(lin_rgb)

    x = xyz_im[..., 0, :, :]
    y = xyz_im[..., 1, :, :]
    z = xyz_im[..., 2, :, :]

    threshold = 0.008856
    L = torch.where(y > threshold, 116.0 * torch.pow(y.clamp(min=threshold), 1.0 / 3.0) - 16.0, 903.3 * y)

    # Compute reference white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w = (4 * xyz_ref_white[0]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w = (9 * xyz_ref_white[1]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    u_p = (4 * x) / (x + 15 * y + 3 * z + eps)
    v_p = (9 * y) / (x + 15 * y + 3 * z + eps)

    u = 13 * L * (u_p - u_w)
    v = 13 * L * (v_p - v_w)

    out = torch.stack([L, u, v], dim=-3)

    return out


def luv_to_rgb(image: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Convert a Luv image to RGB.

    Args:
        image: Luv image to be converted to RGB with shape :math:`[*, 3, H, W]`.
        eps: for numerically stability when dividing.

    Returns:
        Luv version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    L = image[..., 0, :, :]
    u = image[..., 1, :, :]
    v = image[..., 2, :, :]

    # Convert from Luv to XYZ
    y: torch.Tensor = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0), L / 903.3)

    # Compute white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w = (4 * xyz_ref_white[0]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])
    v_w = (9 * xyz_ref_white[1]) / (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2])

    a = u_w + u / (13 * L + eps)
    d = v_w + v / (13 * L + eps)
    c = 3 * y * (5 * d - 3)
    
    z = ((a - 4) * c - 15 * a * d * y) / (12 * d + eps)
    x = -(c / (d + eps) + 3.0 * z)

    xyz_im  = torch.stack([x, y, z], -3)

    rgbs_im = xyz_to_rgb(xyz_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    return rgb_im

# endregion


# region RGB

def rgb_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to BGR.
    
    Args:
        image: RGB Image to be converted to BGRof of shape :math:`[*, 3, H, W]`.

    Returns:
        BGR version of the image with shape of shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    return bgr_to_rgb(image)


def bgr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a BGR image to RGB.

    Args:
        image: BGR Image to be converted to BGR of shape :math:`[*, 3, H, W]`.

    Returns:
        RGB version of the image with shape of shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    # flip image channels
    out: torch.Tensor = image.flip(-3)
    return out


def rgb_to_rgba(image: torch.Tensor, alpha_val: float | torch.Tensor) -> torch.Tensor:
    r"""Convert an image from RGB to RGBA.

    Args:
        image: RGB Image to be converted to RGBA of shape :math:`[*, 3, H, W]`.
        alpha_val: A float number for the alpha value, or a tensor of shape :math:`[*, 1, H, W]`.

    Returns:
        RGBA version of the image with shape :math:`[*, 4, H, W]`.

    .. note:: The current functionality is NOT supported by Torchscript.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"``alpha_val`` type is not a float or ``torch.Tensor``, but got {type(alpha_val)}.")

    # add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)

    a = torch.cast(torch.Tensor, alpha_val)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))

    return torch.cat([r, g, b, a], dim=-3)


def bgr_to_rgba(image: torch.Tensor, alpha_val: float | torch.Tensor) -> torch.Tensor:
    r"""Convert an image from BGR to RGBA.

    Args:
        image: BGR Image to be converted to RGBA of shape :math:`[*, 3, H, W]`.
        alpha_val: A float number for the alpha value or a tensor of shape :math:`[*, 1, H, W]`.

    Returns:
        RGBA version of the image with shape :math:`[*, 4, H, W]`.

    .. note:: The current functionality is NOT supported by Torchscript.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    if not isinstance(alpha_val, (float, torch.Tensor)):
        raise TypeError(f"``alpha_val`` type is not a float or ``torch.Tensor``, but got {type(alpha_val)}.")

    # convert first to RGB, then add alpha channel
    x_rgb = bgr_to_rgb(image)
    return rgb_to_rgba(x_rgb, alpha_val)


def rgba_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from RGBA to RGB.

    Args:
        image: RGBA Image to be converted to RGB of shape :math:`[*, 4, H, W]`.

    Returns:
        RGB version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 4, H, W]`, but got {image.shape}.")
    
    # unpack channels
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)

    # compute new channels
    a_one = torch.tensor(1.0) - a
    r_new = a_one * r + a * r
    g_new = a_one * g + a * g
    b_new = a_one * b + a * b

    return torch.cat([r_new, g_new, b_new], dim=-3)


def rgba_to_bgr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from RGBA to BGR.

    Args:
        image: RGBA Image to be converted to BGR of shape :math:`[*, 4, H, W]`.

    Returns:
        RGB version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 4, H, W]`, but got {image.shape}.")
    
    # convert to RGB first, then to BGR
    x_rgb = rgba_to_rgb(image)
    return rgb_to_bgr(x_rgb)


def rgb_to_linear_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an sRGB image to linear RGB. Used in colorspace conversions.

    Args:
        image: sRGB Image to be converted to linear RGB of shape :math:`[*, 3, H, W]`.

    Returns:
        linear RGB version of the image with shape of :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    lin_rgb = torch.where(image > 0.04045, torch.pow(((image + 0.055) / 1.055), 2.4), image / 12.92)
   
    return lin_rgb


def linear_rgb_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a linear RGB image to sRGB. Used in colorspace conversions.

    Args:
        image: linear RGB Image to be converted to sRGB of shape :math:`[*, 3, H, W]`.

    Returns:
        sRGB version of the image with shape of shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    threshold = 0.0031308
    rgb = torch.where(
        image > threshold, 1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055, 12.92 * image
    )

    return rgb

# endregion


# region Sepia

def rgb_to_sepia(image: torch.Tensor, rescale: bool = True, eps: float = 1e-6) -> torch.Tensor:
    r"""Apply to a tensor the sepia filter.

    Args:
        image: the input tensor with shape of :math:`[*, C, H, W]`.
        rescale: If True, the output tensor will be rescaled (max values be 1. or 255).
        eps: scalar to enforce numerical stability.

    Returns:
        Tensor: The sepia tensor of same size and numbers of channels
        as the input with shape :math:`[*, C, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]
    
    r_out = 0.393 * r + 0.769 * g + 0.189 * b
    g_out = 0.349 * r + 0.686 * g + 0.168 * b
    b_out = 0.272 * r + 0.534 * g + 0.131 * b
    
    sepia_out = torch.stack([r_out, g_out, b_out], dim=-3)
    
    if rescale:
        max_values = sepia_out.amax(dim=-1).amax(dim=-1)
        sepia_out = sepia_out / (max_values[..., None, None] + eps)
    
    return sepia_out

# endregion


# region XYZ

def rgb_to_xyz(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to XYZ.

    Args:
        image: RGB Image to be converted to XYZ with shape :math:`[*, 3, H, W]`.

    Returns:
         XYZ version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    out = torch.stack([x, y, z], -3)

    return out


def xyz_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a XYZ image to RGB.

    Args:
        image: XYZ Image to be converted to RGB with shape :math:`[*, 3, H, W]`.

    Returns:
        RGB version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    x = image[..., 0, :, :]
    y = image[..., 1, :, :]
    z = image[..., 2, :, :]
    
    r = 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z
    g = -0.9692549499965682 * x + 1.8759900014898907 * y + 0.0415559265582928 * z
    b = 0.0556466391351772 * x + -0.2040413383665112 * y + 1.0573110696453443 * z

    out: torch.Tensor = torch.stack([r, g, b], dim=-3)

    return out

# endregion


# region YCbCr

def _rgb_to_y(r: torch.Tensor, g: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y


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


# region YUV

def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`[*, 3, H, W]`.

    Returns:
        YUV version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    y =  0.299 * r + 0.587 * g + 0.114 * b
    u = -0.147 * r - 0.289 * g + 0.436 * b
    v =  0.615 * r - 0.515 * g - 0.100 * b

    out = torch.stack([y, u, v], -3)

    return out


def rgb_to_yuv420(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert an RGB image to YUV 420 (subsampled).

    The image data is assumed to be in the range of (0, 1). Input need to be
    padded to be evenly divisible by 2 horizontal and vertical. This function
    will output chroma siting (0.5, 0.5).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`[*, 3, H, W]`.

    Returns:
        A Tensor containing the Y plane with shape :math:`[*, 1, H, W]`.
        A Tensor containing the UV planes with shape :math:`[*, 2, H/2, W/2]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly divisible by 2, but got {image.shape}.")

    yuvimage = rgb_to_yuv(image)

    return yuvimage[..., :1, :, :], yuvimage[..., 1:3, :, :].unfold(-2, 2, 2).unfold(-2, 2, 2).mean((-1, -2))


def rgb_to_yuv422(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Convert an RGB image to YUV 422 (subsampled).

    The image data is assumed to be in the range of (0, 1). Input needs to be
    padded to be evenly divisible by vertical 2. This function will output
    chroma sitting (0.5).

    Args:
        image: RGB Image to be converted to YUV with shape :math:`[*, 3, H, W]`.

    Returns:
       A Tensor containing the Y plane with shape :math:`[*, 1, H, W]`
       A Tensor containing the UV planes with shape :math:`[*, 2, H, W/2]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
    
    if len(image.shape) < 2 or image.shape[-2] % 2 == 1 or image.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly divisible by 2, but got {image.shape}.")

    yuvimage = rgb_to_yuv(image)

    return yuvimage[..., :1, :, :], yuvimage[..., 1:3, :, :].unfold(-1, 2, 2).mean(-1)


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.

    Args:
        image: YUV Image to be converted to RGB with shape :math:`[*, 3, H, W]`.

    Returns:
        RGB version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"``image`` is not a ``torch.Tensor``, but got {type(image)}.")
    
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"``image`` must have a shape of :math:`[*, 3, H, W]`, but got {image.shape}.")
        
    y = image[..., 0, :, :]
    u = image[..., 1, :, :]
    v = image[..., 2, :, :]
    
    r = y + 1.14 * v  # coefficient for g is 0
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u  # coefficient for b is 0

    out = torch.stack([r, g, b], -3)

    return out


def yuv420_to_rgb(imagey: torch.Tensor, imageuv: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV420 image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Input need to be padded to be evenly divisible by 2 horizontal and vertical.
    This function assumed chroma siting is (0.5, 0.5)

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`[*, 1, H, W]`.
        imageuv: UV (chroma) Image planes to be converted to RGB with shape :math:`[*, 2, H/2, W/2]`.

    Returns:
        RGB version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(imagey, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(imagey)}")

    if not isinstance(imageuv, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(imageuv)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imageuv.shape) < 3 or imageuv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of (*, 2, H/2, W/2). Got {imageuv.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if (
        len(imageuv.shape) < 2
        or len(imagey.shape) < 2
        or imagey.shape[-2] / imageuv.shape[-2] != 2
        or imagey.shape[-1] / imageuv.shape[-1] != 2
    ):
        raise ValueError(
            f"Input imageuv H&W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat([imagey, imageuv.repeat_interleave(2, dim=-1).repeat_interleave(2, dim=-2)], dim=-3)
    # then convert the yuv444 tensor

    return yuv_to_rgb(yuv444image)


def yuv422_to_rgb(imagey: torch.Tensor, imageuv: torch.Tensor) -> torch.Tensor:
    r"""Convert an YUV422 image to RGB.

    The image data is assumed to be in the range of (0, 1) for luma and (-0.5, 0.5) for chroma.
    Input need to be padded to be evenly divisible by 2 vertical. This function assumed chroma siting is (0.5)

    Args:
        imagey: Y (luma) Image plane to be converted to RGB with shape :math:`[*, 1, H, W]`.
        imageuv: UV (luma) Image planes to be converted to RGB with shape :math:`[*, 2, H, W/2]`.

    Returns:
        RGB version of the image with shape :math:`[*, 3, H, W]`.
    """
    if not isinstance(imagey, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(imagey)}")

    if not isinstance(imageuv, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(imageuv)}")

    if len(imagey.shape) < 3 or imagey.shape[-3] != 1:
        raise ValueError(f"Input imagey size must have a shape of (*, 1, H, W). Got {imagey.shape}")

    if len(imageuv.shape) < 3 or imageuv.shape[-3] != 2:
        raise ValueError(f"Input imageuv size must have a shape of (*, 2, H, W/2). Got {imageuv.shape}")

    if len(imagey.shape) < 2 or imagey.shape[-2] % 2 == 1 or imagey.shape[-1] % 2 == 1:
        raise ValueError(f"Input H&W must be evenly disible by 2. Got {imagey.shape}")

    if len(imageuv.shape) < 2 or len(imagey.shape) < 2 or imagey.shape[-1] / imageuv.shape[-1] != 2:
        raise ValueError(
            f"Input imageuv W must be half the size of the luma plane. Got {imagey.shape} and {imageuv.shape}"
        )

    # first upsample
    yuv444image = torch.cat([imagey, imageuv.repeat_interleave(2, dim=-1)], dim=-3)
    # then convert the yuv444 tensor
    return yuv_to_rgb(yuv444image)

# endregion
