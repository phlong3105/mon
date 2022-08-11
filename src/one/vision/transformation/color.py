#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Color space conversion.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from torch.nn import functional as F

from one.constants import *
from one.core import *
from one.vision.acquisition import get_num_channels
from one.vision.acquisition import to_channel_first


# MARK: - BGR ------------------------------------------------------------------

def bgr_to_grayscale(image: Tensor) -> Tensor:
    """
    Convert BGR image to grayscale. Image data is assumed to be in the range of
    [0.0, 1.0]. First flips to RGB, then converts.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.

    Returns:
        Grayscale image of shape [..., 1, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_grayscale(image=bgr_to_rgb(image=image))


def bgr_to_hsv(image: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Convert BGR image to HSV. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, 
            where ... means it can have an arbitrary number of leading 
            dimensions.
        eps (float): Scalar to enforce numerical stability. Defaults to `1e-8`.

    Returns:
        HSV image of shape [..., 3, H, W]. H channel values are in the range
            [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_hsv(image=bgr_to_rgb(image=image))


def bgr_to_lab(image: Tensor) -> Tensor:
    """
    Convert BGR image to Lab. Image data is assumed to be in the range
    of [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        Lab image of shape [..., 3, H, W]. L channel values are  in the range
        [0, 100]. a and b are in the range [-127, 127].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_lab(image=bgr_to_rgb(image=image))


def bgr_to_luv(image: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Convert BGR image to Luv. Image data is assumed to be in the range of
    [0.0, 1.0]. Luv color is computed using the D65 illuminant and Observer 2.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        eps (float): For numerically stability when dividing. Defaults to 1e-12.

    Returns:
        Luv image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_luv(image=bgr_to_rgb(image=image), eps=eps)


def bgr_to_rgb(image: Tensor) -> Tensor:
    """
    Convert BGR image to RGB.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return image.flip(-3)


def bgr_to_rgba(image: Tensor, alpha_val: float | Tensor) -> Tensor:
    """
    Convert BGR image to RGBA.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        alpha_val (float | Tensor): A float number or tensor for the alpha value.

    Returns:
        RGBA image of shape [..., 3, H, W].

    Notes:
        Current functionality is NOT supported by Torchscript.
    """
    assert_tensor_of_channels(image, 3)
    if not isinstance(alpha_val, (float, Tensor)):
        raise TypeError(
            f"`alpha_val` must be a `float` or `Tensor`. "
            f"But got: {type(alpha_val)}."
        )
  
    # Convert first to RGB, then add alpha channel
    rgba = rgb_to_rgba(image=bgr_to_rgb(image=image), alpha_val=alpha_val)
    return rgba


def bgr_to_xyz(image: Tensor) -> Tensor:
    """
    Convert BGR image to XYZ.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        XYZ image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return bgr_to_xyz(image=bgr_to_rgb(image=image))


def bgr_to_ycrcb(image: Tensor) -> Tensor:
    """
    Convert RGB image to YCrCb.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        YCrCb image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_ycrcb(image=bgr_to_rgb(image=image))


def bgr_to_yuv(image: Tensor) -> Tensor:
    """
    Convert BGR image to YUV. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        YUV image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_yuv(image=bgr_to_rgb(image=image))


@TRANSFORMS.register(name="bgr_to_grayscale")
class BgrToGrayscale(Transform):
    """
    Convert BGR image to grayscale. Image data is assumed to be in the range of 
    [0.0, 1.0]. First flips to RGB, then converts.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return bgr_to_grayscale(image=input), \
               bgr_to_grayscale(image=target) if target is not None else None


@TRANSFORMS.register(name="bgr_to_hsv")
class BgrToHsv(Transform):
    """Convert BGR image to HSV. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        eps (float):
            Scalar to enforce numerical stability. Defaults to `1e-8`.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return bgr_to_hsv(image=input,  eps=self.eps), \
               bgr_to_hsv(image=target, eps=self.eps) \
                   if target is not None else None


@TRANSFORMS.register(name="bgr_to_lab")
class BgrToLab(Transform):
    """
    Convert BGR image to Lab. Image data is assumed to be in the range of
    [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return bgr_to_lab(image=input), \
               bgr_to_lab(image=target) if target is not None else None
  

@TRANSFORMS.register(name="bgr_to_luv")
class BgrToLuv(Transform):
    """
    Convert BGR image to Luv. Image data is assumed to be in the range of
    [0.0, 1.0]. Luv color is computed using the D65 illuminant and Observer 2.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return bgr_to_luv(image=input), \
               bgr_to_luv(image=target) if target is not None else None
    

@TRANSFORMS.register(name="bgr_to_rgb")
class BgrToRgb(Transform):
    """
    Convert BGR image to RGB.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return bgr_to_rgb(image=input), \
               bgr_to_rgb(image=target) if target is not None else None


@TRANSFORMS.register(name="bgr_to_rgba")
class BgrToRgba(Transform):
    """
    Convert BGR image to RGBA.

    Args:
        alpha_val (float | Tensor): A float number or tensor for the alpha value.
    """
    
    def __init__(self, alpha_val: float | Tensor):
        super().__init__()
        self.alpha_val = alpha_val
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_rgba(image=input,  alpha_val=self.alpha_val), \
               rgb_to_rgba(image=target, alpha_val=self.alpha_val) \
                   if target is not None else None
    

@TRANSFORMS.register(name="bgr_to_xyz")
class BgrToXyz(Transform):
    """
    Convert BGR image to XYZ.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return bgr_to_xyz(image=input), \
               bgr_to_xyz(image=target) if target is not None else None


@TRANSFORMS.register(name="bgr_to_ycrcb")
class BgrToYcrcb(Transform):
    """
    Convert RGB image to YCrCb.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return bgr_to_ycrcb(image=input), \
               bgr_to_ycrcb(image=target) if target is not None else None
    

@TRANSFORMS.register(name="bgr_to_yuv")
class BgrToYuv(Transform):
    """Convert BGR image to YUV. Image data is assumed to be in the range of
    [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return bgr_to_yuv(image=input), \
               bgr_to_yuv(image=target) if target is not None else None


# MARK: - Grayscale ------------------------------------------------------------

def grayscale_to_rgb(image: Tensor) -> Tensor:
    """
    Convert grayscale image to RGB version of image. Image data is assumed to
    be in the range of [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 1, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 1)
    rgb = torch.cat([image, image, image], dim=-3)

    # we should find a better way to raise this kind of warnings
    # if not torch.is_floating_point(image):
    #     warnings.warn(f"Input image is not of float dtype. Got: {image.dtype}")

    return rgb


@TRANSFORMS.register(name="grayscale_to_rgb")
class GrayscaleToRgb(Transform):
    """
    Convert grayscale image to RGB version of image. Image data is assumed to
    be in the range of [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return grayscale_to_rgb(image=input), \
               grayscale_to_rgb(image=target) if target is not None else None
   
   
# MARK: - HSL/HSV --------------------------------------------------------------

def hls_to_rgb(image: Tensor) -> Tensor:
    """
    Convert HLS image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    if not torch.jit.is_scripting():
        # weird way to use globals compiling with JIT even in the code not used by JIT...
        # __setattr__ can be removed if pytorch version is > 1.6.0 and then use:
        # hls_to_rgb.HLS2RGB = hls_to_rgb.HLS2RGB.to(image.device)
        hls_to_rgb.__setattr__("HLS2RGB", hls_to_rgb.HLS2RGB.to(image))  # type: ignore
        _HLS2RGB = hls_to_rgb.HLS2RGB  # type: ignore
    else:
        _HLS2RGB = torch.tensor([[[0.0]], [[8.0]], [[4.0]]], device=image.device, dtype=image.dtype)  # [3, 1, 1]

    im  = image.unsqueeze(-4)
    h   = torch.select(im, -3, 0)
    l   = torch.select(im, -3, 1)
    s   = torch.select(im, -3, 2)
    h  *= 6 / math.pi  # h * 360 / (2 * math.pi) / 30
    a   = s * torch.min(l, 1.0 - l)

    # kr = (0 + h) % 12
    # kg = (8 + h) % 12
    # kb = (4 + h) % 12
    k    = (h + _HLS2RGB) % 12

    # ll - a * max(min(min(k - 3.0, 9.0 - k), 1), -1)
    mink = torch.min(k - 3.0, 9.0 - k)
    return torch.addcmul(l, a, mink.clamp_(min=-1.0, max=1.0), value=-1)


def hsv_to_bgr(image: Tensor) -> Tensor:
    """
    Convert HSV image to BGR. FH channel values are assumed to be in the range
    [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        BGR image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_bgr(image=hsv_to_rgb(image=image))


def hsv_to_rgb(image: Tensor) -> Tensor:
    """
    Convert HSV image to RGB. H channel values are assumed to be in the range
    [0.0 2pi]. S and V are in the range [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    h   = image[..., 0, :, :] / (2 * math.pi)
    s   = image[..., 1, :, :]
    v   = image[..., 2, :, :]

    hi  = torch.floor(h * 6) % 6
    f   = ((h * 6) % 6) - hi
    one = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p   = v * (one - s)
    q   = v * (one - f * s)
    t   = v * (one - (one - f) * s)

    hi      = hi.long()
    indices = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    
    rgb = torch.stack((
        v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q
    ), dim=-3)
    rgb = torch.gather(rgb, -3, indices)
    return rgb


@TRANSFORMS.register(name="hls_to_rgb")
class HlsToRgb(Transform):
    """
    Convert HLS image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return hls_to_rgb(image=input), \
               hls_to_rgb(image=target) if target is not None else None
    

@TRANSFORMS.register(name="hsv_to_bgr")
class HsvToBgr(Transform):
    """
    Convert HSV image to BGR. FH channel values are assumed to be in the range
    [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return hsv_to_bgr(image=input), \
               hsv_to_bgr(image=target) if target is not None else None
    
    
@TRANSFORMS.register(name="hsv_to_rgb")
class HsvToRgb(Transform):
    """
    Convert HSV image to RGB. H channel values are assumed to be in the range
    [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return hsv_to_rgb(image=input), \
               hsv_to_rgb(image=target) if target is not None else None
    
    
def _integer_to_color(image: np.ndarray, colors: list) -> np.ndarray:
    """Convert integer-encoded image to color image. Fill an image with labels'
    colors.

    Args:
        image (np.ndarray):
            An image in either one-hot or integer.
        colors (list):
            List of all colors.

    Returns:
        color (np.ndarray):
            Colored image.
    """
    if len(colors) <= 0:
        raise ValueError(f"No colors are provided.")
    
    # Convert to channel-first
    image = to_channel_first(image)
    
    # Squeeze dims to 2
    if image.ndim == 3:
        image = np.squeeze(image)
    
    # Draw color
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, len(colors)):
        idx = image == l
        r[idx] = colors[l][0]
        g[idx] = colors[l][1]
        b[idx] = colors[l][2]
    rgb = np.stack([r, g, b], axis=0)
    return rgb


def integer_to_color(image: Tensor, colors: list) -> Tensor:
    mask_np = image.numpy()
    mask_np = integer_to_color(mask_np, colors)
    color   = torch.from_numpy(mask_np)
    return color


def is_color_image(image: Tensor) -> bool:
    """Check if the given image is color encoded."""
    if get_num_channels(image) in [3, 4]:
        return True
    return False


# MARK: - LAB ------------------------------------------------------------------

def lab_to_bgr(image: Tensor, clip: bool = True) -> Tensor:
    """
    Convert Lab image to BGR.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        clip (bool): Whether to apply clipping to insure output BGR values in
            range [0.0 1.0]. Defaults to True.

    Returns:
        BGR image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_bgr(image=lab_to_rgb(image=image, clip=clip))


def lab_to_rgb(image: Tensor, clip: bool = True) -> Tensor:
    """
    Convert Lab image to RGB.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        clip (bool): Whether to apply clipping to insure output RGB values in
            range [0.0 1.0]. Defaults to True.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    L  = image[..., 0, :, :]
    a  = image[..., 1, :, :]
    _b = image[..., 2, :, :]

    fy = (L + 16.0) / 116.0
    fx = (a / 500.0) + fy
    fz = fy - (_b / 200.0)

    # If color data out of range: Z < 0
    fz   = fz.clamp(min=0.0)
    fxyz = torch.stack([fx, fy, fz], dim=-3)

    # Convert from Lab to XYZ
    power = torch.pow(fxyz, 3.0)
    scale = (fxyz - 4.0 / 29.0) / 7.787
    xyz   = torch.where(fxyz > 0.2068966, power, scale)

    # For D65 white point
    xyz_ref_white = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz.device, dtype=xyz.dtype
    )[..., :, None, None]
    xyz_im  = xyz * xyz_ref_white
    rgbs_im = xyz_to_rgb(xyz_im)

    # https://github.com/richzhang/colorization-pytorch/blob/66a1cb2e5258f7c8f374f582acc8b1ef99c13c27/util/util.py#L107
    #     rgbs_im = torch.where(rgbs_im < 0, torch.zeros_like(rgbs_im), rgbs_im)

    # Convert from RGB Linear to sRGB
    rgb_im = linear_rgb_to_rgb(rgbs_im)

    # Clip to [0.0, 1.0] https://www.w3.org/Graphics/Color/srgb
    if clip:
        rgb_im = torch.clamp(rgb_im, min=0.0, max=1.0)

    return rgb_im


@TRANSFORMS.register(name="lab_to_bgr")
class LabToBgr(Transform):
    """
    Convert integer-encoded image to color image. Fill an image with labels'
    colors.
    
    Args:
        clip (bool): Whether to apply clipping to insure output BGR values in
            range [0.0 1.0]. Defaults to True.
    """
    
    def __init__(self, clip: bool = True):
        super().__init__()
        self.clip = clip
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return lab_to_bgr(image=input,  clip=self.clip), \
               lab_to_bgr(image=target, clip=self.clip) \
                   if target is not None else None


@TRANSFORMS.register(name="lab_to_rgb")
class LabToRgb(Transform):
    """
    Convert Lab image to BGR.
    
    Args:
        clip (bool): Whether to apply clipping to insure output BGR values in
            range [0.0 1.0]. Defaults to True.
    """
    
    def __init__(self, clip: bool = True):
        super().__init__()
        self.clip = clip
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return lab_to_rgb(image=input,  clip=self.clip), \
               lab_to_rgb(image=target, clip=self.clip) \
                   if target is not None else None
    

# MARK: - Linear RGB -----------------------------------------------------------

def linear_rgb_to_rgb(image: Tensor) -> Tensor:
    """
    Convert linear RGB image to sRGB.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        sRGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    threshold = 0.0031308
    rgb       = torch.where(
        image > threshold,
        1.055 * torch.pow(image.clamp(min=threshold), 1 / 2.4) - 0.055,
        12.92 * image
    )
    return rgb

   
@TRANSFORMS.register(name="linear_rgb_to_rgb")
class LinearRgbToRgb(Transform):
    """
    Convert linear RGB image to sRGB.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return linear_rgb_to_rgb(image=input), \
               linear_rgb_to_rgb(image=target) if target is not None else None
    
    
# MARK: - Luv ------------------------------------------------------------------

def luv_to_bgr(image: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Convert Luv image to BGR.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        eps (float): For numerically stability when dividing. Defaults to 1e-12.

    Returns:
        BGR image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_bgr(image=luv_to_rgb(image=image, eps=eps))


def luv_to_rgb(image: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Convert Luv image to RGB.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        eps (float): For numerically stability when dividing. Defaults to 1e-12.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    L = image[..., 0, :, :]
    u = image[..., 1, :, :]
    v = image[..., 2, :, :]

    # Convert from Luv to XYZ
    y = torch.where(L > 7.999625, torch.pow((L + 16) / 116, 3.0), L / 903.3)

    # Compute white point
    xyz_ref_white = (0.95047, 1.0, 1.08883)
    u_w = ((4 * xyz_ref_white[0]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))
    v_w = ((9 * xyz_ref_white[1]) /
           (xyz_ref_white[0] + 15 * xyz_ref_white[1] + 3 * xyz_ref_white[2]))

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


@TRANSFORMS.register(name="luv_to_bgr")
class LuvToBgr(Transform):
    """
    Convert Luv image to BGR.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return luv_to_bgr(image=input), \
               luv_to_bgr(image=target) if target is not None else None
    

@TRANSFORMS.register(name="luv_to_rgb")
class LuvToRgb(Transform):
    """
    Convert Luv image to RGB.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return luv_to_rgb(image=input), \
               luv_to_rgb(image=target) if target is not None else None
    

# MARK: - Raw ------------------------------------------------------------------

def raw_to_rgb(image: Tensor, cfa: CFA) -> Tensor:
    """
    Convert raw bayer image to RGB version of image. We are assuming a CFA with
    2 green, 1 red, 1 blue. A bilinear interpolation is used for R/G and a fix
    convolution for the green pixels. To simplify calculations we expect the
    height width to be evenly divisible by 2.0.

    Image data is assumed to be in the range of [0.0, 1.0]. Image H/W is
    assumed to be evenly divisible by 2.0 for simplicity reasons

    Args:
        image (Tensor): Image of shape [..., 1 , H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        cfa (CFA): Configuration of the color filter.
    
    Returns:
        RGB image of shape [..., 3, H, W].

    Example:
        >>> rawinput = torch.randn(2, 1, 4, 6)
        >>> rgb      = raw_to_rgb(rawinput, CFA.RG) # [2, 3, 4, 6]
    """
    assert_tensor_of_channels(image, 1)
    if (len(image.shape) < 2
        or image.shape[-2] % 2 == 1
        or image.shape[-1] % 2 == 1):
        raise ValueError(
            f"image H, W must be evenly divisible by 2. "
            f"But got: {image.shape}."
        )

    imagesize = image.size()
    image     = image.view(-1, 1, image.shape[-2], image.shape[-1])

    # BG is defined as pel 1,1 being blue, that is the top left is actually
    # green. This matches opencv naming so makes sense to keep
    if cfa == CFA.BG:
        r    = image[..., :, ::2, ::2]
        b    = image[..., :, 1::2, 1::2]
        rpad = (0, 1, 0, 1)
        bpad = (1, 0, 1, 0)
    elif cfa == CFA.GB:
        r    = image[..., :, ::2, 1::2]
        b    = image[..., :, 1::2, ::2]
        rpad = (1, 0, 0, 1)
        bpad = (0, 1, 1, 0)
    elif cfa == CFA.RG:
        r    = image[..., :, 1::2, 1::2]
        b    = image[..., :, ::2, ::2]
        rpad = (1, 0, 1, 0)
        bpad = (0, 1, 0, 1)
    elif cfa == CFA.GR:
        r    = image[..., :, 1::2, ::2]
        b    = image[..., :, ::2, 1::2]
        rpad = (0, 1, 1, 0)
        bpad = (1, 0, 0, 1)
    else:
        raise ValueError(f"`cfa` must be one of {CFA}. But got: {cfa}.")

    # upscaling r and b with bi-linear gives reasonable quality
    # Note that depending on where these are sampled we need to pad appropriately
    # the bilinear filter will pretty much be based on for example this layout (RG)
    # (which needs to be padded bottom right)
    # +-+-+
    # |B| |
    # | | |
    # +-+-+
    # While in this layout we need to pad with additional B samples top left to
    # make sure we interpolate from the correct position
    # +-+-+
    # | | |
    # | |B|
    # +-+-+
    # For an image like this (3x2 blue pixels)
    # +------+
    # |B B B |
    # |      |
    # |B B B |
    # |      |
    # +------+
    # It needs to be expanded to this (4x3 pixels scaled to 7x5 for correct interpolation)
    # +-------+
    # |B B B b|
    # |       |
    # |B B B b|
    # |       |
    # |b b b b|
    # +-------+
    # and we crop the area afterwards. This is since the interpolation will be between first and last pixel
    # evenly spaced between them while the B/R samples will be missing in the corners were they are assumed to exist
    # Further we need to do align_corners to start the interpolation from the middle of the samples in the corners, that
    # way we get to keep the known blue samples across the whole image
    rpadded = F.pad(r, list(rpad), "replicate")
    bpadded = F.pad(b, list(bpad), "replicate")
    # Use explicit padding instead of conv2d padding to be able to use reflect
    # which mirror the correct colors for a 2x2 bayer filter
    gpadded = F.pad(image, [1, 1, 1, 1], "reflect")

    ru = F.interpolate(
        rpadded, size=(image.shape[-2] + 1, image.shape[-1] + 1),
        mode="bilinear", align_corners=True
    )
    bu = F.interpolate(
        bpadded, size=(image.shape[-2] + 1, image.shape[-1] + 1),
        mode="bilinear", align_corners=True
    )

    # Remove the extra padding
    ru = F.pad(ru, [-x for x in rpad])
    bu = F.pad(bu, [-x for x in bpad])

    # All unknown pixels are the average of the nearby green samples
    kernel = torch.tensor(
        [[[[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]]]],
        dtype=image.dtype, device=image.device
    )

    # This is done on all samples but result for the known green samples is
    # then overwritten by the input
    gu = F.conv2d(gpadded, kernel)

    # Overwrite the already known samples which otherwise have values from r/b
    # this depends on the CFA configuration
    if cfa == CFA.BG:
        gu[:, :, ::2, 1::2]  = image[:, :, ::2, 1::2]
        gu[:, :, 1::2, ::2]  = image[:, :, 1::2, ::2]
    elif cfa == CFA.GB:
        gu[:, :, ::2, ::2]   = image[:, :, ::2, ::2]
        gu[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
    elif cfa == CFA.RG:
        gu[:, :, 1::2, ::2]  = image[:, :, 1::2, ::2]
        gu[:, :, ::2, 1::2]  = image[:, :, ::2, 1::2]
    elif cfa == CFA.GR:
        gu[:, :, 1::2, 1::2] = image[:, :, 1::2, 1::2]
        gu[:, :, ::2, ::2]   = image[:, :, ::2, ::2]
    else:
        raise ValueError(f"`cfa` must be one of {CFA}. But got: {cfa}.")

    ru = ru.view(imagesize)
    gu = gu.view(imagesize)
    bu = bu.view(imagesize)

    rgb = torch.cat([ru, gu, bu], dim=-3)
    return rgb


# MARK: - RGB ------------------------------------------------------------------

def rgb_to_bgr(image: Tensor) -> Tensor:
    """
    Convert RGB image to BGR.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        BGR image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return image.flip(-3)


def rgb_to_grayscale(
    image      : Tensor,
    rgb_weights: list[float] = (0.299, 0.587, 0.114)
) -> Tensor:
    """
    Convert RGB image to grayscale version of image. Image data is assumed to
    be in the range of [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        rgb_weights (list[float]): Weights that will be applied on each channel
            (RGB). Sum of the weights should add up to one.
            Defaults to (0.299, 0.587, 0.114).
    
    Returns:
        Grayscale image of shape [..., 1, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    rgb_weights = torch.FloatTensor(rgb_weights)
    assert_tensor(rgb_weights)
    if rgb_weights.shape[-1] != 3:
        raise ValueError(
            f"`rgb_weights` must have a shape of [..., 3]. "
            f"But got: {rgb_weights.shape}."
        )
    
    r = image[..., 0:1, :, :]
    g = image[..., 1:2, :, :]
    b = image[..., 2:3, :, :]

    if not torch.is_floating_point(image) and (image.dtype != rgb_weights.dtype):
        raise ValueError(
            f"`image` and `rgb_weights` must have the same dtype. "
            f"But got: {image.dtype} and {rgb_weights.dtype}."
        )

    w_r, w_g, w_b = rgb_weights.to(image).unbind()
    return w_r * r + w_g * g + w_b * b


def rgb_to_hls(image: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Convert RGB image to HLS. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        eps (float): Epsilon value to avoid div by zero. Defaults to 1e-8.

    Returns:
        HLS image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    if not torch.jit.is_scripting():
        # weird way to use globals compiling with JIT even in the code not used by JIT...
        # __setattr__ can be removed if pytorch version is > 1.6.0 and then use:
        # rgb_to_hls.RGB2HSL_IDX = hls_to_rgb.RGB2HSL_IDX.to(image.device)
        rgb_to_hls.__setattr__("RGB2HSL_IDX", rgb_to_hls.RGB2HSL_IDX.to(image))  # type: ignore
        _RGB2HSL_IDX = rgb_to_hls.RGB2HSL_IDX  # type: ignore
    else:
        _RGB2HSL_IDX = torch.tensor([[[0.0]], [[1.0]], [[2.0]]], device=image.device, dtype=image.dtype)  # [3, 1, 1]

    # maxc: Tensor  # not supported by JIT
    # imax: Tensor  # not supported by JIT
    maxc, imax = image.max(-3)
    minc       = image.min(-3)[0]

    # h: Tensor  # not supported by JIT
    # ll: Tensor  # not supported by JIT
    # s: Tensor  # not supported by JIT
    # image_hls: Tensor  # not supported by JIT
    if image.requires_grad:
        l_ = maxc + minc
        s  = maxc - minc
        # weird behaviour with undefined vars in JIT...
        # scripting requires image_hls be defined even if it is not used :S
        h  = l_  # assign to any image...
        image_hls = l_  # assign to any image...
    else:
        # define the resulting image to avoid the torch.stack([h, ll, s])
        # so, h, ll and s require inplace operations
        # stack() increases in a 10% the cost in colab
        image_hls = torch.empty_like(image)
        h         = torch.select(image_hls, -3, 0)
        l_        = torch.select(image_hls, -3, 1)
        s         = torch.select(image_hls, -3, 2)
        torch.add(maxc, minc, out=l_)  # ll = max + min
        torch.sub(maxc, minc, out=s)  # s = max - min

    # precompute image / (max - min)
    im = image / (s + eps).unsqueeze(-3)

    # epsilon cannot be inside the torch.where to avoid precision issues
    s  /= torch.where(l_ < 1.0, l_, 2.0 - l_) + eps  # saturation
    l_ /= 2  # luminance

    # note that r,g and b were previously div by (max - min)
    r = torch.select(im, -3, 0)
    g = torch.select(im, -3, 1)
    b = torch.select(im, -3, 2)
    # h[imax == 0] = (((g - b) / (max - min)) % 6)[imax == 0]
    # h[imax == 1] = (((b - r) / (max - min)) + 2)[imax == 1]
    # h[imax == 2] = (((r - g) / (max - min)) + 4)[imax == 2]
    cond = imax.unsqueeze(-3) == _RGB2HSL_IDX
    if image.requires_grad:
        h = torch.mul((g - b) % 6, torch.select(cond, -3, 0))
    else:
        torch.mul((g - b).remainder(6), torch.select(cond, -3, 0), out=h)
    h += torch.add(b - r, 2) * torch.select(cond, -3, 1)
    h += torch.add(r - g, 4) * torch.select(cond, -3, 2)
    # h = 2.0 * math.pi * (60.0 * h) / 360.0
    h *= math.pi / 3.0  # hue [0, 2*pi]

    if image.requires_grad:
        return torch.stack([h, l_, s], dim=-3)
    return image_hls


def rgb_to_hsv(image: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Convert image from RGB to HSV. Image data is assumed to be in the range of
    [0.0, 1.0].
    
    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        eps (float): Scalar to enforce numerical stability. Defaults to 1e-8.

    Returns:
        HSV image of shape [..., 3, H, W]. H channel values are in the range
            [0.0 2pi]. S and V are in the range [0.0, 1.0].
    """
    assert_tensor_of_channels(image, 3)
    
    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac              = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac     = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = (bc - gc)
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h   = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h   = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h   = (h / 6.0) % 1.0
    h  *= 2.0 * math.pi  # We return 0/2pi output
    hsv = torch.stack((h, s, v), dim=-3)
    
    return hsv


def rgb_to_lab(image: Tensor) -> Tensor:
    """
    Convert RGB image to Lab. Image data is assumed to be in the range of
    [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        Lab image of shape [..., 3, H, W]. L channel values are in the range
            [0, 100]. a and b are in the range [-127, 127].
    """
    assert_tensor_of_channels(image, 3)
    
    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)
    xyz_im  = rgb_to_xyz(lin_rgb)

    # normalize for D65 white point
    xyz_ref_white  = torch.tensor(
        [0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype
    )[..., :, None, None]
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

    lab = torch.stack([L, a, _b], dim=-3)
    return lab


def rgb_to_linear_rgb(image: Tensor) -> Tensor:
    """
    Convert sRGB image to linear RGB. Used in colorspace conversions.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        Linear RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    lin_rgb = torch.where(
        image > 0.04045,
        torch.pow(((image + 0.055) / 1.055), 2.4),
        image / 12.92
    )
    return lin_rgb


def rgb_to_luv(image: Tensor, eps: float = 1e-12) -> Tensor:
    """
    Convert RGB image to Luv. Image data is assumed to be in the range of
    [0.0, 1.0]. Luv color is computed using the D65 illuminant and Observer 2.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        eps (float): For numerically stability when dividing. Defaults to 1e-12.

    Returns:
        Luv image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    # Convert from sRGB to Linear RGB
    lin_rgb = rgb_to_linear_rgb(image)
    
    xyz_im  = rgb_to_xyz(lin_rgb)
    x       = xyz_im[..., 0, :, :]
    y       = xyz_im[..., 1, :, :]
    z       = xyz_im[..., 2, :, :]

    threshold = 0.008856
    L = torch.where(y > threshold,
                    116.0 * torch.pow(y.clamp(min=threshold), 1.0 / 3.0) - 16.0,
                    903.3 * y)

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


def rgb_to_raw(image: Tensor, cfa: CFA) -> Tensor:
    """
    Convert RGB image to RAW version of image with the specified color
    filter array. Image data is assumed to be in the range of [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        cfa (CFA): Which color filter array do we want the output to mimic.
            I.e. which pixels are red/green/blue.

    Returns:
        Raw image of shape [..., 3, H, W].

    Example:
        >>> rgbinput = torch.rand(2, 3, 4, 6)
        >>> raw      = rgb_to_raw(rgbinput, CFA.BG)  # [2, 1, 4, 6]
    """
    assert_tensor_of_channels(image, 3)
    
    # Pick the image with green pixels clone to make sure grad works
    output = image[..., 1:2, :, :].clone()

    # Overwrite the r/b positions (depending on the cfa configuration) with
    # blue/red pixels
    if cfa == CFA.BG:
        output[..., :, ::2, ::2]   = image[..., 0:1, ::2, ::2]    # red
        output[..., :, 1::2, 1::2] = image[..., 2:3, 1::2, 1::2]  # blue
    elif cfa == CFA.GB:
        output[..., :, ::2, 1::2]  = image[..., 0:1, ::2, 1::2]  # red
        output[..., :, 1::2, ::2]  = image[..., 2:3, 1::2, ::2]  # blue
    elif cfa == CFA.RG:
        output[..., :, 1::2, 1::2] = image[..., 0:1, 1::2, 1::2]  # red
        output[..., :, ::2, ::2]   = image[..., 2:3, ::2, ::2]    # blue
    elif cfa == CFA.GR:
        output[..., :, 1::2, ::2]  = image[..., 0:1, 1::2, ::2]  # red
        output[..., :, ::2, 1::2]  = image[..., 2:3, ::2, 1::2]  # blue

    return output


def rgb_to_rgba(image: Tensor, alpha_val: float | Tensor) -> Tensor:
    """
    Convert image from RGB to RGBA.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        alpha_val (float | Tensor): A float number or tensor for the alpha value.

    Returns:
        RGBA image of shape of shape [..., 4, H, W]

    Notes:
        Current functionality is NOT supported by Torchscript.
    """
    assert_tensor_of_channels(image, 3)
    
    if not isinstance(alpha_val, (float, Tensor)):
        raise TypeError(
            f"`alpha_val` must be `float` or `Tensor`. "
            f"But got: {type(alpha_val)}."
        )
  
    # Add one channel
    r, g, b = torch.chunk(image, image.shape[-3], dim=-3)
    a       = cast(Tensor, alpha_val)

    if isinstance(alpha_val, float):
        a = torch.full_like(r, fill_value=float(alpha_val))
    rgba = torch.cat([r, g, b, a], dim=-3)

    return rgba


def rgb_to_xyz(image: Tensor) -> Tensor:
    """
    Convert RGB image to XYZ.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        XYZ image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    r   = image[..., 0, :, :]
    g   = image[..., 1, :, :]
    b   = image[..., 2, :, :]

    x   = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y   = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z   = 0.019334 * r + 0.119193 * g + 0.950227 * b
    xyz = torch.stack([x, y, z], -3)
    
    return xyz


def rgb_to_ycrcb(image: Tensor) -> Tensor:
    """
    Convert RGB image to YCrCb.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        YCrCb image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    r = image[..., 0, :, :]
    g = image[..., 1, :, :]
    b = image[..., 2, :, :]

    delta = 0.5
    y     = 0.299 * r + 0.587 * g + 0.114 * b
    cb    = (b - y) * 0.564 + delta
    cr    = (r - y) * 0.713 + delta
    ycrcb = torch.stack([y, cr, cb], -3)
    
    return ycrcb


def rgb_to_yuv(image: Tensor) -> Tensor:
    """
    Convert RGB image to YUV. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        YUV image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    r   = image[..., 0, :, :]
    g   = image[..., 1, :, :]
    b   = image[..., 2, :, :]

    y   =  0.299 * r + 0.587 * g + 0.114 * b
    u   = -0.147 * r - 0.289 * g + 0.436 * b
    v   =  0.615 * r - 0.515 * g - 0.100 * b
    yuv = torch.stack([y, u, v], -3)
    
    return yuv


def rgb_to_yuv420(image: Tensor) -> Tensors:
    """
    Convert RGB image to YUV 420 (sub-sampled). Image data is assumed to be in
    the range of [0.0, 1.0]. Input need to be padded to be evenly divisible by
    2 horizontal and vertical. This function will output chroma siting [0.5, 0.5].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        A Tensor containing the Y plane with shape [..., 1, H, W]
        A Tensor containing the UV planes with shape [..., 2, H/2, W/2]
    """
    assert_tensor_of_channels(image, 3)
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"`image` must have a shape of [..., 3, H, W]. "
            f"But got: {image.shape}."
        )
    if (len(image.shape) < 2 or
        image.shape[-2] % 2 == 1 or
        image.shape[-1] % 2 == 1):
        raise ValueError(
            f"`image` H, W must be evenly divisible by 2. "
            f"But got: {image.shape}."
        )

    yuvimage = rgb_to_yuv(image)
    return (
        yuvimage[..., :1, :, :],
        F.avg_pool2d(yuvimage[..., 1:3, :, :], (2, 2))
    )


def rgb_to_yuv422(image: Tensor) -> Tensors:
    """
    Convert RGB image to YUV 422 (sub-sampled). Image data is assumed to be in
    the range of [0.0, 1.0]. Input need to be padded to be evenly divisible by
    2 vertical. This function will output chroma siting (0.5).

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
       A Tensor containing the Y plane with shape [..., 1, H, W].
       A Tensor containing the UV planes with shape [..., 2, H, W/2].
    """
    assert_tensor_of_channels(image, 3)
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(
            f"`image` must have a shape of [..., 3, H, W]. "
            f"But got: {image.shape}."
        )
    if (len(image.shape) < 2 or
        image.shape[-2] % 2 == 1 or
        image.shape[-1] % 2 == 1):
        raise ValueError(
            f"`image` H, W must be evenly divisible by 2. "
            f"But got: {image.shape}."
        )

    yuvimage = rgb_to_yuv(image)
    return (
        yuvimage[..., :1, :, :],
        F.avg_pool2d(yuvimage[..., 1:3, :, :], (1, 2))
    )


def rgba_to_bgr(image: Tensor) -> Tensor:
    """
    Convert RGBA image to BGR.

    Args:
        image (Tensor): Image of shape [..., 4, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 4)
    # Convert to RGB first, then to BGR
    return rgb_to_bgr(image=rgba_to_rgb(image=image))
    

def rgba_to_rgb(image: Tensor) -> Tensor:
    """
    Convert RGBA image to RGB.

    Args:
        image (Tensor): Image of shape [..., 4, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 4)
    
    # Unpack channels
    r, g, b, a = torch.chunk(image, image.shape[-3], dim=-3)

    # Compute new channels
    a_one = torch.tensor(1.0) - a
    r_new = a_one * r + a * r
    g_new = a_one * g + a * g
    b_new = a_one * b + a * b
    rgb   = torch.cat([r_new, g_new, b_new], dim=-3)

    return rgb


@TRANSFORMS.register(name="raw_to_rgb")
class RawToRgb(Transform):
    """
    Convert raw bayer image to RGB version of image. We are assuming a CFA
    with 2 green, 1 red, 1 blue. A bilinear interpolation is used for R/G and a
    fix convolution for the green pixels. To simplify calculations we expect
    the Height Width to be evenly divisible by 2.0.
    
    Image data is assumed to be in the range of [0.0, 1.0]. Image H/W is
    assumed to be evenly divisible by 2.0 for simplicity reasons.
    """
    
    def __init__(self, cfa: CFA):
        super().__init__()
        self.cfa = cfa
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return raw_to_rgb(image=input,  cfa=self.cfa), \
               raw_to_rgb(image=target, cfa=self.cfa) \
                   if target is not None else None


@TRANSFORMS.register(name="rgb_to_bgr")
class RgbToBgr(Transform):
    """
    Convert RGB image to BGR. Image data is assumed to be in the range of
    [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_bgr(image=input), \
               rgb_to_bgr(image=target) if target is not None else None
    

@TRANSFORMS.register(name="rgb_to_grayscale")
class RgbToGrayscale(Transform):
    """
    Convert RGB image to grayscale version of image. Image data is assumed to
    be in the range of [0.0, 1.0].
    
    Args:
        rgb_weights (list[float]): Weights that will be applied on each channel
            (RGB). Sum of the weights should add up to one.
            Defaults to (0.299, 0.587, 0.114).
    """
    
    def __init__(self, rgb_weights: list[float] = (0.299, 0.587, 0.114)):
        super().__init__()
        self.rgb_weights = rgb_weights
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_grayscale(image=input,  rgb_weights=self.rgb_weights), \
               rgb_to_grayscale(image=target, rgb_weights=self.rgb_weights) \
                   if target is not None else None


@TRANSFORMS.register(name="rgb_to_hls")
class RgbToHls(Transform):
    """
    Convert RGB image to HLS. Image data is assumed to be in the range of
    [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_hls(image=input), \
               rgb_to_hls(image=target) if target is not None else None
    

@TRANSFORMS.register(name="rgb_to_hsv")
class RgbToHsv(Transform):
    """
    Convert RGB image to HSV. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        eps (float):
            Scalar to enforce numerical stability. Defaults to 1e-8.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_hsv(image=input,  eps=self.eps), \
               rgb_to_hsv(image=target, eps=self.eps) \
                   if target is not None else None
    

@TRANSFORMS.register(name="rgb_to_lab")
class RgbToLab(Transform):
    """
    Convert RGB image to Lab. Image data is assumed to be in the range of
    [0.0 1.0]. Lab color is computed using the D65 illuminant and Observer 2.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_lab(image=input), \
               rgb_to_lab(image=target) if target is not None else None
    

@TRANSFORMS.register(name="rgb_to_linear_rgb")
class RgbToLinearRgb(Transform):
    """
    Convert sRGB image to linear RGB.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_linear_rgb(image=input), \
               rgb_to_linear_rgb(image=target) if target is not None else None
    
    
@TRANSFORMS.register(name="rgb_to_luv")
class RgbToLuv(Transform):
    """
    Convert RGB image to Luv. Image data is assumed to be in the range of
    [0.0, 1.0]. Luv color is computed using the D65 illuminant and Observer 2.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_luv(image=input), \
               rgb_to_luv(image=target) if target is not None else None
    

@TRANSFORMS.register(name="rgb_to_raw")
class RgbToRaw(Transform):
    """
    Convert RGB image to RAW version of image with the specified color filter
    array. Image data is assumed to be in the range of [0.0, 1.0].
    """
    
    def __init__(self, cfa: CFA):
        super().__init__()
        self.cfa = cfa
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_raw(image=input,  cfa=self.cfa), \
               rgb_to_raw(image=target, cfa=self.cfa) \
                   if target is not None else None
    

@TRANSFORMS.register(name="rgb_to_rgba")
class RgbToRgba(Transform):
    """
    Convert RGB image to RGBA.

    Args:
        alpha_val (float | Tensor): A float number or tensor for the alpha value.
    """
    
    def __init__(self, alpha_val: float | Tensor):
        super().__init__()
        self.alpha_val = alpha_val
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_rgba(image=input,  alpha_val=self.alpha_val), \
               rgb_to_rgba(image=target, alpha_val=self.alpha_val) \
                   if target is not None else None


@TRANSFORMS.register(name="rgb_to_xyz")
class RgbToXyz(Transform):
    """
    Convert RGB image to XYZ.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_xyz(image=input), \
               rgb_to_xyz(image=target) if target is not None else None
    

@TRANSFORMS.register(name="rgb_to_ycrcb")
class RgbToYcrcb(Transform):
    """
    Convert RGB image to YCrCb.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_ycrcb(image=input), \
               rgb_to_ycrcb(image=target) if target is not None else None
    
   
@TRANSFORMS.register(name="rgb_to_yuv")
class RgbToYuv(Transform):
    """
    Convert RGB image to YUV. Image data is assumed to be in the range of
    [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_yuv(image=input), \
               rgb_to_yuv(image=target) if target is not None else None
        
        
@TRANSFORMS.register(name="rgb_to_yuv420")
class RgbToYuv420(Transform):
    """
    Convert RGB image to YUV 420 (sub-sampled). Image data is assumed to be in
    the range of [0.0, 1.0]. Input need to be padded to be evenly divisible by
    2 horizontal and vertical. This function will output chroma siting [0.5, 0.5].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_yuv420(image=input), \
               rgb_to_yuv420(image=target) if target is not None else None
     

@TRANSFORMS.register(name="rgb_to_yuv422")
class RgbToYuv422(Transform):
    """
    Convert RGB image to YUV 422 (sub-sampled). Image data is assumed to be in
    the range of [0.0, 1.0]. Input need to be padded to be evenly divisible by
    2 vertical. This function will output chroma siting (0.5).
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgb_to_yuv422(image=input), \
               rgb_to_yuv422(image=target) if target is not None else None
    

@TRANSFORMS.register(name="rgba_to_bgr")
class RgbaToBgr(Transform):
    """
    Convert RGBA image to BGR.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgba_to_bgr(image=input), \
               rgba_to_bgr(image=target) if target is not None else None


@TRANSFORMS.register(name="rgba_to_rgb")
class RgbaToRgb(Transform):
    """
    Convert RGBA image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return rgba_to_rgb(image=input), \
               rgba_to_rgb(image=target) if target is not None else None
    

# MARK: - XYZ ------------------------------------------------------------------

def xyz_to_bgr(image: Tensor) -> Tensor:
    """
    Convert XYZ image to BGR.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        BGR image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_bgr(image=xyz_to_rgb(image=image))


def xyz_to_rgb(image: Tensor) -> Tensor:
    """
    Convert XYZ image to RGB.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    x = image[..., 0, :, :]
    y = image[..., 1, :, :]
    z = image[..., 2, :, :]

    r = ( 3.2404813432005266 * x + -1.5371515162713185 * y + -0.4985363261688878 * z)
    g = (-0.9692549499965682 * x +  1.8759900014898907 * y +  0.0415559265582928 * z)
    b = ( 0.0556466391351772 * x + -0.2040413383665112 * y +  1.0573110696453443 * z)
    rgb = torch.stack([r, g, b], dim=-3)
    return rgb


@TRANSFORMS.register(name="xyz_to_bgr")
class XyzToBgr(Transform):
    """
    Convert XYZ image to BGR.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return xyz_to_bgr(image=input), \
               xyz_to_bgr(image=target) if target is not None else None


@TRANSFORMS.register(name="xyz_to_rgb")
class XyzToRgb(Transform):
    """
    Convert XYZ image to RGB.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return xyz_to_rgb(image=input), \
               xyz_to_rgb(image=target) if target is not None else None


# MARK: - YCrCb ----------------------------------------------------------------

def ycrcb_to_bgr(image: Tensor) -> Tensor:
    """
    Convert YCrCb image to BGR. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        BGR image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_bgr(image=ycrcb_to_rgb(image=image))


def ycrcb_to_rgb(image: Tensor) -> Tensor:
    """
    Convert YCrCb image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0].

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    y  = image[..., 0, :, :]
    cb = image[..., 1, :, :]
    cr = image[..., 2, :, :]

    delta      = 0.5
    cb_shifted = cb - delta
    cr_shifted = cr - delta

    r   = y + 1.403 * cr_shifted
    g   = y - 0.714 * cr_shifted - 0.344 * cb_shifted
    b   = y + 1.773 * cb_shifted
    rgb = torch.stack([r, g, b], -3)

    return rgb


@TRANSFORMS.register(name="ycrcb_to_bgr")
class YcrcbToBgr(Transform):
    """
    Convert YCbCr image to BGR. Image data is assumed to be in the range of
    [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return ycrcb_to_bgr(image=input), \
               ycrcb_to_bgr(image=target) if target is not None else None
    

@TRANSFORMS.register(name="ycrcb_to_rgb")
class YcrcbToRgb(Transform):
    """
    Convert YCbCr to RGB. Image data is assumed to be in the range of
    [0.0, 1.0].
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return ycrcb_to_rgb(image=input), \
               ycrcb_to_rgb(image=target) if target is not None else None


# MARK: - YUV ------------------------------------------------------------------

def yuv420_to_rgb(image_y: Tensor, image_uv: Tensor) -> Tensor:
    """
    Convert YUV420 image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Input need to be padded to
    be evenly divisible by 2 horizontal and vertical. This function assumed
    chroma siting is [0.5, 0.5]

    Args:
        image_y (Tensor): Y (luma) image plane of shape [..., 1, H, W] to be
            converted to RGB, where ... means it can have an arbitrary number
            of leading dimensions.
        image_uv (Tensor): UV (chroma) image planes of shape [..., 2, H/2, W/2]
            to be converted to RGB, where ... means it can have an arbitrary
            number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image_y,  1)
    assert_tensor_of_channels(image_uv, 2)
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


def yuv422_to_rgb(image_y: Tensor, image_uv: Tensor) -> Tensor:
    """
    Convert YUV422 image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma. Input need to be padded to
    be evenly divisible by 2 vertical. This function assumed chroma siting is
    (0.5).

    Args:
        image_y (Tensor): Y (luma) image plane of shape [..., 1, H, W] to be
            converted to RGB, where ... means it can have an arbitrary number
            of leading dimensions.
        image_uv (Tensor): UV (luma) image planes of shape [..., 2, H/2, W/2]
            to be converted to RGB, where ... means it can have an arbitrary
            number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image_y, 1)
    assert_tensor_of_channels(image_uv, 2)
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
    yuv444image = torch.cat([
        image_y, image_uv.repeat_interleave(2, dim=-1)
    ], dim=-3)
    # Then convert the yuv444 image
    return yuv_to_rgb(yuv444image)


def yuv_to_bgr(image: Tensor) -> Tensor:
    """
    Convert YUV image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        BGR image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    return rgb_to_bgr(image=yuv_to_rgb(image=image))


def yuv_to_rgb(image: Tensor) -> Tensor:
    """
    Convert YUV image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.

    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.

    Returns:
        RGB image of shape [..., 3, H, W].
    """
    assert_tensor_of_channels(image, 3)
    
    y   = image[..., 0, :, :]
    u   = image[..., 1, :, :]
    v   = image[..., 2, :, :]

    r   = y + 1.14 * v  # coefficient for g is 0
    g   = y + -0.396 * u - 0.581 * v
    b   = y + 2.029 * u  # coefficient for b is 0
    rgb = torch.stack([r, g, b], -3)

    return rgb


@TRANSFORMS.register(name="yuv_to_bgr")
class YuvToBgr(Transform):
    """
    Convert YUV image to Bgr. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return yuv_to_bgr(image=input), \
               yuv_to_bgr(image=target) if target is not None else None
    
    
@TRANSFORMS.register(name="yuv_to_rgb")
class YuvToRgb(Transform):
    """
    Convert YUV image to RGB. Image data is assumed to be in the range of
    [0.0, 1.0] for luma and [-0.5, 0.5] for chroma.
    """
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return yuv_to_rgb(image=input), \
               yuv_to_rgb(image=target) if target is not None else None


# Tricks to speed up a little the conversions by presetting small tensors
# (in the functions they are moved to the proper device)
hls_to_rgb.__setattr__("HLS2RGB",     torch.tensor([[[0.0]], [[8.0]], [[4.0]]]))  # [3, 1, 1]
rgb_to_hls.__setattr__("RGB2HSL_IDX", torch.tensor([[[0.0]], [[1.0]], [[2.0]]]))  # [3, 1, 1]
