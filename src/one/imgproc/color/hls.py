#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HLS color space.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torch import nn
from torch import Tensor

from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.utils import batch_image_processing
from one.imgproc.utils import channel_last_processing

__all__ = [
    "hls_to_rgb",
    "rgb_to_hls",
    "HlsToRgb",
    "RgbToHls"
]


# MARK: - Functional

@dispatch(Tensor)
def hls_to_rgb(image: Tensor) -> Tensor:
    """Convert a HLS image to RGB. Image data is assumed to be in the range
    of [0.0, 1.0].

    Args:
        image (Tensor[B, 3, H, W]):
            HLS image to be converted to RGB.

    Returns:
        rgb (Tensor[B, 3, H, W]):
            RGB version of the image.
    """
    if not torch.jit.is_scripting():
        # weird way to use globals compiling with JIT even in the code not used by JIT...
        # __setattr__ can be removed if pytorch version is > 1.6.0 and then use:
        # hls_to_rgb.HLS2RGB = hls_to_rgb.HLS2RGB.to(image.device)
        hls_to_rgb.__setattr__("HLS2RGB", hls_to_rgb.HLS2RGB.to(image))  # type: ignore
        _HLS2RGB = hls_to_rgb.HLS2RGB  # type: ignore
    else:
        _HLS2RGB = torch.tensor([[[0.0]], [[8.0]], [[4.0]]], device=image.device, dtype=image.dtype)  # [3, 1, 1]

    im = image.unsqueeze(-4)
    h  = torch.select(im, -3, 0)
    l  = torch.select(im, -3, 1)
    s  = torch.select(im, -3, 2)
    h *= 6 / math.pi  # h * 360 / (2 * math.pi) / 30
    a  = s * torch.min(l, 1.0 - l)

    # kr = (0 + h) % 12
    # kg = (8 + h) % 12
    # kb = (4 + h) % 12
    k = (h + _HLS2RGB) % 12

    # ll - a * max(min(min(k - 3.0, 9.0 - k), 1), -1)
    mink = torch.min(k - 3.0, 9.0 - k)
    return torch.addcmul(l, a, mink.clamp_(min=-1.0, max=1.0), value=-1)


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray)
def hls_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a HLS image to RGB. Image data is assumed to be in the range
    of [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            HLS image to be converted to RGB.

    Returns:
        rgb (np.ndarray[B, 3, H, W]):
            RGB version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_HLS2RGB)


@dispatch(Tensor, float)
def rgb_to_hls(image: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert a RGB image to HLS. Image data is assumed to be in the range
    of [0.0, 1.0].

    NOTE: this method cannot be compiled with JIT in pytohrch < 1.7.0

    Args:
        image (Tensor[B, 3, H, W]):
            RGB image to be converted to HLS.
        eps (float):
            Epsilon value to avoid div by zero.

    Returns:
        hls (Tensor[B, 3, H, W]):
            HLS version of the image.
    """
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
        # NOTE: stack() increases in a 10% the cost in colab
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


@batch_image_processing
@channel_last_processing
@dispatch(np.ndarray, float)
def rgb_to_hls(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert a RGB image to HLS. Image data is assumed to be in the range
    of [0.0, 1.0].

    Args:
        image (np.ndarray[B, 3, H, W]):
            RGB image to be converted to HLS.
        eps (float):
            Epsilon value to avoid div by zero.

    Returns:
        hls (np.ndarray[B, 3, H, W]):
            HLS version of the image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)


# MARK: - Modules

@TRANSFORMS.register(name="hls_to_rgb")
class HlsToRgb(nn.Module):
    """Convert an image from HLS to RGB. Image data is assumed to be in the
    range of [0.0, 1.0].

    Reference:
        https://en.wikipedia.org/wiki/HSL_and_HSV
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return hls_to_rgb(image)


@TRANSFORMS.register(name="rgb_to_hls")
class RgbToHls(nn.Module):
    """Convert an image from RGB to HLS. Image data is assumed to be in the
    range of [0.0, 1.0].
    """

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        return rgb_to_hls(image)


# Tricks to speed up a little the conversions by presetting small tensors
# (in the functions they are moved to the proper device)
hls_to_rgb.__setattr__("HLS2RGB",     torch.tensor([[[0.0]], [[8.0]], [[4.0]]]))  # [3, 1, 1]
rgb_to_hls.__setattr__("RGB2HSL_IDX", torch.tensor([[[0.0]], [[1.0]], [[2.0]]]))  # [3, 1, 1]
