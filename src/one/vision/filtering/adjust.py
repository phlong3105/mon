#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from multipledispatch import dispatch
from torch import nn
from torch import Tensor
from torchvision.transforms.functional import adjust_brightness
from torchvision.transforms.functional import adjust_contrast
from torchvision.transforms.functional import adjust_gamma
from torchvision.transforms.functional import adjust_hue
from torchvision.transforms.functional import adjust_saturation
from torchvision.transforms.functional import adjust_sharpness
from torchvision.transforms.functional import autocontrast
from torchvision.transforms.functional import equalize
from torchvision.transforms.functional import invert
from torchvision.transforms.functional import posterize
from torchvision.transforms.functional import solarize
from torchvision.transforms.functional_tensor import _assert_channels
from torchvision.transforms.functional_tensor import _assert_image_tensor
from torchvision.transforms.functional_tensor import _hsv2rgb
from torchvision.transforms.functional_tensor import _rgb2hsv

from one.core import get_num_channels
from one.core import TensorOrArray
from one.core import TRANSFORMS

__all__ = [
    "adjust_brightness",
    "adjust_contrast",
    #"adjust_gamma",
    "adjust_hsv",
    "adjust_hue",
    "adjust_saturation",
    "adjust_sharpness",
    "autocontrast",
    "equalize",
    "invert",
    "posterize",
    "solarize",
    "AdjustBrightness",
    "AdjustContrast",
    "AdjustGamma",
    "AdjustHsv",
    "AdjustHue",
    "AdjustSaturation",
    "AdjustSharpness",
    "AutoContrast",
    "Equalize",
    "Invert",
    "Posterize",
    "Solarize",
]


# MARK: - Functional

@dispatch(Tensor, h_factor=float, s_factor=float, v_factor=float)
def adjust_hsv(
    image   : Tensor,
    h_factor: float = 0.5,
    s_factor: float = 0.5,
    v_factor: float = 0.5,
) -> Tensor:
    if not isinstance(image, Tensor):
        raise TypeError(f"`image` must be a `Tensor`. But got: {type(image)}")

    _assert_image_tensor(image)

    _assert_channels(image, [1, 3])
    if get_num_channels(image) == 1:  # Match PIL behaviour
        return image

    orig_dtype = image.dtype
    if image.dtype == torch.uint8:
        image = image.to(dtype=torch.float32) / 255.0

    image       = _rgb2hsv(image)
    h, s, v     = image.unbind(dim=-3)
    h           = (h * h_factor).clamp(0, 1)
    s           = (s * s_factor).clamp(0, 1)
    v           = (v * v_factor).clamp(0, 1)
    image       = torch.stack((h, s, v), dim=-3)
    img_hue_adj = _hsv2rgb(image)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj


@dispatch(np.ndarray, h_factor=float, s_factor=float, v_factor=float)
def adjust_hsv(
    image   : np.ndarray,
    h_factor: float = 0.5,
    s_factor: float = 0.5,
    v_factor: float = 0.5,
) -> np.ndarray:
    if not (isinstance(image, np.ndarray)):
        raise TypeError(f"`image` must be a `np.ndarray`. But got: {type(image)}.")
    
    # Random gains
    r  = np.random.uniform(-1, 1, 3) * [h_factor, s_factor, v_factor] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype         = image.dtype  # uint8

    x       = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)  # No return needed
    return image


# MARK: - Modules

@TRANSFORMS.register(name="adjust_brightness")
class AdjustBrightness(nn.Module):
    """Adjust brightness of an image.

    Args:
        brightness_factor (float):
            How much to adjust the brightness. Can be any non-negative number.
            0 gives a black image, 1 gives the original image while 2 increases
            the brightness by a factor of 2.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustBrightness(1.)(x)
        image([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustBrightness(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    # MARK: Magic Functions
    
    def __init__(self, brightness_factor: float):
        super().__init__()
        self.brightness_factor = brightness_factor

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (Tensor[..., 1 or 3, H, W]):
                Image to be adjusted. If img is Tensor, it is expected to
                be in [..., 1 or 3, H, W] format, where ... means it can have
                an arbitrary number of leading dimensions.

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Brightness adjusted image.
        """
        return adjust_brightness(image, self.brightness_factor)


@TRANSFORMS.register(name="adjust_contrast")
class AdjustContrast(nn.Module):
    """Adjust contrast of an image.

    Args:
        contrast_factor (float):
            How much to adjust the contrast. Can be any non-negative number.
            0 gives a solid gray image, 1 gives the original image while 2
            increases the contrast by a factor of 2.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustContrast(0.5)(x)
        image([[[[0.5000, 0.5000, 0.5000],
                  [0.5000, 0.5000, 0.5000],
                  [0.5000, 0.5000, 0.5000]]]])

        >>> x = torch.ones(2, 5, 3, 3)
        >>> y = torch.ones(2)
        >>> AdjustContrast(y)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    # MARK: Magic Functions
    
    def __init__(self, contrast_factor: float):
        super().__init__()
        self.contrast_factor = contrast_factor

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (Tensor[..., 1 or 3, H, W]):
                Image to be adjusted. If img is Tensor, it is expected to
                be in [..., 1 or 3, H, W] format, where ... means it can have
                an arbitrary number of leading dimensions.

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Contrast adjusted image.
        """
        return adjust_contrast(image, self.contrast_factor)


@TRANSFORMS.register(name="adjust_gamma")
class AdjustGamma(nn.Module):
    """Perform gamma correction on an image.
    
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:
    
    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction
    
    Args:
        gamma (float, Tensor):
            Non-negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker, while gamma smaller
            than 1 make dark regions lighter.
        gain (float, Tensor):
            Constant multiplier.

    Example:
        >>> x = torch.ones(1, 1, 3, 3)
        >>> AdjustGamma(1.0, 2.0)(x)
        image([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x  = torch.ones(2, 5, 3, 3)
        >>> y1 = torch.ones(2) * 1.0
        >>> y2 = torch.ones(2) * 2.0
        >>> AdjustGamma(y1, y2)(x).shape
        torch.Size([2, 5, 3, 3])
    """

    # MARK: Magic Functions
    
    def __init__(self, gamma: float, gain: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.gain  = gain

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (Tensor[..., 1 or 3, H, W]):
                Image to be adjusted. If img is Tensor, it is expected to
                be in [..., 1 or 3, H, W] format, where ... means it can have
                an arbitrary number of leading dimensions.

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Gamma correction adjusted image.
        """
        return adjust_gamma(image, self.gamma, self.gain)


@TRANSFORMS.register(name="adjust_hsv")
class AdjustHsv(nn.Module):
    """Adjust HSV of an image.

    Args:
        h_factor (float):
            How much to shift the hue channel.
        s_factor (float):
            How much to shift the saturation channel.
        v_factor (float):
            How much to shift the value channel.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        h_factor: float = 0.5,
        s_factor: float = 0.5,
        v_factor: float = 0.5,
    ):
        super().__init__()
        self.h_factor = h_factor
        self.s_factor = s_factor
        self.v_factor = v_factor

    # MARK: Forward Pass
    
    def forward(self, image: TensorOrArray) -> TensorOrArray:
        """

        Args:
            image (TensorOrArray[..., 1 or 3, H, W]):
                If image is Tensor, it is expected to be in
                [..., 1 or 3, H, W] format, where ... means it can have an
                arbitrary number of leading dimensions. If img is PIL Image
                mode "1", "I", "F" and modes with transparency (alpha channel)
                are not supported.

        Returns:
            (TensorOrArray[..., 1 or 3, H, W]):
                Hue adjusted image.
        """
        return adjust_hsv(image, self.h_factor, self.s_factor, self.v_factor)
    

@TRANSFORMS.register(name="adjust_hue")
class AdjustHue(nn.Module):
    """Adjust hue of an image.
    
    Image hue is adjusted by converting the image to HSV and cyclically
    shifting the intensities in the hue channel (H). Image is then
    converted back to original image mode.
    
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    
    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue
    
    Args:
        hue_factor (float, Tensor):
            How much to shift the hue channel. Should be in [-0.5, 0.5].
            0.5 and -0.5 give complete reversal of hue channel in HSV space in
            positive and negative direction respectively. 0 means no shift.
            Therefore, both -0.5 and 0.5 will give an image with complementary
            colors while 0 gives the original image.
 
    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> AdjustHue(3.141516)(x)
        image([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x = torch.ones(2, 3, 3, 3)
        >>> y = torch.ones(2) * 3.141516
        >>> AdjustHue(y)(x).shape
        torch.Size([2, 3, 3, 3])
    """

    # MARK: Magic Functions
    
    def __init__(self, hue_factor: float):
        super().__init__()
        self.hue_factor = hue_factor

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (Tensor[..., 1 or 3, H, W]):
                If image is Tensor, it is expected to be in
                [..., 1 or 3, H, W] format, where ... means it can have an
                arbitrary number of leading dimensions. If img is PIL Image
                mode "1", "I", "F" and modes with transparency (alpha channel)
                are not supported.

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Hue adjusted image.
        """
        return adjust_hue(image, self.hue_factor)


@TRANSFORMS.register(name="adjust_saturation")
class AdjustSaturation(nn.Module):
    """Adjust color saturation of an image.

    Args:
        saturation_factor (float, Tensor):
            How much to adjust the saturation. 0 will give a black and white
            image, 1 will give the original image while 2 will enhance the
            saturation by a factor of 2.

    Example:
        >>> x = torch.ones(1, 3, 3, 3)
        >>> AdjustSaturation(2.)(x)
        image([[[[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]],
        <BLANKLINE>
                 [[1., 1., 1.],
                  [1., 1., 1.],
                  [1., 1., 1.]]]])

        >>> x   = torch.ones(2, 3, 3, 3)
        >>> y   = 2
        >>> out = AdjustSaturation(y)(x)
        >>> torch.nn.functional.mse_loss(x, out)
        image(0.)
    """

    # MARK: Magic Functions
    
    def __init__(self, saturation_factor: float):
        super().__init__()
        self.saturation_factor = saturation_factor

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (Tensor[..., 1 or 3, H, W]):
                Image to be adjusted. If img is Tensor, it is expected to
                be in [..., 1 or 3, H, W] format, where ... means it can have
                an arbitrary number of leading dimensions.

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Saturation adjusted image.
        """
        return adjust_saturation(image, self.saturation_factor)


@TRANSFORMS.register(name="adjust_sharpness")
class AdjustSharpness(nn.Module):
    """Adjust the sharpness of an image.

    Args:
        sharpness_factor (float):
            How much to adjust the sharpness. Can be any non-negative number.
            0 gives a blurred image, 1 gives the original image while 2
            increases the sharpness by a factor of 2.
    """

    # MARK: Magic Functions
    
    def __init__(self, sharpness_factor: float):
        super().__init__()
        self.sharpness_factor = sharpness_factor

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """

        Args:
            image (Tensor[..., 1 or 3, H, W]):
                Image to be adjusted. If img is Tensor, it is expected to
                be in [..., 1 or 3, H, W] format, where ... means it can have
                an arbitrary number of leading dimensions.

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Sharpness adjusted image.
        """
        return adjust_sharpness(image, self.sharpness_factor)
    

# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="auto_contrast")
class AutoContrast(nn.Module):
    """Maximize contrast of an image by remapping its pixels per channel so
    that the lowest becomes black and the lightest becomes white.
    """

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """

        Args:
            image (Tensor[..., 1 or 3, H, W]):
                Image to be adjusted. If img is Tensor, it is expected to
                be in [..., 1 or 3, H, W] format, where ... means it can have
                an arbitrary number of leading dimensions.

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                An image that was auto contrasted.
        """
        return autocontrast(image)
    

# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="equalize")
class Equalize(nn.Module):
    """Equalize the histogram of an image by applying a non-linear mapping to
    the input in order to create a uniform distribution of grayscale values in
    the output.
    """

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """

        Args:
            image (Tensor[..., 1 or 3, H, W]):
                If img is Tensor, it is expected to be in
                [..., 1 or 3, H, W] format, where ... means it can have an
                arbitrary number of leading dimensions. Tensor dtype must
                be `torch.uint8` and values are expected to be in `[0, 255]`.
                If img is PIL Image, it is expected to be in mode "P", "L" or
                "RGB".

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                An image that was equalized.
        """
        return equalize(image)
    

# noinspection PyMethodMayBeStatic
@TRANSFORMS.register(name="invert")
class Invert(nn.Module):
    """Invert the colors of an RGB/grayscale image.

    Example:
        >>> img = torch.rand(1, 2, 4, 4)
        >>> Invert()(img).shape
        torch.Size([1, 2, 4, 4])

        >>> img = 255. * torch.rand(1, 2, 3, 4, 4)
        >>> Invert(torch.tensor(255.))(img).shape
        torch.Size([1, 2, 3, 4, 4])

        >>> img = torch.rand(1, 3, 4, 4)
        >>> Invert(torch.tensor([[[[1.]]]]))(img).shape
        torch.Size([1, 3, 4, 4])
    """

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """

        Args:
            image (Tensor[..., 1 or 3, H, W]):
                Image to have its colors inverted. If img is Tensor, it
                is expected to be in [..., 1 or 3, H, W] format, where ...
                means it can have an arbitrary number of leading dimensions.
                If img is PIL Image, it is expected to be in mode "L" or "RGB".

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Color inverted image.
        """
        return invert(image)


@TRANSFORMS.register(name="posterize")
class Posterize(nn.Module):
    """Posterize an image by reducing the number of bits for each color channel.

    Args:
        bits (int):
            Number of bits to keep for each channel (0-8).
    """

    # MARK: Magic Functions
    
    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """

        Args:
            image (Tensor[..., 1 or 3, H, W]):
                If img is Tensor, it should be of type torch.uint8 and it is
                expected to be in [..., 1 or 3, H, W] format, where ... means it
                can have an arbitrary number of leading dimensions. If img is
                PIL Image, it is expected to be in mode "L" or "RGB".

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Posterized image.
        """
        return posterize(image, self.bits)


@TRANSFORMS.register(name="solarize")
class Solarize(nn.Module):
    """Solarize an RGB/grayscale image by inverting all pixel values above a
    threshold.

    Args:
        threshold (float):
            All pixels equal or above this value are inverted.
    """

    # MARK: Magic Functions
    
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    # MARK: Forward Pass
    
    def forward(self, image: Tensor) -> Tensor:
        """

        Args:
            image (Tensor[..., 1 or 3, H, W]):
                If img is Tensor, it is expected to be in
                [..., 1 or 3, H, W] format, where ... means it can have an
                arbitrary number of leading dimensions. If img is PIL Image,
                it is expected to be in mode "L" or "RGB".

        Returns:
            (Tensor[..., 1 or 3, H, W]):
                Solarized image.
        """
        return solarize(image, self.threshold)
