#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transformation on pixel intensity.
"""

from __future__ import annotations

import math
import numbers

import torchvision.transforms.functional_tensor as F_t
from torch import Tensor

from one.constants import *
from one.core import *
from one.vision.acquisition import get_image_shape
from one.vision.acquisition import get_num_channels
from one.vision.transformation.color import hsv_to_rgb
from one.vision.transformation.color import rgb_to_grayscale
from one.vision.transformation.color import rgb_to_hsv


# H1: - Adjust ---------------------------------------------------------------

def add_weighted(
    image1: Tensor,
    alpha : float,
    image2: Tensor,
    beta  : float,
    gamma : float = 0.0,
) -> Tensor:
    """
    Calculate the weighted sum of two tensors.
    
    Function calculates the weighted sum of two tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1 (Tensor): First image of shape [..., C, H, W].
        alpha (float): Weight of the image1 elements.
        image2 (Tensor): Second image of same shape as `src1`.
        beta (float): Weight of the image2 elements.
        gamma (float): Scalar added to each sum. Defaults to 0.0.

    Returns:
        Weighted image of shape [..., C, H, W].
    """
    assert_tensor(image1)
    assert_tensor(image2)
    assert_same_shape(image1, image2)
    assert_float(alpha)
    assert_float(beta)
    assert_float(gamma)
    bound  = 1.0 if image1.is_floating_point() else 255.0
    output = image1 * alpha + image2 * beta + gamma
    output = output.clamp(0, bound).to(image1.dtype)
    return output


def adjust_brightness(image: Tensor, brightness_factor: float) -> Tensor:
    """
    Adjust brightness of an image.

    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        brightness_factor (float): How much to adjust the brightness. Can be
            any non-negative number. 0 gives a black image, 1 gives the original
            image while 2 increases the brightness by a factor of 2.
        
    Returns:
        Brightness adjusted image of shape [..., 1 or 3, H, W].
    """
    assert_positive_number(brightness_factor)
    assert_tensor_of_channels(image, [1, 3])
    return blend(
	    image1 = image,
	    alpha  = brightness_factor,
	    image2 = torch.zeros_like(image)
    )


def adjust_contrast(image: Tensor, contrast_factor: float) -> Tensor:
    """
    Adjust contrast of an image.

    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non-negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        Contrast adjusted image of shape [..., 1 or 3, H, W].
    """
    assert_positive_number(contrast_factor)
    assert_tensor_of_channels(image, [1, 3])
    c     = get_num_channels(image)
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    if c == 3:
        mean = torch.mean(
            rgb_to_grayscale(image=image).to(dtype),
            dim=(-3, -2, -1), keepdim=True
        )
    else:
        mean = torch.mean(image.to(dtype), dim=(-3, -2, -1), keepdim=True)
    return blend(image1=image, alpha=contrast_factor, image2=mean)


def adjust_gamma(image: Tensor, gamma: float, gain: float = 1.0) -> Tensor:
    """
    Adjust gamma of an image.

    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        gamma (float): How much to adjust the gamma. Can be any non-negative
            number. 0 gives a black image, 1 gives the original image while 2
            increases the brightness by a factor of 2.
        gain (float): Default to 1.0.
        
    Returns:
        Gamma adjusted image f shape [..., 1 or 3, H, W].
    """
    assert_positive_number(gamma)
    assert_tensor_of_channels(image, [1, 3])

    result = image
    dtype  = image.dtype
    if not torch.is_floating_point(image):
        result = F_t.convert_image_dtype(result, torch.float32)
    result = (gain * result ** gamma).clamp(0, 1)
    result = F_t.convert_image_dtype(result, dtype)
    return result


def adjust_hue(image: Tensor, hue_factor: float) -> Tensor:
    """
    Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and cyclically
    shifting the intensities in the hue channel (H). The image is then
    converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        hue_factor (float): How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively. 0 means
            no shift. Therefore, both -0.5 and 0.5 will give an image with
            complementary colors while 0 gives the original image.

    Returns:
        Hue adjusted image of shape [..., 1 or 3, H, W].
    """
    assert_number_in_range(hue_factor, -0.5, 0.5)
    assert_tensor_of_channels(image, [1, 3])

    orig_dtype = image.dtype
    if image.dtype == torch.uint8:
        image  = image.to(dtype=torch.float32) / 255.0

    image       = rgb_to_hsv(image)
    h, s, v     = image.unbind(dim=-3)
    h           = (h + hue_factor) % 1.0
    image       = torch.stack((h, s, v), dim=-3)
    img_hue_adj = hsv_to_rgb(image)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj


def adjust_saturation(image: Tensor, saturation_factor: float) -> Tensor:
    """
    Adjust color saturation of an image.

    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        saturation_factor (float): How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        Saturation adjusted image of shape [..., 1 or 3, H, W].
    """
    assert_positive_number(saturation_factor)
    assert_tensor_of_channels(image, [1, 3])
    
    if get_num_channels(image) == 1:
        return image
    
    return blend(
        image1 = image,
        alpha  = saturation_factor,
        image2 = rgb_to_grayscale(image=image)
    )


def adjust_sharpness(image: Tensor, sharpness_factor: float) -> Tensor:
    """
    Adjust sharpness of an image.
    
    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be adjusted ,
            where ... means it can have an arbitrary number of leading
            dimensions.
        sharpness_factor (float): How much to adjust the sharpness. 0 will give
            a black and white image, 1 will give the original image while 2 will
            enhance the saturation by a factor of 2.
    
    Returns:
        Sharpness adjusted image of shape [..., 1 or 3, H, W].
    """
    assert_positive_number(sharpness_factor)
    assert_tensor_of_channels(image, [1, 3])
    
    if image.size(-1) <= 2 or image.size(-2) <= 2:
        return image

    return blend(
        image1 = image,
        image2 = F_t._blurred_degenerate_image(img=image),
        alpha  = sharpness_factor,
    )


def autocontrast(image: Tensor) -> Tensor:
    """
    Maximize contrast of an image by remapping its pixels per channel so that
    the lowest becomes black and the lightest becomes white.
    
    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be adjusted ,
            where ... means it can have an arbitrary number of leading
            dimensions.
    
    Returns:
        Auto-contrast adjusted image of shape [..., 1 or 3, H, W].
    """
    assert_tensor_of_channels(image, [1, 3])
    assert_tensor_of_atleast_ndim(image, 3)

    bound = 1.0 if image.is_floating_point() else 255.0
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32

    minimum          = image.amin(dim=(-2, -1), keepdim=True).to(dtype)
    maximum          = image.amax(dim=(-2, -1), keepdim=True).to(dtype)
    scale            = bound / (maximum - minimum)
    eq_idxs          = torch.isfinite(scale).logical_not()
    minimum[eq_idxs] = 0
    scale[eq_idxs]   = 1

    return ((image - minimum) * scale).clamp(0, bound).to(image.dtype)


def blend(
    image1: Tensor,
    image2: Tensor,
    alpha : float,
    gamma : float = 0.0
) -> Tensor:
    """
    Blends 2 images together:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        image1 (Tensor): Source image of shape [..., C, H, W].
        image2 (Tensor): Second image of shape [..., C, H, W] that we want to
            overlay on top of `image1`.
        alpha (float): Alpha transparency of the overlay.
        gamma (float): Scalar added to each sum. Defaults to 0.0.

    Returns:
        Blended image of shape [..., C, H, W].
    """
    return add_weighted(
        image1 = image2,
        alpha  = alpha,
        image2 = image1,
        beta   = 1.0 - alpha,
        gamma  = gamma
    )


def denormalize(
    image  : Tensor,
    mean   : Tensor | Floats = IMG_MEAN,
    std    : Tensor | Floats = IMG_STD,
    eps    : float = 1e-6,
    inplace: bool  = False,
) -> Tensor:
    """
    Denormalize an image Tensor with mean and standard deviation.
    
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        image (Tensor): Float image of shape [..., C, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        mean (Tensor | Floats): Sequence of means for each channel.
            Defaults to IMG_MEAN.
        std (Tensor | Floats): Sequence of standard deviations for each
            channel. Defaults to IMG_STD.
        eps (float): Float number to avoid zero division. Defaults to 1e-6.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Denormalized image with same size as input.
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if not image.is_floating_point():
        raise TypeError(f"Input tensor should be a float Tensor. Got {image.dtype}.")
    if not inplace:
        image = image.clone()
    
    shape  = image.shape
    device = image.device
    dtype  = image.dtype
    if isinstance(mean, float):
        mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
    elif isinstance(mean, (list, tuple)):
        mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
    elif isinstance(mean, Tensor):
        mean = mean.to(dtype=dtype, device=image.device)
    
    if isinstance(std, float):
        std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
    elif isinstance(std, (list, tuple)):
        std = torch.as_tensor(std, dtype=dtype, device=image.device)
    elif isinstance(std, Tensor):
        std = std.to(dtype=dtype, device=image.device)
        
    std_inv  = 1.0 / (std + eps)
    mean_inv = -mean * std_inv
    std_inv  = std_inv.view(-1, 1, 1)  if std_inv.ndim == 1  else std_inv
    mean_inv = mean_inv.view(-1, 1, 1) if mean_inv.ndim == 1 else mean_inv
    image.sub_(mean_inv).div_(std_inv)
    return image


def denormalize_simple(
    image  : Tensor,
    inplace: bool = False,
) -> Tensor:
    """
    Denormalize an image Tensor with mean and standard deviation.
    
    Args:
        image (Tensor): Float image of shape [..., C, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        
    Returns:
        Denormalized image with same size as input.
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if not image.is_floating_point():
        raise TypeError(f"Input tensor should be a float Tensor. Got {image.dtype}.")
    if not inplace:
        image = image.clone()
    
    image = torch.clamp(image * 255, 0, 255)
    return image

def erase(
    image  : Tensor,
    i      : int,
    j      : int,
    h      : int,
    w      : int,
    v      : Tensor,
    inplace: bool = False
) -> Tensor:
    """
    Erase the input Tensor Image with given value.

    Args:
        image (Tensor): Image of shape [..., C, H, W] to be adjusted, where ...
            means it can have an arbitrary number of leading dimensions.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v (Tensor):  Erasing value.
        inplace (bool): If True, make this operation inplace. Defaults to False.

    Returns:
        Erased image of shape [..., C, H, W].
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if not inplace:
        image = image.clone()
    image[..., i: i + h, j: j + w] = v
    return image


def equalize(image: Tensor) -> Tensor:
    """
    Equalize the histogram of an image by applying a non-linear mapping to the
    input in order to create a uniform distribution of grayscale values in
    the output.
    
    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be adjusted ,
            where ... means it can have an arbitrary number of leading
            dimensions.
    
    Returns:
        Equalized image of shape [..., 1 or 3, H, W].
    """
    assert_tensor_of_ndim_in_range(image, 3, 4)
    assert_tensor_of_channels(image, [1, 3])
    
    if image.dtype != torch.uint8:
        raise TypeError(
            f"Only `torch.uint8` image tensors are supported. "
            f"But got: {image.dtype}."
        )
    if image.ndim == 3:
        return F_t._equalize_single_image(image)

    return torch.stack([F_t._equalize_single_image(x) for x in image])


def invert(image: Tensor) -> Tensor:
    """
    Invert the colors of an RGB/grayscale image.
    
    Args:
        image (Tensor): Image of shape [..., 1 or 3, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
      
    Returns:
        Inverted image of shape [..., 1 or 3, H, W].
    """
    assert_tensor_of_channels(image, [1, 3])
    bound = torch.tensor(
        data   = 1 if image.is_floating_point() else 255,
        dtype  = image.dtype,
        device = image.device
    )
    return bound - image
  

def normalize(
    image  : Tensor,
    mean   : Tensor | Floats = IMG_MEAN,
    std    : Tensor | Floats = IMG_STD,
    eps    : float           = 1e-6,
    inplace: bool            = False,
) -> Tensor:
    """
    Normalize a float tensor image with mean and standard deviation.

    Args:
        image (Tensor): Float image of shape [..., C, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        mean (Tensor | Floats): Sequence of means for each channel.
            Defaults to IMG_MEAN.
        std (Tensor | Floats): Sequence of standard deviations for each
            channel. Defaults to IMG_STD.
        eps (float): Float number to avoid zero division. Defaults to 1e-6.
        inplace (bool): If True, make this operation inplace. Defaults to False.

    Returns:
        Normalized image of shape [..., C, H, W].
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if not inplace:
        image = image.clone()
    
    image = image.to(dtype=torch.float32)
    if not image.is_floating_point():
        raise TypeError(
            f"Input tensor should be a float Tensor. Got {image.dtype}."
        )
    
    shape  = image.shape
    device = image.device
    dtype  = image.dtype
    if isinstance(mean, float):
        mean = torch.tensor([mean] * shape[1], device=device, dtype=dtype)
    elif isinstance(mean, (list, tuple)):
        mean = torch.as_tensor(mean, dtype=dtype, device=image.device)
    elif isinstance(mean, Tensor):
        mean = mean.to(dtype=dtype, device=image.device)
    
    if isinstance(std, float):
        std = torch.tensor([std] * shape[1], device=device, dtype=dtype)
    elif isinstance(std, (list, tuple)):
        std = torch.as_tensor(std, dtype=dtype, device=image.device)
    elif isinstance(std, Tensor):
        std = std.to(dtype=dtype, device=image.device)
    std  += eps
    
    mean  = mean.view(-1, 1, 1) if mean.ndim == 1 else mean
    std   = std.view(-1, 1, 1)  if std.ndim == 1  else std
    image.sub_(mean).div_(std)
    return image


def normalize_simple(
    image  : Tensor,
    inplace: bool = False,
) -> Tensor:
    """
    Normalize a float tensor image with mean and standard deviation.

    Args:
        image (Tensor): Float image of shape [..., C, H, W] to be adjusted,
            where ... means it can have an arbitrary number of leading
            dimensions.
        inplace (bool): If True, make this operation inplace. Defaults to False.

    Returns:
        Normalized image of shape [..., C, H, W].
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if not inplace:
        image = image.clone()
    
    image = image.to(dtype=torch.float32)
    if not image.is_floating_point():
        raise TypeError(
            f"Input tensor should be a float Tensor. Got {image.dtype}."
        )
    image = image / 255.0
    return image


def posterize(image: Tensor, bits: int) -> Tensor:
    """
    Posterize an image by reducing the number of bits for each color channel.
    
    Args:
        image (Tensor): Image of shape [..., C, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        bit (int): Number of bits to keep for each channel (0-8).
        
    Returns:
        Posterized image of shape [..., C, H, W].
    """
    assert_tensor_of_channels(image, [1, 3])
    if image.dtype != torch.uint8:
        raise TypeError(
            f"Only `torch.uint8` image tensors are supported. "
            f"But got: {image.dtype}"
        )
    mask = -int(2 ** (8 - bits))  # JIT-friendly for: ~(2 ** (8 - bits) - 1)
    return image & mask


def solarize(image: Tensor, threshold: float) -> Tensor:
    """
    Solarize an RGB/grayscale image by inverting all pixel values above a
    threshold.

    Args:
        image (Tensor):Image of shape [..., C, H, W] to be transformed, where
            ... means it can have an arbitrary number of leading dimensions.
        threshold (float): All pixels equal or above this value are inverted.
            
    Returns:
        Solarized image of shape [..., C, H, W].
    """
    assert_tensor_of_channels(image, [1, 3])
    
    bound = 1 if image.is_floating_point() else 255
    if threshold > bound:
        raise TypeError("Threshold should be less than bound of img.")
    
    inverted_img = invert(image)
    return torch.where(image >= threshold, inverted_img, image)


@TRANSFORMS.register(name="add_weighted")
class AddWeighted(Transform):
    """
    Calculate the weighted sum of two tensors.
    
    Function calculates the weighted sum of two tensors as follows:
        output = image1 * alpha + image2 * beta + gamma

    Args:
        alpha (float): Weight of the image1 elements.
        beta (float): Weight of the image2 elements.
        gamma (float): Scalar added to each sum. Defaults to 0.0.
        p (float | None): Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        alpha: float,
        beta : float,
        gamma: float,
        p    : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    # noinspection PyMethodOverriding
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return add_weighted(
            image1 = input,
            alpha  = self.alpha,
            image2 = target,
            beta   = self.beta,
            gamma  = self.gamma
        )


@TRANSFORMS.register(name="adjust_brightness")
class AdjustBrightness(Transform):
    """
    Adjust brightness of an image.

    Args:
        brightness_factor (float): How much to adjust the brightness. Can be
            any non-negative number. 0 gives a black image, 1 gives the original
            image while 2 increases the brightness by a factor of 2.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        brightness_factor: float,
        p                : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.brightness_factor = brightness_factor
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            adjust_brightness(
                image             = input,
                brightness_factor = self.brightness_factor
            ), \
            adjust_brightness(
                image             = target,
                brightness_factor = self.brightness_factor
            ) if target is not None else None


@TRANSFORMS.register(name="adjust_contrast")
class AdjustContrast(Transform):
    """
    Adjust contrast of an image.

    Args:
        contrast_factor (float):
            How much to adjust the contrast. Can be any non-negative number.
            0 gives a solid gray image, 1 gives the original image while 2
            increases the contrast by a factor of 2.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        contrast_factor: float,
        p              : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.contrast_factor = contrast_factor
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            adjust_contrast(
                image           = input,
                contrast_factor = self.contrast_factor
            ), \
            adjust_contrast(
                image           = target,
                contrast_factor = self.contrast_factor
            ) if target is not None else None


@TRANSFORMS.register(name="adjust_gamma")
class AdjustGamma(Transform):
    """
    Adjust gamma of an image.

    Args:
        gamma (float): How much to adjust the gamma. Can be any non-negative
            number. 0 gives a black image, 1 gives the original image while 2
            increases the brightness by a factor of 2.
        gain (float): Default to 1.0.
        p (float | None): Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        gamma: float,
        gain : float        = 1.0,
        p    : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.gamma = gamma
        self.gain  = gain
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            adjust_gamma(
                image = input,
                gamma = self.gamma,
                gain  = self.gain
            ), \
            adjust_gamma(
                image = target,
                gamma = self.gamma,
                gain  = self.gain
            ) if target is not None else None


@TRANSFORMS.register(name="adjust_hue")
class AdjustHue(Transform):
    """
    Adjust hue of an image.

    Args:
        hue_factor (float): How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively. 0 means
            no shift. Therefore, both -0.5 and 0.5 will give an image with
            complementary colors while 0 gives the original image.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        hue_factor: float,
        p         : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.hue_factor = hue_factor
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            adjust_hue(
                image      = input,
                hue_factor = self.hue_factor
            ), \
            adjust_hue(
                image      = target,
                hue_factor = self.hue_factor
            ) if target is not None else None
        

@TRANSFORMS.register(name="adjust_saturation")
class AdjustSaturation(Transform):
    """
    Adjust color saturation of an image.

    Args:
        saturation_factor (float): How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        saturation_factor: float,
        p                : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.saturation_factor = saturation_factor
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            adjust_saturation(
                image             = input,
                saturation_factor = self.saturation_factor
            ), \
            adjust_saturation(
                image             = target,
                saturation_factor = self.saturation_factor
            ) if target is not None else None
     

@TRANSFORMS.register(name="adjust_sharpness")
class AdjustSharpness(Transform):
    """
    Adjust color sharpness of an image.

    Args:
        sharpness_factor (float): How much to adjust the sharpness. 0 will give
            a black and white image, 1 will give the original image while 2
            will enhance the saturation by a factor of 2.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        sharpness_factor: float,
        p               : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.sharpness_factor = sharpness_factor
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            adjust_sharpness(
                image            = input,
                sharpness_factor = self.sharpness_factor
            ), \
            adjust_sharpness(
                image            = target,
                sharpness_factor = self.sharpness_factor
            ) if target is not None else None


@TRANSFORMS.register(name="autocontrast")
class AutoContrast(Transform):
    """
    Maximize contrast of an image by remapping its pixels per channel so that
    the lowest becomes black and the lightest becomes white.
    
    Args:
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return autocontrast(image=input),  \
               autocontrast(image=target) if target is not None else None
          

@TRANSFORMS.register(name="add_weighted")
class Blend(Transform):
    """
    Blends 2 images together.

    Args:
        alpha (float): Alpha transparency of the overlay.
        gamma (float): Scalar added to each sum. Defaults to 0.0.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        alpha: float,
        gamma: float,
        p    : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    # noinspection PyMethodOverriding
    def forward(self, input : Tensor, target: Tensor, *args, **kwargs) -> Tensor:
        return blend(
            image1 = input,
            image2 = target,
            alpha  = self.alpha,
            gamma  = self.gamma,
        )


@TRANSFORMS.register(name="color_jitter")
class ColorJitter(Transform):
    """
    Randomly change the brightness, contrast, saturation and hue of an image.
    
    Args:
        brightness (Floats | None): How much to jitter brightness.
            `brightness_factor` is chosen uniformly from
            [max(0, 1 - brightness), 1 + brightness] or the given [min, max].
            Should be non negative numbers. Defaults to 0.0.
        contrast (Floats | None): How much to jitter contrast. `contrast_factor`
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the
            given [min, max]. Should be non-negative numbers. Defaults to 0.0.
        saturation (Floats | None): How much to jitter saturation.
            `saturation_factor` is chosen uniformly from
            [max(0, 1 - saturation), 1 + saturation] or the given [min, max].
            Should be non-negative numbers. Defaults to 0.0.
        hue (Floats | None): How much to jitter hue. `hue_factor` is chosen
            uniformly from [-hue, hue] or the given [min, max]. Should have
            0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5. Defaults to 0.0.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        brightness: Floats | None = 0.0,
        contrast  : Floats | None = 0.0,
        saturation: Floats | None = 0.0,
        hue       : Floats | None = 0.0,
        p         : float  | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast   = self._check_input(contrast,   "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue        = self._check_input(
            hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False
        )
    
    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}, "
            f"contrast={self.contrast}, "
            f"saturation={self.saturation}, "
            f"hue={self.hue})"
        )
        return s
        
    @torch.jit.unused
    def _check_input(
        self,
        value             : Any,
        name              : str,
        center            : int   = 1,
        bound             : tuple = (0, float("inf")),
        clip_first_on_zero: bool  = True
    ) -> Floats | None:
        if isinstance(value, numbers.Number):
            assert_positive_number(value)
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)):
            assert_sequence_of_length(value, 2)
            assert_number_in_range(value[0], bound[0], bound[1])
            assert_number_in_range(value[1], bound[0], bound[1])
        else:
            raise TypeError(
                f"{name} should be a single number or a list/tuple with "
                f"length 2."
            )

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value
    
    @staticmethod
    def get_params(
        brightness: Floats | None,
        contrast  : Floats | None,
        saturation: Floats | None,
        hue       : Floats | None,
    ) -> tuple[
        Tensor,
        float | None,
        float | None,
        float | None,
        float | None
    ]:
        """Get the parameters for the randomized transform to be applied on
        image.

        Args:
            brightness (Floats | None): The range from which the
                `brightness_factor` is chosen uniformly. Pass None to turn off
                the transformation.
            contrast (Floats | None): The range from which the `contrast_factor`
                is chosen uniformly.  Pass None to turn off the transformation.
            saturation (Floats | None): The range from which the
                `saturation_factor` is chosen uniformly. Pass None to turn off
                the transformation.
            hue (Floats | None): The range from which the `hue_factor` is chosen
                uniformly. Pass None to turn off the transformation.

        Returns:
            The parameters used to apply the randomized transform along with
                their random order.
        """
        fn_idx = torch.randperm(4)
        b = None if brightness is None \
            else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast   is None \
            else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None \
            else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None \
            else float(torch.empty(1).uniform_(hue[0], hue[1]))
        return fn_idx, b, c, s, h
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        fn_idx, \
        brightness_factor, contrast_factor, saturation_factor, hue_factor \
            = self.get_params(
                brightness = self.brightness,
                contrast   = self.contrast,
                saturation = self.saturation,
                hue        = self.hue
            )

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                input  = adjust_brightness(input,  brightness_factor)
                target = adjust_brightness(target, brightness_factor) \
                    if target is not None else None
            elif fn_id == 1 and contrast_factor is not None:
                input  = adjust_contrast(input, contrast_factor)
                target = adjust_contrast(target, contrast_factor) \
                    if target is not None else None
            elif fn_id == 2 and saturation_factor is not None:
                input  = adjust_saturation(input,  saturation_factor)
                target = adjust_saturation(target, saturation_factor) \
                    if target is not None else None
            elif fn_id == 3 and hue_factor is not None:
                input  = adjust_hue(input,  hue_factor)
                target = adjust_hue(target, hue_factor) \
                    if target is not None else None

        return input, target


@TRANSFORMS.register(name="denormalize")
class Denormalize(Transform):
    """
    Denormalize an image Tensor with mean and standard deviation.
 
    image[channel] = (image[channel] * std[channel]) + mean[channel]
    where `mean` is [M_1, ..., M_n] and `std` [S_1, ..., S_n] for `n` channels.

    Args:
        mean (Tensor | Floats): Sequence of means for each channel.
            Defaults to IMG_MEAN.
        std (Tensor | Floats): Sequence of standard deviations for each
            channel. Defaults to IMG_STD.
        eps (float): Float number to avoid zero division. Defaults to 1e-6.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        mean   : Tensor | Floats = IMG_MEAN,
        std    : Tensor | Floats = IMG_STD,
        eps    : float           = 1e-6,
        inplace: bool            = False,
        p      : float | None    = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.mean    = mean
        self.std     = std
        self.eps     = eps
        self.inplace = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            denormalize(
                image   = input,
                mean    = self.mean,
                std     = self.std,
                eps     = self.eps,
                inplace = self.inplace
            ), \
            denormalize(
                image   = target,
                mean    = self.mean,
                std     = self.std,
                eps     = self.eps,
                inplace = self.inplace
            ) if target is not None else None


@TRANSFORMS.register(name="erase")
class Erase(Transform):
    """
    Equalize the histogram of an image by applying a non-linear mapping to
    the input in order to create a uniform distribution of grayscale values in
    the output.
    
    Args:
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the erased region.
        w (int): Width of the erased region.
        v (Tensor): Erasing value.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        i      : int,
        j      : int,
        h      : int,
        w      : int,
        v      : Tensor,
        inplace: bool         = False,
        p      : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.i       = i
        self.j       = j
        self.h       = h
        self.w       = w
        self.v       = v
        self.inplace = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            erase(
                image   = input,
                i       = self.i,
                j       = self.j,
                h       = self.h,
                w       = self.w,
                v       = self.v,
                inplace = self.inplace,
            ), \
            erase(
                image   = target,
                i       = self.i,
                j       = self.j,
                h       = self.h,
                w       = self.w,
                v       = self.v,
                inplace = self.inplace,
            ) if target is not None else None


@TRANSFORMS.register(name="equalize")
class Equalize(Transform):
    """
    Equalize the histogram of an image by applying a non-linear mapping to
    the input in order to create a uniform distribution of grayscale values in
    the output.
    
    Args:
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return equalize(image=input), \
               equalize(image=target) if target is not None else None


@TRANSFORMS.register(name="invert")
class Invert(Transform):
    """
    Invert the colors of an RGB/grayscale image.
    
    Args:
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """

    def __init__(self, p: float | None = None, *args, **kwargs):
        super().__init__(p=p, *args, **kwargs)
   
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return invert(image=input), \
               invert(image=target) if target is not None else None


@TRANSFORMS.register(name="normalize")
class Normalize(Transform):
    """
    Normalize a tensor image with mean and standard deviation.
 
    Args:
        mean (Tensor | Floats): Sequence of means for each channel.
            Defaults to IMG_MEAN.
        std (Tensor | Floats): Sequence of standard deviations for each
            channel. Defaults to IMG_STD.
        eps (float): Float number to avoid zero division. Defaults to 1e-6.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to
            None means process as normal.
    """
    
    def __init__(
        self,
        mean   : Tensor | Floats = IMG_MEAN,
        std    : Tensor | Floats = IMG_STD,
        eps    : float           = 1e-6,
        inplace: bool            = False,
        p      : float | None    = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.mean    = mean
        self.std     = std
        self.eps     = eps
        self.inplace = inplace
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return \
            normalize(
                image   = input,
                mean    = self.mean,
                std     = self.std,
                eps     = self.eps,
                inplace = self.inplace
            ), \
            normalize(
                image   = target,
                mean    = self.mean,
                std     = self.std,
                eps     = self.eps,
                inplace = self.inplace
            ) if target is not None else None


@TRANSFORMS.register(name="posterize")
class Posterize(Transform):
    """
    Posterize an image by reducing the number of bits for each color channel.

    Args:
        bits (int): Number of bits to keep for each channel (0-8).
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        bits: int,
        p   : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.bits = bits
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return posterize(image=input,  bits=self.bits), \
               posterize(image=target, bits=self.bits) \
                   if target is not None else None


@TRANSFORMS.register(name="random_erase")
class RandomErase(Transform):
    """
    Randomly selects a rectangle region in an image Tensor and erases its
    pixels.
    
    References:
        'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896
    
    Args:
        scale (Floats): Range of proportion of erased area against input image.
        ratio (Floats): Range of aspect ratio of erased area.
        value (int | float | str | tuple | list): Erasing value. Defaults to 0.
            If a single int, it is used to erase all pixels. If a tuple of
            length 3, it is used to erase R, G, B channels respectively. If a
            str of `random`, erasing each pixel with random values.
        inplace (bool): If True, make this operation inplace. Defaults to False.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        scale  : Floats                           = (0.02, 0.33),
        ratio  : Floats                           = (0.3, 3.3),
        value  : int | float | str | tuple | list = 0,
        inplace: bool                             = False,
        p      : float | None                     = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        if not isinstance(value, (numbers.Number, str, tuple, list)):
            raise TypeError(
                "Argument value should be either a number or str or a sequence."
            )
        if isinstance(value, str) and value != "random":
            raise ValueError("If value is str, it should be 'random'.")
        if not isinstance(scale, (tuple, list)):
            raise TypeError("Scale should be a sequence.")
        if not isinstance(ratio, (tuple, list)):
            raise TypeError("Ratio should be a sequence.")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            print("Scale and ratio should be of kind (min, max).")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("Scale should be between 0 and 1.")
        
        self.scale   = scale
        self.ratio   = ratio
        self.value   = value
        self.inplace = inplace
        
    @staticmethod
    def get_params(
        image: Tensor,
        scale: Floats,
        ratio: Floats,
        value: list[float] | None = None
    ) -> tuple[int, int, int, int, Tensor]:
        """Get parameters for `erase` for a random erasing.

        Args:
            image (Tensor): Tensor image to be erased.
            scale (Floats): Range of proportion of erased area against input
                image.
            ratio (Floats): Range of aspect ratio of erased area.
            value (list[float] | None): Erasing value. If None, it is
                interpreted as "random" (erasing each pixel with random values).
                If `len(value)` is 1, it is interpreted as a number, i.e.
                `value[0]`.

        Returns:
            Params (i, j, h, w, v) to be passed to `erase` for random erasing.
        """
        img_c, img_h, img_w = get_image_shape(image)
        area                = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area   = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, image
        
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        if isinstance(self.value, (int, float)):
            value = [self.value]
        elif isinstance(self.value, str):
            value = None
        elif isinstance(self.value, tuple):
            value = list(self.value)
        else:
            value = self.value
    
        if value is not None and not (len(value) in (1, input.shape[-3])):
            raise ValueError(
                f"If value is a sequence, it should have either a single value "
                f"or {input.shape[-3]} (number of input channels)."
            )
    
        x, y, h, w, v = self.get_params(
            image = input,
            scale = self.scale,
            ratio = self.ratio,
            value = value
        )
        return \
            erase(
                image   = input,
                i       = x,
                j       = y,
                h       = h,
                w       = w,
                v       = v,
                inplace = self.inplace,
            ), \
            erase(
                image   = target,
                i       = x,
                j       = y,
                h       = h,
                w       = w,
                v       = v,
                inplace = self.inplace,
            ) if target is not None else None
    

@TRANSFORMS.register(name="solarize")
class Solarize(Transform):
    """
    Solarize an RGB/grayscale image by inverting all pixel values above a
    threshold.

    Args:
        threshold (float): All pixels equal or above this value are inverted.
        p (float | None): Probability of the image being adjusted. Defaults to 
            None means process as normal.
    """
    
    def __init__(
        self,
        threshold: float,
        p        : float | None = None,
        *args, **kwargs
    ):
        super().__init__(p=p, *args, **kwargs)
        self.threshold = threshold
    
    def forward(
        self,
        input : Tensor,
        target: Tensor | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        return solarize(image=input,  threshold=self.threshold), \
               solarize(image=target, threshold=self.threshold) \
                   if target is not None else None
    
    
# H1: - Assertion ------------------------------------------------------------

def is_integer_image(image: Tensor) -> bool:
    """
    Return True if the given image is integer-encoded.
    """
    assert_tensor(image)
    c = get_num_channels(image)
    if c == 1:
        return True
    return False


def is_normalized(image: Tensor) -> Tensor:
    """
    Return True if the given image is normalized.
    """
    assert_tensor(image)
    return abs(torch.max(image)) <= 1.0


def is_one_hot_image(image: Tensor) -> bool:
    """
    Return True if the given image is one-hot encoded.
    """
    assert_tensor(image)
    c = get_num_channels(image)
    if c > 1:
        return True
    return False


def assert_integer_image(image: Tensor):
    if not is_integer_image(image):
        raise ValueError()
    

def assert_normalized(image: Tensor):
    if not is_normalized(image):
        raise ValueError()


def assert_one_hot_image(image: Tensor):
    if not is_one_hot_image(image):
        raise ValueError()
