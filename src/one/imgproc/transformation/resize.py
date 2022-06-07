#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://mathworld.wolfram.com/topics/GeometricTransformations.html

List of operation:
    - Cantellation
    - Central Dilation
    - Collineation
    - Dilation
    - Elation
    - Elliptic Rotation
    - Expansion
    - Geometric Correlation
    - Geometric Homology
    - Harmonic Homology
    - Homography
    - Perspective Collineation
    - Polarity
    - Projective Collineation
    - Projective Correlation
    - Projectivity
    - Stretch
    - Twirl
    - Unimodular Transformation
"""

from __future__ import annotations

from typing import Any
from typing import Optional
from typing import Union

import cv2
import numpy as np
import PIL.Image
from torch import nn
from torch import Tensor
from torchvision.transforms import functional_pil as F_pil
from torchvision.transforms import functional_tensor as F_t

from one.core import Color
from one.core import error_console
from one.core import get_image_hw
from one.core import Int2Or3T
from one.core import Int3T
from one.core import InterpolationMode
from one.core import to_channel_last
from one.core import to_size
from one.core import TRANSFORMS
from one.imgproc.utils import batch_image_processing

__all__ = [
    "letterbox_resize",
    "resize_numpy_image",
    "resize_pil_image",
    "resize_tensor_image",
    "resize",
    "Resize",
]


# MARK: - Functional

def _cast_squeeze_in(image: np.ndarray, req_dtypes: list[Any]) -> tuple[np.ndarray, bool, bool, Any]:
    need_expand = False
    # make image HWC
    if image.ndim == 4:
        image       = np.squeeze(image, axis=0)
        need_expand = True
    image = to_channel_last(image)

    out_dtype = image.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        image     = image.astype(req_dtype)
    return image, need_cast, need_expand, out_dtype


def _cast_squeeze_out(image: np.ndarray, need_cast: bool, need_expand: bool, out_dtype: Any) -> np.ndarray:
    if need_expand:
        image = np.expand_dims(image, axis=0)

    if need_cast:
        if out_dtype in (np.uint8, np.int8, np.int16, np.int32, np.int64):
            # it is better to round before cast
            image = np.round(image)
        image = image.astype(out_dtype)

    return image


def letterbox_resize(
    image     : np.ndarray,
    size      : Optional[Int2Or3T] = 768,
    stride    : int                = 32,
    color     : Color              = (114, 114, 114),
    auto      : bool               = True,
    scale_fill: bool               = False,
    scale_up  : bool               = True
):
    """Resize image to a `stride`-pixel-multiple rectangle.
    
    For YOLOv5, stride = 32.
    For Scaled-YOLOv4, stride = 128
    
    References:
        https://github.com/ultralytics/yolov3/issues/232
        
    Args:
        image:
        size:
        stride:
        color:
        auto:
        scale_fill:
        scale_up:

    Returns:

    """
    # old_size = image.old_size[:2]  # current old_size [height, width]
    old_size = get_image_hw(image)
    
    if size is None:
        return image, None, None, None
    size = to_size(size)
    
    # Scale ratio (new / old)
    r = min(size[0] / old_size[0], size[1] / old_size[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio     = r, r  # width, height ratios
    new_unpad = int(round(old_size[1] * r)), int(round(old_size[0] * r))
    dw, dh    = size[1] - new_unpad[0], size[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scale_fill:  # stretch
        dw, dh    = 0.0, 0.0
        new_unpad = (size[1], size[0])
        ratio     = size[1] / old_size[1], size[0] / old_size[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if old_size[::-1] != new_unpad:  # resize
        image = cv2.resize(src=image, dsize=new_unpad, interpolation=cv2.INTER_LINEAR)
        
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return image, ratio, (dw, dh)


@batch_image_processing
def resize_numpy_image(
    image        : np.ndarray,
    size         : Optional[Int2Or3T] = None,
    interpolation: InterpolationMode  = InterpolationMode.LINEAR,
    max_size     : Optional[int]      = None,
    antialias    : Optional[bool]     = None
) -> np.ndarray:
    """Resize a numpy image. Adapted from:
    `torchvision.transforms.functional_tensor.resize()`
    
    Args:
        image (np.ndarray[C, H, W]):
            Image to be resized.
        size (Int2Or3T[H, W, C*], optional):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode):
            Interpolation method.
        max_size (int, optional):
        
        antialias (bool, optional):

    Returns:
        resize (np.ndarray[H, W, C]):
            Resized image.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")
    cv_interpolation = InterpolationMode.cv_modes_mapping[interpolation]
    if cv_interpolation not in list(InterpolationMode.cv_modes_mapping.values()):
        raise ValueError(
            "This interpolation mode is unsupported with np.ndarray input"
        )

    if size is None:
        return image
    size = to_size(size)[::-1]  # W, H
    
    if antialias is None:
        antialias = False
    if antialias and cv_interpolation not in [cv2.INTER_LINEAR, cv2.INTER_CUBIC]:
        raise ValueError("Antialias option is supported for linear and cubic "
                         "interpolation modes only")

    w, h = get_image_hw(image)
    # Specified size only for the smallest edge
    if isinstance(size, int) or len(size) == 1:
        short, long         = (w, h) if w <= h else (h, w)
        requested_new_short = size if isinstance(size, int) else size[0]

        if short == requested_new_short:
            return image

        new_short = requested_new_short,
        new_long  = int(requested_new_short * long / short)
        
        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the "
                    f"requested size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short = int(max_size * new_short / new_long),
                new_long  = max_size
        
        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        
    else:  # specified both h and w
        new_w, new_h = size[0], size[1]
    
    image, need_cast, need_expand, out_dtype = _cast_squeeze_in(
        image, [np.float32, np.float64]
    )
    
    image = cv2.resize(image, dsize=(new_w, new_h), interpolation=cv_interpolation)
    
    if cv_interpolation == cv2.INTER_CUBIC and out_dtype == np.uint8:
        image = np.clip(image, 0, 255)
    
    image = _cast_squeeze_out(image, need_cast=need_cast, need_expand=need_expand, out_dtype=out_dtype)
    
    return image


def resize_pil_image(
    image        : PIL.Image.Image,
    size         : Optional[Int2Or3T] = None,
    interpolation: InterpolationMode  = InterpolationMode.BILINEAR,
    max_size     : Optional[int]      = None,
    antialias    : Optional[bool]     = None
) -> PIL.Image:
    """Resize a pil image. Adapted from:
    `torchvision.transforms.functional_pil.resize()`
    
    Args:
        image (PIL.Image.Image[H, W, C]):
            Image.
        size (Int2Or3T[H, W, C*], optional):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode):
            Interpolation method.
        max_size (int, optional):
        
        antialias (bool, optional):

    Returns:
        resize (PIL.Image.Image[H, W, C]):
            Resized image.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")
    
    if antialias is not None and not antialias:
        error_console.log(
            "Anti-alias option is always applied for PIL Image input. "
            "Argument antialias is ignored."
        )
    pil_interpolation = InterpolationMode.pil_modes_mapping()[interpolation]

    if size is None:
        return image
    size = to_size(size)  # H, W
    
    return F_pil.resize(
        image         = image,
        size          = size[::-1],  # W, H
        interpolation = pil_interpolation,
        max_size      = max_size,
        antialias     = antialias
    )


def resize_tensor_image(
    image        : Tensor,
    size         : Optional[Int2Or3T] = None,
    interpolation: InterpolationMode  = InterpolationMode.BILINEAR,
    max_size     : Optional[int]      = None,
    antialias    : Optional[bool]     = None
) -> Tensor:
    """Resize a tensor image. Adapted from:
    `torchvision.transforms.functional_tensor.resize()`
    
    Args:
        image (Tensor[H, W, C]):
            Image.
        size (Int2Or3T[H, W, C*], optional):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode):
            Interpolation method.
        max_size (int, optional):
        
        antialias (bool, optional):

    Returns:
        resize (Tensor[H, W, C]):
            Resized image.
    """
    if image.ndim != 3:
        raise ValueError("Currently only support image with `ndim == 3`.")
    if isinstance(interpolation, int):
        interpolation = InterpolationMode.from_value(interpolation)
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError("Argument interpolation should be a InterpolationMode")

    if size is None:
        return image
    size = to_size(size)  # H, W

    return F_t.resize(
        img           = image,
        size          = size,  # H, W
        interpolation = interpolation.value,
        max_size      = max_size,
        antialias     = antialias
    )


def resize(
    image        : Union[Tensor, np.ndarray, PIL.Image.Image],
    size         : Optional[Int2Or3T] = None,
    interpolation: InterpolationMode  = InterpolationMode.LINEAR,
    max_size     : Optional[int]      = None,
    antialias    : Optional[bool]     = None
) -> Union[Tensor, np.ndarray, PIL.Image.Image]:
    """Resize an image. Adapted from:
    `torchvision.transforms.functional.resize()`
    
    Args:
        image (Tensor, np.ndarray, PIL.Image.Image):
            Image of shape [H, W, C].
        size (Int2Or3T[H, W, C*], optional):
            Desired output size. Default: `None`.
        interpolation (InterpolationMode):
            Interpolation method.
        max_size (int, optional):
        
        antialias (bool, optional):

    Returns:
        resize (Tensor, np.ndarray, PIL.Image.Image):
            Resized image of shape [H, W, C].
    """
    if size is None:
        return image
    if isinstance(image, Tensor):
        if interpolation is InterpolationMode.LINEAR:
            interpolation = InterpolationMode.BILINEAR
        return resize_tensor_image(image, size, interpolation, max_size, antialias)
    elif isinstance(image, np.ndarray):
        if interpolation is InterpolationMode.BILINEAR:
            interpolation = InterpolationMode.LINEAR
        return resize_numpy_image(image, size, interpolation, max_size, antialias)
    else:
        return resize_pil_image(image, size, interpolation, max_size, antialias)


# MARK: - Modules

@TRANSFORMS.register(name="resize")
class Resize(nn.Module):
    r"""Resize the input image to the given size.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    
    .. warning::
        Output image might be different depending on its type: when
        downsampling, the interpolation of PIL images and tensors is slightly
        different, because PIL applies antialiasing. This may lead to
        significant differences in the performance of a network.
        Therefore, it is preferable to train and serve a model with the same
        input types. See also below the `antialias` parameter, which can help
        making the output of PIL images and tensors closer.

    Args:
        size (Int3T):
            Desired output size. If size is a sequence like [H, W], the output
            size will be matched to this. If size is an int, the smaller edge
            of the image will be matched to this number maintaining the aspect
            ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            .. note::
                In torchscript mode size as single int is not supported, use a
                sequence of length 1: `[size, ]`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
        max_size (int, optional):
            Maximum allowed for the longer edge of the resized image: if
            the longer edge of the image is greater than `max_size` after being
            resized according to `size`, then the image is resized again so
            that the longer edge is equal to `max_size`. As a result, `size`
            might be overruled, i.e the smaller edge may be shorter than `size`.
            This is only supported if `size` is an int (or a sequence of length
            1 in torchscript mode).
        antialias (bool, optional):
            Antialias flag. If `img` is PIL Image, the flag is ignored and
            anti-alias is always used. If `img` is Tensor, the flag is False by
            default and can be set to True for `InterpolationMode.BILINEAR`
            only mode. This can help making the output for PIL images and
            tensors closer.

            .. warning::
                There is no autodiff support for `antialias=True` option with
                input `img` as Tensor.
    """
    
    def __init__(
        self,
        size         : Optional[Int3T],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        max_size     : Optional[int]     = None,
        antialias    : Optional[bool]    = None
    ):
        super().__init__()
        self.size          = size
        self.interpolation = interpolation
        self.max_size      = max_size
        self.antialias     = antialias
    
    def forward(
        self, image: Union[Tensor, np.ndarray, PIL.Image.Image]
    ) -> Union[Tensor, np.ndarray, PIL.Image.Image]:
        """
        
        Args:
            image (Tensor, np.ndarray, PIL.Image.Image):
                Image to be cropped. (0,0) denotes the top left corner of the
                image.
        
        Returns:
            image (Tensor, np.ndarray, PIL.Image.Image):
                Resized image.
        """
        return resize(
            image, self.size, self.interpolation, self.max_size, self.antialias,
        )
