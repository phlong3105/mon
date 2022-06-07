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

import random

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torchvision.transforms.functional import center_crop
from torchvision.transforms.functional import crop
from torchvision.transforms.functional import resized_crop

from one.core import Int2T
from one.core import InterpolationMode
from one.core import is_channel_last
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.utils import batch_image_processing

__all__ = [
    "center_crop",
    "crop",
    "crop_zero_region",
    "lowhighres_images_random_crop",
    "resized_crop",
    "CenterCrop",
    "Crop",
    "LowHighResImagesRandomCrop",
    "ResizedCrop",
]


# MARK: - Functional

@batch_image_processing
def crop_zero_region(image: TensorOrArray) -> TensorOrArray:
    """Crop the zero region around the non-zero region in image.
    
    Args:
        image (TensorOrArray[C, H, W]):
            Image to with zeros background.
            
    Returns:
        image (TensorOrArray[C, H, W]):
            Cropped image.
    """
    if isinstance(image, Tensor):
        any   = torch.any
        where = torch.where
    elif isinstance(image, np.ndarray):
        any   = np.any
        where = np.where
    
    if is_channel_last(image):
        cols       = any(image, axis=0)
        rows       = any(image, axis=1)
        xmin, xmax = where(cols)[0][[0, -1]]
        ymin, ymax = where(rows)[0][[0, -1]]
        image      = image[ymin:ymax + 1, xmin:xmax + 1]
    else:
        cols       = any(image, axis=1)
        rows       = any(image, axis=2)
        xmin, xmax = where(cols)[0][[0, -1]]
        ymin, ymax = where(rows)[0][[0, -1]]
        image      = image[:, ymin:ymax + 1, xmin:xmax + 1]
    return image


def lowhighres_images_random_crop(
    lowres : Tensor,
    highres: Tensor,
    size   : int,
    scale  : int
) -> tuple[Tensor, Tensor]:
    """Random cropping a pair of low and high resolution images."""
    lowres_left    = random.randint(0, lowres.shape[2] - size)
    lowres_right   = lowres_left   + size
    lowres_top     = random.randint(0, lowres.shape[1] - size)
    lowres_bottom  = lowres_top    + size
    highres_left   = lowres_left   * scale
    highres_right  = lowres_right  * scale
    highres_top    = lowres_top    * scale
    highres_bottom = lowres_bottom * scale
    lowres         = lowres[ :,  lowres_top:lowres_bottom,   lowres_left:lowres_right]
    highres        = highres[:, highres_top:highres_bottom, highres_left:highres_right]
    return lowres, highres


# MARK: - Modules

@TRANSFORMS.register(name="center_crop")
class CenterCrop(nn.Module):
    """Crops the given image at the center.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded
    with 0 and then cropped.

    Args:
        output_size (Int2T):
            [height, width] of the crop box. If int or sequence with single int,
            it is used for both directions.
    """
    
    def __init__(self, output_size: Int2T):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (PIL Image or Tensor):
                Image to be cropped.
        
        Returns:
            (PIL Image or Tensor):
                Cropped image.
        """
        return center_crop(image, self.output_size)
    
    
@TRANSFORMS.register(name="crop")
class Crop(nn.Module):
    """Crop the given image at specified location and output size.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded
    with 0 and then cropped.

    Args:
        top (int):
            Vertical component of the top left corner of the crop box.
        left (int):
            Horizontal component of the top left corner of the crop box.
        height (int):
            Height of the crop box.
        width (int):
            Width of the crop box.
    """
    
    def __init__(self, top: int, left: int, height: int, width: int):
        super().__init__()
        self.top    = top
        self.left   = left
        self.height = height
        self.width  = width
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (PIL Image or Tensor):
                Image to be cropped. (0,0) denotes the top left corner of the
                image.
        
        Returns:
            (PIL Image or Tensor):
                Cropped image.
        """
        return crop(image, self.top, self.left, self.height, self.width)


@TRANSFORMS.register(name="lowhighres_images_random_crop")
class LowHighResImagesRandomCrop(nn.Module):
    """Random cropping a pair of low and high resolution images.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    
    Args:
        size (int):
            The patch size.
        scale (int):
            Scale factor.
    """

    def __init__(self, size: int, scale: int):
        super().__init__()
        self.size  = size
        self.scale = scale
    
    def forward(self, low_res: Tensor, high_res: Tensor) -> tuple[Tensor, Tensor]:
        """
        
        Args:
            low_res (PIL Image or Tensor):
                Low resolution image.
            high_res (PIL Image or Tensor):
                High resolution image.
            
        Returns:
            Cropped images.
        """
        return lowhighres_images_random_crop(
            lowres=low_res, highres=high_res, size=self.size, scale=self.scale
        )


@TRANSFORMS.register(name="resized_crop")
class ResizedCrop(nn.Module):
    """Crop the given image and resize it to desired size.
    If the image is Tensor, it is expected to have [..., H, W] shape,
    where ... means an arbitrary number of leading dimensions.
    
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.

    Args:
        top (int):
            Vertical component of the top left corner of the crop box.
        left (int):
            Horizontal component of the top left corner of the crop box.
        height (int):
            Height of the crop box.
        width (int):
            Width of the crop box.
        size (list[int]):
            Desired output size. Same semantics as `resize`.
        interpolation (InterpolationMode):
            Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`.
            Default is `InterpolationMode.BILINEAR`. If input is Tensor, only
            `InterpolationMode.NEAREST`, `InterpolationMode.BILINEAR` and
            `InterpolationMode.BICUBIC` are supported.
            For backward compatibility integer values (e.g. `PIL.Image.NEAREST`)
            are still acceptable.
    """
    
    def __init__(
        self,
        top          : int,
        left         : int,
        height       : int,
        width        : int,
        size         : list[int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR
    ):
        super().__init__()
        self.top           = top
        self.left          = left
        self.height        = height
        self.width         = width
        self.size          = size
        self.interpolation = interpolation
    
    def forward(self, image: Tensor) -> Tensor:
        """
        
        Args:
            image (PIL Image or Tensor):
                Image to be cropped. (0,0) denotes the top left corner of the
                image.
        
        Returns:
            (PIL Image or Tensor):
                Cropped image.
        """
        return resized_crop(
            image, self.top, self.left, self.height, self.width, self.size,
            self.interpolation
        )
