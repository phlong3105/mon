#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from copy import copy
from copy import deepcopy
from typing import Union

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as F
from multipledispatch import dispatch
from PIL import ExifTags
from torch import Tensor

from one.core import assert_tensor_of_ndim_in_range
from one.core import error_console
from one.core import Int2Or3T
from one.core import Int2T
from one.core import Int3T
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.math import make_divisible
from one.vision.transformation.transform import Transform

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


# MARK: - Functional


def check_image_size(size: Int2Or3T, stride: int = 32) -> int:
    """Verify image size is a multiple of stride and return the new size.
    
    Args:
        size (Int2Or3T):
            Image size of shape [C*, H, W].
        stride (int):
            Stride. Default: `32`.
    
    Returns:
        new_size (int):
            Appropriate size.
    """
    if isinstance(size, (list, tuple)):
        if len(size) == 3:    # [C, H, W]
            size = size[1]
        elif len(size) == 2:  # [H, W]
            size = size[0]
        
    new_size = make_divisible(size, int(stride))  # ceil gs-multiple
    if new_size != size:
        error_console.log(
            "WARNING: image_size %g must be multiple of max stride %g, "
            "updating to %g" % (size, stride, new_size)
        )
    return new_size


def get_exif_size(image: PIL.Image) -> Int2T:
    """Return the exif-corrected PIL size.
    
    Args:
        image (PIL.Image):
            Image.
            
    Returns:
        size (Int2T[H, W]):
            Image size.
    """
    size = image.size  # (width, height)
    try:
        rotation = dict(image._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            size = (size[1], size[0])
        elif rotation == 8:  # rotation 90
            size = (size[1], size[0])
    except:
        pass
    return size[1], size[0]


def get_image_center(image: Tensor) -> Tensor:
    """Get image center as  (x=h/2, y=w/2).
    
    Args:
        image (Tensor[..., C, H, W]):
            Image Tensor.
   
    Returns:
        center (Tensor[2]):
            Image center as (x=h/2, y=w/2).
    """
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2])


def get_image_center4(image: Tensor) -> Tensor:
    """Get image center as (x=h/2, y=w/2, x=h/2, y=w/2).
    
    Args:
        image (Tensor[..., C, H, W]):
            Image.
   
    Returns:
        center (Tensor[4]):
            Image center as (x=h/2, y=w/2, x=h/2, y=w/2).
    """
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    

def get_image_hw(image: Union[Tensor, np.ndarray, PIL.Image]) -> Int2T:
    """Returns the size of an image as [H, W].
    
    Args:
        image (Tensor, np.ndarray, PIL Image):
            Image.
   
    Returns:
        size (Int2T):
            Image size as [H, W].
    """
    if isinstance(image, (Tensor, np.ndarray)):
        if is_channel_first(image):  # [.., C, H, W]
            return [image.shape[-2], image.shape[-1]]
        else:  # [.., H, W, C]
            return [image.shape[-3], image.shape[-2]]
    elif F._is_pil_image(image):
        return list(image.size)
    else:
        raise TypeError(
            f"`image` must be a `Tensor`, `np.ndarray`, or `PIL.Image. "
            f"But got: {type(image)}."
        )
    
    
get_image_size = get_image_hw


def get_image_shape(image: Union[Tensor, np.ndarray, PIL.Image]) -> Int3T:
    """Returns the shape of an image as [H, W, C].

    Args:
        image (Tensor, np.ndarray, PIL Image):
            Image.

    Returns:
        shape (Int3T):
            Image shape as [C, H, W].
    """
    if isinstance(image, (Tensor, np.ndarray)):
        if is_channel_first(image):  # [.., C, H, W]
            return [image.shape[-3], image.shape[-2], image.shape[-1]]
        else:  # [.., H, W, C]
            return [image.shape[-1], image.shape[-3], image.shape[-2]]
    elif F._is_pil_image(image):
        return list(image.size)
    else:
        raise TypeError(
            f"`image` must be a `Tensor`, `np.ndarray`, or `PIL.Image`. "
            f"But got: {type(image)}."
        )


def get_num_channels(image: TensorOrArray) -> int:
    """Get number of channels of the image.
    
    Args:
        image (Tensor, np.ndarray):
            Image.

    Returns:
        num_channels (int):
            Image channels.
    """
    if not isinstance(image, (Tensor, np.ndarray)):
        raise TypeError(
            f"`image` must be a `Tensor` or `np.ndarray`. "
            f"But got: {type(image)}."
        )
    if image.ndim == 4:
        if is_channel_first(image):
            _, c, h, w = list(image.shape)
        else:
            _, h, w, c = list(image.shape)
        return c
    elif image.ndim == 3:
        if is_channel_first(image):
            c, h, w = list(image.shape)
        else:
            h, w, c = list(image.shape)
        return c
    else:
        raise ValueError(
            f"`image.ndim` must be == 3 or 4. But got: {image.ndim}."
        )


def is_channel_first(image: TensorOrArray) -> bool:
    """Return `True` if the image is in channel first format."""
    if image.ndim == 5:
        _, _, s2, s3, s4 = list(image.shape)
        if (s2 < s3) and (s2 < s4):
            return True
        elif (s4 < s2) and (s4 < s3):
            return False
    elif image.ndim == 4:
        _, s1, s2, s3 = list(image.shape)
        if (s1 < s2) and (s1 < s3):
            return True
        elif (s3 < s1) and (s3 < s2):
            return False
    elif image.ndim == 3:
        s0, s1, s2 = list(image.shape)
        if (s0 < s1) and (s0 < s2):
            return True
        elif (s2 < s0) and (s2 < s1):
            return False
    raise ValueError(
        f"`image.ndim` must be == 3, 4, or 5. But got: {image.ndim}."
    )


def is_channel_last(image: TensorOrArray) -> bool:
    """Return `True` if the image is in channel last format."""
    return not is_channel_first(image)


def is_integer_image(image: TensorOrArray) -> bool:
    """Return `True` if the given image is integer-encoded."""
    c = get_num_channels(image)
    if c == 1:
        return True
    return False


def is_normalized(image: TensorOrArray) -> TensorOrArray:
    """Return `True` if the given image is normalized."""
    if isinstance(image, Tensor):
        return abs(torch.max(image)) <= 1.0
    elif isinstance(image, np.ndarray):
        return abs(np.amax(image)) <= 1.0
    else:
        raise TypeError(
            f"`image` must be a `Tensor` or `np.ndarray`. "
            f"But got: {type(image)}."
        )


def is_one_hot_image(image: TensorOrArray) -> bool:
    """Return `True` if the given image is one-hot encoded."""
    c = get_num_channels(image)
    if c > 1:
        return True
    return False


@dispatch(Tensor, keep_dims=bool)
def to_channel_first(image: Tensor, keep_dims: bool = True) -> Tensor:
    """Convert image to channel first format.
    
    Args:
        image (Tensor):
            Image Tensor of arbitrary channel format.
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
    
    Returns:
        image (np.ndarray):
            Image Tensor in channel first format.
    """
    image = copy(image)
    if is_channel_first(image):
        pass
    elif image.ndim == 2:
        image     = image.unsqueeze(0)
    elif image.ndim == 3:
        image     = image.permute(2, 0, 1)
    elif image.ndim == 4:
        image     = image.permute(0, 3, 1, 2)
        keep_dims = True
    elif image.ndim == 5:
        image     = image.permute(0, 1, 4, 2, 3)
        keep_dims = True
    else:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 5. But got: {image.ndim}."
        )
    return image.unsqueeze(0) if not keep_dims else image


@dispatch(np.ndarray, keep_dims=bool)
def to_channel_first(image: np.ndarray, keep_dims: bool = True) -> np.ndarray:
    """Convert image to channel first format.
    
    Args:
        image (np.ndarray):
            Image array of arbitrary channel format.
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
        
    Returns:
        image (np.ndarray):
            Image array in channel first format.
    """
    image = copy(image)
    if is_channel_first(image):
        pass
    elif image.ndim == 2:
        image    = np.expand_dims(image, 0)
    elif image.ndim == 3:
        image    = np.transpose(image, (2, 0, 1))
    elif image.ndim == 4:
        image    = np.transpose(image, (0, 3, 1, 2))
        keep_dims = True
    elif image.ndim == 5:
        image    = np.transpose(image, (0, 1, 4, 2, 3))
        keep_dims = True
    else:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 5. But got: {image.ndim}."
        )
    return np.expand_dims(image, 0) if not keep_dims else image


@dispatch(Tensor, keep_dims=bool)
def to_channel_last(image: Tensor, keep_dims: bool = True) -> Tensor:
    """Convert image to channel last format.
    
    Args:
        image (Tensor):
            Image Tensor of arbitrary channel format.
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
    
    Returns:
        image (np.ndarray):
            Image Tensor in channel last format.
    """
    image       = copy(image)
    input_shape = image.shape
    
    if is_channel_last(image):
        pass
    elif image.ndim == 2:
        pass
    elif image.ndim == 3:
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = image.squeeze()
        else:
            image = image.permute(1, 2, 0)
    elif image.ndim == 4:  # [..., C, H, W] -> [..., H, W, C]
        image = image.permute(0, 2, 3, 1)
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    elif image.ndim == 5:
        image = image.permute(0, 1, 3, 4, 2)
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[2] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 5. But got: {image.ndim}."
        )
    return image
    

@dispatch(np.ndarray, keep_dims=bool)
def to_channel_last(image: np.ndarray, keep_dims: bool = True) -> np.ndarray:
    """Convert image to channel last format.
    
    Args:
        image (np.ndarray):
            Image array of arbitrary channel format.
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
            
    Returns:
        image (np.ndarray):
            Image array in channel last format.
    """
    image       = copy(image)
    input_shape = image.shape
    
    if is_channel_last(image):
        pass
    elif image.ndim == 2:
        pass
    elif image.ndim == 3:
        if input_shape[0] == 1:
            # Grayscale for proper plt.imshow needs to be [H, W]
            image = image.squeeze()
        else:
            image = np.transpose(image, (1, 2, 0))
    elif image.ndim == 4:
        image = np.transpose(image, (0, 2, 3, 1))
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[1] == 1:
            image = image.squeeze(-1)
    elif image.ndim == 5:
        image = np.transpose(image, (0, 1, 3, 4, 2))
        if input_shape[0] == 1 and not keep_dims:
            image = image.squeeze(0)
        if input_shape[2] == 1:
            image = image.squeeze(-1)
    else:
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 5. But got: {image.ndim}."
        )
    return image


def to_image(
    input      : Tensor,
    keep_dims  : bool = True,
    denormalize: bool = False
) -> np.ndarray:
    """Converts a PyTorch Tensor to a numpy image. In case the image is in the
    GPU, it will be copied back to CPU.

    Args:
        input (Tensor):
            Image arbitrary shape.
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
        denormalize (bool):
            If `True`, converts the image in the range [0.0, 1.0] to the range
            [0, 255]. Default: `False`.
        
    Returns:
        image (np.ndarray):
            Image of the form [H, W], [H, W, C] or [..., H, W, C].
    """
    from one.vision.transformation import denormalize_naive
    
    assert_tensor_of_ndim_in_range(input, 2, 4)
   
    image = input.cpu().detach().numpy()
    
    # NOTE: Channel last format
    image = to_channel_last(image, keep_dims=keep_dims)
    
    # NOTE: Denormalize
    if denormalize:
        image = denormalize_naive(image)
        
    return image.astype(np.uint8)


def to_pil_image(image: TensorOrArray) -> PIL.Image:
    """Convert image from `np.ndarray` or `Tensor` to PIL image."""
    if torch.is_tensor(image):
        # Equivalent to: `np_image = image.numpy()` but more efficient
        return F.pil_to_tensor(image)
    elif isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image.astype(np.uint8), "RGB")
    raise TypeError(f"Do not support {type(image)}.")


def to_tensor(
    image    : Union[Tensor, np.ndarray, PIL.Image],
    keep_dims: bool = True,
    normalize: bool = False,
) -> Tensor:
    """Convert a `PIL Image` or `np.ndarray` image to a 4d tensor.
    
    Args:
        image (Tensor, np.ndarray, PIL.Image):
            Image array or PIL.Image in [H, W, C], [H, W] or [..., H, W, C].
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
        normalize (bool):
            If `True`, converts the tensor in the range [0, 255] to the range
            [0.0, 1.0]. Default: `False`.
    
    Returns:
        img (Tensor):
            Image Tensor.
    """
    from one.vision.transformation import normalize_naive

    if not (F._is_numpy(image) or torch.is_tensor(image)
            or F._is_pil_image(image)):
        raise TypeError(
            f"`image` must be a `Tensor`, `np.ndarray`, or `PIL.Image. "
            f"But got: {type(image)}."
        )
    
    if ((F._is_numpy(image) or torch.is_tensor(image))
        and not (2 <= image.ndim <= 4)):
        raise ValueError(
            f"Require 2 <= `image.ndim` <= 4. But got: {image.ndim}."
        )

    # img = image
    img = deepcopy(image)
    
    # NOTE: Handle PIL Image
    if F._is_pil_image(img):
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = np.array(img, mode_to_nptype.get(img.mode, np.uint8), copy=True)
        if image.mode == "1":
            img = 255 * img
    
    # NOTE: Handle numpy array
    if F._is_numpy(img):
        img = torch.from_numpy(img).contiguous()
    
    # NOTE: Channel first format
    img = to_channel_first(img, keep_dims=keep_dims)
   
    # NOTE: Normalize
    if normalize:
        img = normalize_naive(img)
    
    # NOTE: Convert type
    if isinstance(img, torch.ByteTensor):
        return img.to(dtype=torch.get_default_dtype())
    
    # NOTE: Place in memory
    img = img.contiguous()
    return img


# MARK: - Module

@TRANSFORMS.register(name="to_image")
class ToImage(Transform):
    """Converts a PyTorch Tensor to a numpy image. In case the image is in the
    GPU, it will be copied back to CPU.

    Args:
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
        denormalize (bool):
            If `True`, converts the image in the range [0.0, 1.0] to the range
            [0, 255]. Default: `False`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        keep_dims  : bool = True,
        denormalize: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keep_dims   = keep_dims
        self.denormalize = denormalize
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[np.ndrray, Union[np.ndrray, None]]:
        return \
            to_image(
                input       = input,
                keep_dims   = self.keep_dims,
                denormalize = self.denormalize
            ), \
            to_image(
                input       = target,
                keep_dims   = self.keep_dims,
                denormalize = self.denormalize
            ) if target is not None else None
    

@TRANSFORMS.register(name="to_tensor")
class ToTensor(Transform):
    """Convert a `PIL Image` or `np.ndarray` image to a 4D tensor.
    
    Args:
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
        normalize (bool):
            If `True`, converts the tensor in the range [0, 255] to the range
            [0.0, 1.0]. Default: `False`.
    """
    
    # MARK: Magic Functions

    def __init__(
        self,
        keep_dims: bool = False,
        normalize: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.keep_dims = keep_dims
        self.normalize = normalize
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input : Tensor,
        target: Union[Tensor, None] = None,
        *args, **kwargs
    ) -> tuple[Tensor, Union[Tensor, None]]:
        return \
            to_tensor(
                image     = input,
                keep_dims = self.keep_dims,
                normalize = self.normalize
            ), \
            to_tensor(
                image     = input,
                keep_dims = self.keep_dims,
                normalize = self.normalize
            ) if target is not None else None
