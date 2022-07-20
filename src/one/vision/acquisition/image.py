#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import multiprocessing
import os
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import PIL
import torch
import torchvision
import torchvision.transforms.functional as F
from joblib import delayed
from joblib import Parallel
from multipledispatch import dispatch
from PIL import ExifTags
from PIL import Image
from torch import Tensor

from one.core import assert_numpy_of_ndim
from one.core import assert_numpy_of_ndim_in_range
from one.core import assert_tensor
from one.core import assert_tensor_of_atleast_ndim
from one.core import assert_tensor_of_ndim
from one.core import assert_tensor_of_ndim_in_range
from one.core import create_dirs
from one.core import error_console
from one.core import Ints
from one.core import is_image_file
from one.core import TensorOrArray
from one.core import Tensors
from one.core import Transform
from one.core import TRANSFORMS
from one.core import VisionBackend
from one.core import VisionBackend_
from one.math import make_divisible

# Get orientation exif tag

for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


# MARK: - Functional

def check_image_size(size: Ints, stride: int = 32) -> int:
    """Verify image size is a multiple of stride and return the new size.
    
    Args:
        size (Ints):
            Image size of shape [C, H, W].
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


def get_exif_size(image: PIL.Image) -> Ints:
    """Return the exif-corrected PIL size.
    
    Args:
        image (PIL.Image):
            Image.
            
    Returns:
        size (Ints[H, W]):
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
    assert_tensor(image)
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
    assert_tensor(image)
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    

def get_image_hw(image: Tensor) -> Ints:
    """Returns the size of an image as [H, W].
    
    Args:
        image (Tensor):
            Image.
   
    Returns:
        size (Ints):
            Image size as [H, W].
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if is_channel_first(image):  # [.., C, H, W]
        return [image.shape[-2], image.shape[-1]]
    else:  # [.., H, W, C]
        return [image.shape[-3], image.shape[-2]]
    
    
get_image_size = get_image_hw


def get_image_shape(image: Tensor) -> Ints:
    """Returns the shape of an image as [H, W, C].

    Args:
        image (Tensor):
            Image.

    Returns:
        shape (Ints):
            Image shape as [C, H, W].
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if is_channel_first(image):  # [.., C, H, W]
        return [image.shape[-3], image.shape[-2], image.shape[-1]]
    else:  # [.., H, W, C]
        return [image.shape[-1], image.shape[-3], image.shape[-2]]


def get_num_channels(image: Tensor) -> int:
    """Get number of channels of the image.
    
    Args:
        image (Tensor):
            Image.

    Returns:
        num_channels (int):
            Image channels.
    """
    assert_tensor_of_ndim_in_range(image, 3, 4)
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
    return 0


def is_channel_first(image: Tensor) -> bool:
    """Return `True` if the image is in channel first format."""
    assert_tensor_of_ndim_in_range(image, 3, 5)
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
    return False


def is_channel_last(image: Tensor) -> bool:
    """Return `True` if the image is in channel last format."""
    return not is_channel_first(image)


def read_image_cv(path: str) -> Tensor:
    """Read image using OpenCV and return a Tensor.
    
    Args:
        path (str):
            Image file.
    
    Returns:
        image (Tensor[1, C, H, W]):
            Image Tensor.
    """
    image = cv2.imread(path)             # BGR
    image = image[:, :, ::-1]            # BGR -> RGB
    # image = np.ascontiguousarray(image)  # Numpy
    image = to_tensor(image=image, keep_dims=False)
    return image


'''
def read_image_libvips(path: str) -> np.ndarray:
    """Read image using libvips."""
    image   = pyvips.Image.new_from_file(path, access="sequential")
    mem_img = image.write_to_memory()
    image   = np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width, 3)
    return image
'''


def read_image_pil(path: str) -> Tensor:
    """Read image using PIL and return a Tensor.
    
    Args:
        path (str):
            Image file.
    
    Returns:
        image (Tensor[1, C, H, W]):
            Image Tensor.
    """
    image = Image.open(path)                         # PIL Image
    image = to_tensor(image=image, keep_dims=False)  # Tensor[C, H, W]
    return image


def read_image(path: str, backend: VisionBackend_ = VisionBackend.CV) -> Tensor:
    """Read image with the corresponding backend.
    
    Args:
        path (str):
            Image file.
        backend (VisionBackend_):
            Vision backend used to read images. Default: `VisionBackend.CV`.
            
    Returns:
        image (Tensor[1, C, H, W]):
            Image Tensor.
    """
    backend = VisionBackend.from_value(backend)
    if backend == VisionBackend.CV:
        return read_image_cv(path)
    elif backend == VisionBackend.LIBVIPS:
        # return read_image_libvips(path)
        pass
    elif backend == VisionBackend.PIL:
        return read_image_pil(path)
    else:
        raise ValueError(f"Do not supported {backend}.")
    

def to_channel_first(
    image: Tensor, keep_dims: bool = True, inplace: bool = False
) -> Tensor:
    """Convert image to channel first format.
    
    Args:
        image (Tensor):
            Image Tensor of arbitrary channel format.
        keep_dims (bool):
            If `False` unsqueeze the image to match the shape [..., C, H, W].
            Else, keep the original dimension. Default: `True`.
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.
            
    Returns:
        image (Tensor):
            Image Tensor in channel first format.
    """
    assert_tensor_of_ndim_in_range(image, 2, 5)
    if not inplace:
        image = image.clone()
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
    return image.unsqueeze(0) if not keep_dims else image


def to_channel_last(
    image: Tensor, keep_dims: bool = True, inplace: bool = False
) -> Tensor:
    """Convert image to channel last format.
    
    Args:
        image (Tensor):
            Image Tensor of arbitrary channel format.
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.
            
    Returns:
        image (np.ndarray):
            Image Tensor in channel last format.
    """
    assert_tensor_of_ndim_in_range(image, 2, 5)
    if not inplace:
        image = image.clone()

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
    return image


def to_image(
    image      : Tensor,
    keep_dims  : bool = True,
    denormalize: bool = False,
    inplace    : bool = False,
) -> np.ndarray:
    """Converts a PyTorch Tensor to a numpy image. In case the image is in the
    GPU, it will be copied back to CPU.

    Args:
        image (Tensor):
            Image of arbitrary shape.
        keep_dims (bool):
            If `False` squeeze the input image to match the shape [H, W, C] or
            [H, W]. Else, keep the original dimension. Default: `True`.
        denormalize (bool):
            If `True`, converts the image in the range [0.0, 1.0] to the range
            [0, 255]. Default: `False`.
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.
            
    Returns:
        image (np.ndarray):
            Image of the form [H, W], [H, W, C] or [..., H, W, C].
    """
    from one.vision.transformation import denormalize_naive
    
    assert_tensor_of_ndim_in_range(image, 2, 4)
    
    if not inplace:
        image = image.clone()
    image: np.ndarray = image.cpu().detach().numpy()
    
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
    inplace  : bool = False,
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
        inplace (bool):
            If `True`, make this operation inplace. Default: `False`.
            
    Returns:
        img (Tensor):
            Image Tensor.
    """
    from one.vision.transformation.intensity import normalize_naive

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

    # NOTE: Handle PIL Image
    if F._is_pil_image(image):
        mode = image.mode
        if not inplace:
            image = deepcopy(image)
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        image = np.array(image, mode_to_nptype.get(image.mode, np.uint8), copy=True)
        if mode == "1":
            image = 255 * image
    
    # NOTE: Handle numpy array
    if F._is_numpy(image):
        if not inplace:
            image = deepcopy(image)
        image = torch.from_numpy(image).contiguous()
    
    # NOTE: Channel first format
    image = to_channel_first(image, keep_dims=keep_dims)
   
    # NOTE: Normalize
    if normalize:
        image = normalize_naive(image)
    
    # NOTE: Convert type
    if isinstance(image, torch.ByteTensor):
        return image.to(dtype=torch.get_default_dtype())
    
    # NOTE: Place in memory
    image = image.contiguous()
    return image


@dispatch(np.ndarray, str, str, str, str)
def write_image(
    image    : np.ndarray,
    dir      : str,
    name     : str,
    prefix   : str = "",
    extension: str = ".png"
):
    """Save the image using `PIL`.

    Args:
        image (np.ndarray):
            A single image.
        dir (str):
            Saving directory.
        name (str):
            Name of the image file.
        prefix (str):
            Filename prefix. Default: ``.
        extension (str):
            Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
            Default: `png`.
    """
    from one.vision.transformation import denormalize_naive
    assert_numpy_of_ndim_in_range(image, 2, 3)
    
    # NOTE: Unnormalize
    image = denormalize_naive(image)
    
    # NOTE: Convert to channel first
    if is_channel_first(image):
        image = to_channel_last(image)
    
    # NOTE: Convert to PIL image
    if not Image.isImageType(t=image):
        image = Image.fromarray(image.astype(np.uint8))
    
    # NOTE: Write image
    if create_dirs(paths=[dir]) == 0:
        base, ext = os.path.splitext(name)
        if ext:
            extension = ext
        if "." not in extension:
            extension = f".{extension}"
        if prefix in ["", None]:
            filepath = os.path.join(dir, f"{base}{extension}")
        else:
            filepath = os.path.join(dir, f"{prefix}_{base}{extension}")
        image.save(filepath)


@dispatch(Tensor, str, str, str, str)
def write_image(
    image    : Tensor,
    dir      : str,
    name     : str,
    prefix   : str = "",
    extension: str = ".png"
):
    """Save the image using `torchvision`.

    Args:
        image (Tensor):
            A single image.
        dir (str):
            Saving directory.
        name (str):
            Name of the image file.
        prefix (str):
            Filename prefix. Default: ``.
        extension (str):
            Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
            Default: `.png`.
    """
    from one.vision.transformation import denormalize_naive
    assert_tensor_of_ndim_in_range(image, 2, 3)
    
    # NOTE: Convert image
    
    image = denormalize_naive(image)
    image = to_channel_last(image)
    
    # NOTE: Write image
    if create_dirs(paths=[dir]) == 0:
        base, ext = os.path.splitext(name)
        if ext:
            extension = ext
        if "." not in extension:
            extension = f".{extension}"
        if prefix in ["", None]:
            filepath = os.path.join(dir, f"{base}{extension}")
        else:
            filepath = os.path.join(dir, f"{prefix}_{base}{extension}")
        if extension in [".jpg", ".jpeg"]:
            torchvision.io.image.write_jpeg(input=image, filename=filepath)
        elif extension in [".png"]:
            torchvision.io.image.write_png(input=image, filename=filepath)


@dispatch(np.ndarray, str, str, str)
def write_images(
    images   : np.ndarray,
    dir      : str,
    name     : str,
    extension: str = ".png"
):
    """Save multiple images using `PIL`.

    Args:
        images (np.ndarray):
            A batch of images.
        dir (str):
            Saving directory.
        name (str):
            Name of the image file.
        extension (str):
            Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
            Default: `.png`.
    """
    assert_numpy_of_ndim(images, 4)
    num_jobs = multiprocessing.cpu_count()
    Parallel(n_jobs=num_jobs)(
        delayed(write_image)(image, dir, name, f"{index}", extension)
        for index, image in enumerate(images)
    )


@dispatch(Tensor, str, str, str)
def write_images(
    images   : Tensor,
    dir      : str,
    name     : str,
    extension: str = ".png"
):
    """Save multiple images using `torchvision`.

    Args:
        images (Tensor):
            A image of image.
        dir (str):
            Saving directory.
        name (str):
            Name of the image file.
        extension (str):
            Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
            Default: `.png`.
    """
    assert_tensor_of_ndim(images, 4)
    num_jobs = multiprocessing.cpu_count()
    Parallel(n_jobs=num_jobs)(
        delayed(write_image)(image, dir, name, f"{index}", extension)
        for index, image in enumerate(images)
    )


@dispatch(list, str, str, str)
def write_images(
    images   : list,
    dir      : str,
    name     : str,
    extension: str = ".png"
):
    """Save multiple images.

    Args:
        images (list):
            A list of images.
        dir (str):
            Saving directory.
        name (str):
            Name of the image file.
        extension (str):
            Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
            Default: `.png`.
    """
    if (isinstance(images, list) and
        all(isinstance(image, np.ndarray) for image in images)):
        cat_image = np.concatenate([images], axis=0)
        write_images(cat_image, dir, name, extension)
    elif (isinstance(images, list) and
          all(torch.is_tensor(image) for image in images)):
        cat_image = torch.stack(images)
        write_images(cat_image, dir, name, extension)
    else:
        raise TypeError(f"Do not support {type(images)}.")


@dispatch(dict, str, str, str)
def write_images(
    images   : dict,
    dir      : str,
    name     : str,
    extension: str = ".png"
):
    """Save multiple images.

    Args:
        images (dict):
            A list of images.
        dir (str):
            Saving directory.
        name (str):
            Name of the image file.
        extension (str):
            Image file extension. One of: [`.jpg`, `.jpeg`, `.png`].
            Default: `.png`.
    """
    if (isinstance(images, dict) and
        all(isinstance(image, np.ndarray) for _, image in images.items())):
        cat_image = np.concatenate(
            [image for key, image in images.items()], axis=0
        )
        write_images(cat_image, dir, name, extension)
    elif (isinstance(images, dict) and
          all(torch.is_tensor(image) for _, image in images)):
        values    = list(tuple(images.values()))
        cat_image = torch.stack(values)
        write_images(cat_image, dir, name, extension)
    else:
        raise TypeError


# MARK: - Modules


class ImageLoader:
    """Image Loader retrieves and loads image(s) from a filepath, a pathname
    pattern, or directory.

    Args:
        data (str):
            Data source. Can be a path to an image file or a directory.
            It can be a pathname pattern to images.
        batch_size (int):
            Number of samples in one forward & backward pass. Default: `1`.
        backend (VisionBackend_):
            Vision backend used to read images. Default: `VisionBackend.CV`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        data      : str,
        batch_size: int            = 1,
        backend   : VisionBackend_ = VisionBackend.CV
    ):
        super().__init__()
        self.data       = data
        self.batch_size = batch_size
        self.backend    = backend
        self.images     = []
        self.num_images = -1
        self.index      = 0
        
        self.list_files(data=self.data)

    def __len__(self):
        """Return the number of images in the `image_files`."""
        return self.num_images  # Number of images
    
    def __iter__(self):
        """Return an iterator starting at index 0."""
        self.index = 0
        return self

    def __next__(self) -> tuple[Tensor, list, list, list]:
        """Next iterator.
        
        Examples:
            >>> video_stream = ImageLoader("cam_1.mp4")
            >>> for index, image in enumerate(video_stream):
        
        Returns:
            images (Tensor[B, C, H, W]):
                Images tensor.
            indexes (list):
                List of image indexes.
            files (list):
                List of image files.
            rel_paths (list):
                List of images' relative paths corresponding to data.
        """
        if self.index >= self.num_images:
            raise StopIteration
        else:
            images    = []
            indexes   = []
            files     = []
            rel_paths = []

            for i in range(self.batch_size):
                if self.index >= self.num_images:
                    break
                
                file     = self.images[self.index]
                rel_path = file.replace(self.data, "")
                image    = read_image(
                    path    = self.images[self.index],
                    backend = self.backend
                )
                # image  = image[:, :, ::-1]  # BGR to RGB
                
                images.append(image)
                indexes.append(self.index)
                files.append(file)
                rel_paths.append(rel_path)

                self.index += 1
            
            # return np.array(images), indexes, files, rel_paths
            return torch.stack(images), indexes, files, rel_paths
    
    # MARK: Configure
    
    def list_files(self, data: str):
        """Initialize list of image files in data source.
        
        Args:
            data (str):
                Data source. Can be a path to an image file or a directory.
                It can be a pathname pattern to image files.
        """
        if is_image_file(data):
            self.images = [data]
        elif os.path.isdir(data):
            self.images = [
                i for i in glob(os.path.join(data, "**/*"), recursive=True)
                if is_image_file(i)
            ]
        elif isinstance(data, str):
            self.images = [i for i in glob(data) if is_image_file(i)]
        else:
            raise IOError("Error when listing image files.")
        self.num_images = len(self.images)


class ImageWriter:
    """Video Writer saves images to a destination directory.

    Args:
        dst (str):
            Output directory or filepath.
        extension (str):
            Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
            Default: `jpg`.
    """

    # MARK: Magic Functions

    def __init__(self, dst: str, extension: str = ".jpg"):
        super().__init__()
        self.dst	   = dst
        self.extension = extension
        self.index     = 0

    def __len__(self):
        """Return the number of already written images."""
        return self.index

    # MARK: Write

    def write_image(
        self,
        image     : Tensor,
        image_file: Union[str, None] = None
    ):
        """Write image.

        Args:
            image (Tensor[C, H, W]):
                Image.
            image_file (str, None):
                Path to save image. Default: `None`.
        """
        image = to_image(image=image, keep_dims=False, denormalize=True)
        
        if image_file is not None:
            image_file = (image_file[1:] if image_file.startswith("\\")
                          else image_file)
            image_file = (image_file[1:] if image_file.startswith("/")
                          else image_file)
            image_name = os.path.splitext(image_file)[0]
        else:
            image_name = f"{self.index}"
        
        output_file = os.path.join(self.dst, f"{image_name}{self.extension}")
        parent_dir  = str(Path(output_file).parent)
        create_dirs(paths=[parent_dir])
        
        cv2.imwrite(output_file, image)
        self.index += 1

    def write_images(
        self,
        images     : Tensors,
        image_files: Union[list[str], None] = None
    ):
        """Write batch of images.

        Args:
            images (Tensors):
                Images.
            image_files (list[str], None):
                Paths to save images. Default: `None`.
        """
        if image_files is None:
            image_files = [None for _ in range(len(images))]

        for image, image_file in zip(images, image_files):
            self.write_image(image=image, image_file=image_file)


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
                image       = input,
                keep_dims   = self.keep_dims,
                denormalize = self.denormalize
            ), \
            to_image(
                image       = target,
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


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
