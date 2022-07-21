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
from one.core import Tensors
from one.core import to_3d_tensor_list
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
    """If the input image size is not a multiple of the stride, then the image
    size is updated to the next multiple of the stride.
    
    Args:
        size (Ints): the size of the image.
        stride (int): the stride of the network. Defaults to 32.
    
    Returns:
        The new size of the image.
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
    """If the image has an EXIF orientation tag, and the orientation is 6 or
    8, then the image is rotated 90 or 270 degrees, so we swap the width and
    height.
    
    Args:
        image (PIL.Image): PIL.Image.
    
    Returns:
        The height and width of the image.
    """
    size = image.size  # (width, height)
    try:
        rotation = dict(image._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            size = [size[1], size[0]]
        elif rotation == 8:  # rotation 90
            size = [size[1], size[0]]
    except:
        pass
    return [size[1], size[0]]


def get_image_center(image: Tensor) -> Tensor:
    """It takes an image and returns the center of the image as (x=h/2, y=w/2).
    
    Args:
        image (Tensor): The image to get the center of.
    
    Returns:
        The center of the image.
    """
    assert_tensor(image)
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2])


def get_image_center4(image: Tensor) -> Tensor:
    """It takes an image and returns the coordinates of the center of the image
    as (x=h/2, y=w/2, x=h/2, y=w/2).
    
    Args:
        image (Tensor): the image tensor.
    
    Returns:
        The center of the image.
    """
    assert_tensor(image)
    h, w = get_image_hw(image)
    return torch.Tensor([h / 2, w / 2, h / 2, w / 2])
    

def get_image_hw(image: Tensor) -> Ints:
    """Given an image tensor, return its height and width
    
    Args:
        image (Tensor[..., C, H, W])): The image tensor.
    
    Returns:
        The height and width of the image.
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if is_channel_first(image):  # [.., C, H, W]
        return [image.shape[-2], image.shape[-1]]
    else:  # [.., H, W, C]
        return [image.shape[-3], image.shape[-2]]
    
    
get_image_size = get_image_hw


def get_image_shape(image: Tensor) -> Ints:
    """
    It returns the shape of the image as a list of integers.
    
    Args:
        image (Tensor): The image tensor.
    
    Returns:
        The shape of the image as [C, H, W].
    """
    assert_tensor_of_atleast_ndim(image, 3)
    if is_channel_first(image):  # [.., C, H, W]
        return [image.shape[-3], image.shape[-2], image.shape[-1]]
    else:  # [.., H, W, C]
        return [image.shape[-1], image.shape[-3], image.shape[-2]]


def get_num_channels(image: Tensor) -> int:
    """
    It returns the number of channels in an image.
    
    Args:
        image (Tensor): The image to get the number of channels from.
    
    Returns:
        The number of channels in the image.
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
    """
    If the first dimension is the smallest, then it's channel first.
    
    Args:
        image (Tensor): The image to be checked.
    
    Returns:
        A boolean value.
    """
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
    """
    If the image is not channel first, then it is channel last.
    
    Args:
        image (Tensor): The image tensor.
    
    Returns:
        A boolean value.
    """
    return not is_channel_first(image)


def read_image_cv(path: str) -> Tensor:
    """
    It reads an image from a path, converts it to RGB, and converts it to a
    PyTorch tensor.
    
    Args:
        path (str): The path to the image file.
    
    Returns:
        A tensor of shape [1, C, H, W].
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
    """
    It reads an image from a file, converts it to a tensor, and returns the
    tensor.
    
    Args:
        path (str): The path to the image file.
    
    Returns:
        A tensor of shape [1, C, H, W].
    """
    image = Image.open(path)                         # PIL Image
    image = to_tensor(image=image, keep_dims=False)  # Tensor[C, H, W]
    return image


def read_image(path: str, backend: VisionBackend_ = VisionBackend.CV) -> Tensor:
    """
    It reads an image from a file path and returns a tensor.
    
    Args:
        path (str): The path to the image file.
        backend (VisionBackend_): Vision backend to use.
    
    Returns:
        A tensor of shape [1, C, H, W].
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
    """
    It takes a tensor of any shape and returns a tensor of the same shape,
    but with the channel first.
    
    Args:
        image (Tensor): The image to convert.
        keep_dims (bool): If True, the dimension will be kept. Defaults to True.
        inplace (bool): If True, the operation will be done in-place.
            Defaults to False
    
    Returns:
        A tensor with the channel first.
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
    """
    It takes a tensor of any shape and returns a tensor of the same shape,
    but with the channel last.
    
    Args:
        image (Tensor): The image to convert.
        keep_dims (bool): If True, the dimension will be kept.
            Defaults to True
        inplace (bool): If True, the operation will be done in-place.
            Defaults to False
    
    Returns:
        A tensor with the channel last.
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
    """
    It converts a tensor to a numpy array.
    
    Args:
        image (Tensor): Tensor
        keep_dims (bool): If True, the function will keep the dimensions of
            the input tensor. Defaults to True.
        denormalize (bool): If True, the image will be denormalized to [0, 255].
            Defaults to False.
        inplace (bool): If True, the input tensor will be modified inplace.
            Defaults to False.
    
    Returns:
        A numpy array of the image.
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


def to_pil_image(image: Tensor | np.ndarray) -> PIL.Image:
    """
    It converts a tensor or numpy array to a PIL image.
    
    Args:
        image (Tensor | np.ndarray): The image to be transformed.
    
    Returns:
        A PIL image.
    """
    if torch.is_tensor(image):
        # Equivalent to: `np_image = image.numpy()` but more efficient
        return F.pil_to_tensor(image)
    elif isinstance(image, np.ndarray):
        return PIL.Image.fromarray(image.astype(np.uint8), "RGB")
    raise TypeError(f"Do not support {type(image)}.")


def to_tensor(
    image    : Tensor | np.ndarray | PIL.Image,
    keep_dims: bool = True,
    normalize: bool = False,
    inplace  : bool = False,
) -> Tensor:
    """
    Convert a Tensor, np.ndarray, or PIL.Image to a Tensor with channel
    first format.
    
    Args:
        image (Tensor | np.ndarray | PIL.Image): The image to be converted.
        keep_dims (bool): If True, the channel dimension will be kept. If False
            unsqueeze the image to match the shape [..., C, H, W].
            Defaults to True
        normalize (bool): If True, normalize the image to [0, 1].
            Defaults to False
        inplace (bool): If True, the input image will be modified inplace.
            Defaults to False
    
    Returns:
        A tensor.
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
    """
    It takes an image, a directory, a name, a prefix, and an extension, and
    writes the image to the directory with the name and prefix and extension.
    
    Args:
        image (np.ndarray): The image to write.
        dir (str): The directory to write the image to.
        name (str): The name of the image.
        prefix (str): The prefix to add to the name of the image.
        extension (str): The extension of the image file. Defaults to .png
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
    """
    It takes an image, a directory, a name, a prefix, and an extension, and
    writes the image to the directory with the name and prefix and extension.
    
    Args:
        image (np.ndarray): The image to write.
        dir (str): The directory to write the image to.
        name (str): The name of the image.
        prefix (str): The prefix to add to the name of the image.
        extension (str): The extension of the image file. Defaults to .png
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
    """
    It writes a batch of images to disk.
    
    Args:
        images (np.ndarray): a 4D numpy array of shape [B, H, W, C].
        dir (str): The directory to write the image to.
        name (str): The name of the image.
        extension (str): The extension of the image file. Defaults to .png
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
    """
    It writes a batch of images to disk.
    
    Args:
        images (Tensor): a 4D tensor of shape [B, C, H, W]
        dir (str): The directory to write the images to.
        name (str): The name of the image.
        extension (str): The file extension of the image. Defaults to .png
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
    """
    It takes a list of images, and writes them to a directory.
    
    Args:
        images (list): list of images to be written.
        dir (str): the directory to save the images to.
        name (str): The name of the image.
        extension (str): The file extension of the image. Defaults to .png
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
    """
    It takes a dictionary of images, concatenates them, and writes them to a
    file.
    
    Args:
        images (dict): a dictionary of images, where the keys are the names of
            the images and the values are the images themselves.
        dir (str): The directory to save the images to.
        name (str): The name of the image.
        extension (str): The file extension of the image. Defaults to .png
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
        raise TypeError(f"Do not support {type(images)}.")


# MARK: - Modules


class ImageLoader:
    """Image Loader retrieves and loads image(s) from a filepath, a pathname
    pattern, or directory.

    Args:
        data (str):
            The path to the image file. Can be a path to an image file or a
            directory, or a pathname pattern to images.
        batch_size (int):
            The number of images to be processed at once. Defaults to 1
        backend (VisionBackend_):
           The backend to use for image processing. Default to VisionBackend.CV.
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

    def __len__(self) -> int:
        """
        The function returns the number of images in the dataset.
        
        Returns:
            The number of images in the dataset.
        """
        return self.num_images
    
    def __iter__(self):
        """
        The __iter__() function returns an iterator object.
        
        Returns:
            The object itself.
        """
        self.index = 0
        return self

    def __next__(self) -> tuple[Tensor, list, list, list]:
        """
        It reads a batch of images from the disk, converts them to RGB, and
        returns them as a tensor.
        
        Examples:
            >>> images = ImageLoader("cam_1.mp4")
            >>> for index, image in enumerate(images):
        
        Returns:
            Images tensor of shape [B, C, H, W].
            List of image indexes
            List of image files.
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
            data (str): The path to the image file. Can be a path to an image
                file or a directory, or a pathname pattern to images.
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
        dst (str): The destination folder where the images will be saved.
        extension (str): The extension of the file to be saved.
            Defaults to .jpg
    """

    # MARK: Magic Functions

    def __init__(self, dst: str, extension: str = ".jpg"):
        super().__init__()
        self.dst	   = dst
        self.extension = extension
        self.index     = 0

    def __len__(self) -> int:
        """
        The function returns the number of items in the stack.
        
        Returns:
            The index of the last item in the list.
        """
        return self.index

    # MARK: Write

    def write_image(
        self,
        image      : Tensor,
        image_file : str | None = None,
        denormalize: bool       = True
    ):
        """
        It takes an image, converts it to a numpy array, and saves it to disk.
        
        Args:
            image (Tensor): The image to write.
            image_file (str | None): The name of the image file.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to False.
        """
        image = to_image(image=image, keep_dims=False, denormalize=denormalize)
        
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
        image_files: list[str] | None = None,
        denormalize: bool             = True,
    ):
        """
        Write a list of images to disk.
        
        Args:
            images (Tensors): A list of tensors, each of which is a single
                image.
            image_files (list[str] | None): Paths to save images.
                Defaults to None.
            denormalize (bool): If True, the image will be denormalized to
                [0, 255]. Defaults to False.
        """
        if image_files is None:
            image_files = [None for _ in range(len(images))]
        
        images = to_3d_tensor_list(images)
        for image, image_file in zip(images, image_files):
            self.write_image(
                image=image, image_file=image_file, denormalize=denormalize
            )


@TRANSFORMS.register(name="to_image")
class ToImage(Transform):
    """Converts a Tensor to a numpy image. In case the image is in the GPU,
    it will be copied back to CPU.

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
