#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image I/O.

This module implements the basic I/O functionalities of images.
"""

from __future__ import annotations

__all__ = [
    "read_image",
    "read_image_shape",
    "write_image",
    "write_image_cv",
    "write_image_torch",
    "write_images_cv",
    "write_images_torch",
]

import multiprocessing
from typing import Sequence

import cv2
import joblib
import numpy as np
import torch
import torchvision

from mon.core import pathlib
from mon.core.image import utils


# region Read

def read_image(
    path     : pathlib.Path,
    flags    : int  = cv2.IMREAD_COLOR,
    to_tensor: bool = False,
    normalize: bool = False,
) -> torch.Tensor | np.ndarray:
    """Read an image from a file path using :obj:`cv2`. Optionally, convert it
    to RGB format, and :obj:`torch.Tensor` type of shape `[1, C, H, W]`.

    Args:
        path: An image's file path.
        flags: A flag to read the image. One of:
            - cv2.IMREAD_UNCHANGED           = -1,
            - cv2.IMREAD_GRAYSCALE           = 0,
            - cv2.IMREAD_COLOR               = 1,
            - cv2.IMREAD_ANYDEPTH            = 2,
            - cv2.IMREAD_ANYCOLOR            = 4,
            - cv2.IMREAD_LOAD_GDAL           = 8,
            - cv2.IMREAD_REDUCED_GRAYSCALE_2 = 16,
            - cv2.IMREAD_REDUCED_COLOR_2     = 17,
            - cv2.IMREAD_REDUCED_GRAYSCALE_4 = 32,
            - cv2.IMREAD_REDUCED_COLOR_4     = 33,
            - cv2.IMREAD_REDUCED_GRAYSCALE_8 = 64,
            - cv2.IMREAD_REDUCED_COLOR_8     = 65,
            - cv2.IMREAD_IGNORE_ORIENTATION  = 128
            Default: ``cv2.IMREAD_COLOR``.
        to_tensor: If ``True``, convert the image from :obj:`numpy.ndarray`
            to :obj:`torch.Tensor`. Default: ``False``.
        normalize: If ``True``, normalize the image to ``[0.0, 1.0]``.
            Default: ``False``.
        
    Return:
        A :obj:`numpy.ndarray` image of shape0 `[H, W, C]` with value in
        range ```[0, 255]``` or a :obj:`torch.Tensor` image of shape
        `[1, C, H, W]` with value in range ```[0.0, 1.0]```.
    """
    image = cv2.imread(str(path), flags)  # BGR
    if image.ndim == 2:  # HW -> HW1 (OpenCV read grayscale image)
        image = np.expand_dims(image, axis=-1)
    if utils.is_color_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if to_tensor:
        image = utils.to_image_tensor(image, False, normalize)
    return image


def read_image_shape(path: pathlib.Path) -> tuple[int, int, int]:
    """Read an image from a file path using :obj:`cv2` and get its shape as
    `[H, W, C]`.
    
    Args:
        path: An image file path.
    """
    image = cv2.imread(str(path))  # BGR
    return image.shape

# endregion


# region Write

def write_image(path: pathlib.Path, image: torch.Tensor | np.ndarray):
    """Write an image to a file path.
    
    Args:
        image: An image to write.
        path: A directory to write the image to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        torchvision.utils.save_image(image, str(path))
    else:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    

def write_image_cv(
    image      : torch.Tensor | np.ndarray,
    dir_path   : pathlib.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
    denormalize: bool = False
):
    """Write an image to a directory using :obj:`cv2`.
    
    Args:
        image: An image to write.
        dir_path: A directory to write the image to.
        name: An image's name.
        prefix: A prefix to add to the :obj:`name`.
        extension: An extension of the image file. Default: ``'.png'``.
        denormalize: If ``True``, convert the image to ``[0, 255]``.
            Default: ``False``.
    """
    # Convert image
    if isinstance(image, torch.Tensor):
        image = utils.to_image_nparray(image=image, keepdim=True, denormalize=denormalize)
    image = utils.to_channel_last_image(image=image)
    if 2 <= image.ndim <= 3:
        raise ValueError(
            f"image's number of dimensions must be between ``2`` and ``3``, "
            f"but got {image.ndim}."
        )
    # Write image
    dir_path  = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    name      = pathlib.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}"  if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    file_path = dir_path / name
    cv2.imwrite(str(file_path), image)


def write_image_torch(
    image      : torch.Tensor | np.ndarray,
    dir_path   : pathlib.Path,
    name       : str,
    prefix     : str  = "",
    extension  : str  = ".png",
    denormalize: bool = False
):
    """Write an image to a directory.
    
    Args:
        image: An image to write.
        dir_path: A directory to write the image to.
        name: An image's name.
        prefix: A prefix to add to the :obj:`name`.
        extension: An extension of the image file. Default: ``'.png'``.
        denormalize: If ``True``, convert the image to ``[0, 255]``.
            Default: ``False``.
    """
    # Convert image
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        image = utils.to_channel_first_image(image=image)
    image = utils.denormalize_image(image=image) if denormalize else image
    image = image.to(torch.uint8)
    image = image.cpu()
    if 2 <= image.ndim <= 3:
        raise ValueError(
            f"`image`'s number of dimensions must be between ``2`` and ``3``, "
            f"but got {image.ndim}."
        )
    # Write image
    dir_path  = pathlib.Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    name      = pathlib.Path(name)
    stem      = name.stem
    extension = extension  # name.suffix
    extension = f"{name.suffix}" if extension == "" else extension
    extension = f".{extension}" if "." not in extension else extension
    stem      = f"{prefix}_{stem}" if prefix != "" else stem
    name      = f"{stem}{extension}"
    file_path = dir_path / name
    if extension in [".jpg", ".jpeg"]:
        torchvision.io.image.write_jpeg(input=image, filename=str(file_path))
    elif extension in [".png"]:
        torchvision.io.image.write_png(input=image, filename=str(file_path))


def write_images_cv(
    images     : list[torch.Tensor | np.ndarray],
    dir_path   : pathlib.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".png",
    denormalize: bool      = False
):
    """Write a :obj:`list` of images to a directory using :obj:`cv2`.
   
    Args:
        images: A :obj:`list` of 3D images.
        dir_path: A directory to write the images to.
        names: A :obj:`list` of images' names.
        prefixes: A prefix to add to the :obj:`names`.
        extension: An extension of image files. Default: ``'.png'``.
        denormalize: If ``True``, convert image to ``[0, 255]``.
            Default: ``False``.
    """
    if isinstance(names, str):
        names    = [names    for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    if not len(images) == len(names):
        raise ValueError(
            f"The length of `images` and `names` must be the same, "
            f"but got {len(images)} and {len(names)}."
        )
    if not len(images) == len(prefixes):
        raise ValueError(
            f"The length of `images` and `prefixes` must be the "
            f"same, but got {len(images)} and {len(prefixes)}."
        )
    num_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(write_image_cv)(
            image, dir_path, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )


def write_images_torch(
    images     : Sequence[torch.Tensor | np.ndarray],
    dir_path   : pathlib.Path,
    names      : list[str],
    prefixes   : list[str] = "",
    extension  : str       = ".png",
    denormalize: bool      = False
):
    """Write a :obj:`list` of images to a directory using :obj:`torchvision`.
   
    Args:
        images: A :obj:`list` of 3D images.
        dir_path: A directory to write the images to.
        names: A :obj:`list` of images' names.
        prefixes: A prefix to add to the :obj:`names`.
        extension: An extension of image files. Default: ``'.png'``.
        denormalize: If ``True``, convert image to ``[0, 255]``.
            Default: ``False``.
    """
    if isinstance(names, str):
        names    = [names    for _ in range(len(images))]
    if isinstance(prefixes, str):
        prefixes = [prefixes for _ in range(len(prefixes))]
    if not len(images) == len(names):
        raise ValueError(
            f"The length of `images` and `names` must be the same, "
            f"but got {len(images)} and {len(names)}."
        )
    if not len(images) == len(prefixes):
        raise ValueError(
            f"The length of `images` and `prefixes` must be the same, "
            f"but got {len(images)} and {len(prefixes)}."
        )
    num_jobs = multiprocessing.cpu_count()
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(write_image_torch)(
            image, dir_path, names[i], prefixes[i], extension, denormalize
        )
        for i, image in enumerate(images)
    )

# endregion
