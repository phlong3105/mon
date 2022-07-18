#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import multiprocessing
import os
import sys
from glob import glob
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torchvision
from joblib import delayed
from joblib import Parallel
from multipledispatch import dispatch
from PIL import Image
from torch import Tensor

from one.core import create_dirs
from one.core import is_image_file
from one.core import Tensors
from one.core import VisionBackend
from one.vision.transformation import denormalize_naive
from one.vision.transformation import is_channel_first
from one.vision.transformation import to_channel_last
from one.vision.transformation import to_image
from one.vision.transformation import to_tensor


# MARK: - Functional

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


def read_image(
	path   : str,
	backend: Union[VisionBackend, str, int] = VisionBackend.CV,
) -> Tensor:
	"""Read image with the corresponding backend.
	
	Args:
		path (str):
			Image file.
		backend (VisionBackend, str, int):
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
	if not 2 <= image.ndim <= 3:
		raise ValueError(
			f"Require 2 <= `image.ndim` <= 3. But got: {image.ndim}."
		)
	
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
	if not 2 <= image.ndim <= 3:
		raise ValueError(
			f"Require 2 <= `image.ndim` <= 3. But got: {image.ndim}."
		)
	
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
	if not images == 4:
		raise ValueError(
			f"Require `image.ndim` == 4. But got: {images.ndim}."
		)
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
	if not images == 4:
		raise ValueError(
			f"Require `image.ndim` == 4. But got: {images.ndim}."
		)
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
		backend (VisionBackend, str, int):
			Vision backend used to read images. Default: `VisionBackend.CV`.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		data      : str,
		batch_size: int = 1,
		backend   : Union[VisionBackend, str, int] = VisionBackend.CV
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
		image = to_image(input=image, keep_dims=False, denormalize=True)
		
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


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
