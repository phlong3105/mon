#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import inspect
import os
import sys
from glob import glob
from pathlib import Path
from typing import Optional
from typing import Union

import cv2
import numpy as np
from PIL import Image

from one.core import Arrays
from one.core import create_dirs
from one.core import is_channel_first
from one.core import is_image_file
from one.core import to_4d_array
from one.core import to_channel_last
from one.core import VisionBackend


# MARK: - Functional

def read_image_cv(path: str) -> np.ndarray:
	"""Read image using opencv."""
	image = cv2.imread(path)   # BGR
	image = image[:, :, ::-1]  # BGR -> RGB
	image = np.ascontiguousarray(image)
	return image

'''
def read_image_libvips(path: str) -> np.ndarray:
	"""Read image using libvips."""
	image   = pyvips.Image.new_from_file(path, access="sequential")
	mem_img = image.write_to_memory()
	image   = np.frombuffer(mem_img, dtype=np.uint8).reshape(image.height, image.width, 3)
	return image
'''


def read_image_pil(path: str) -> np.ndarray:
	"""Read image using PIL."""
	image = Image.open(path)
	return np.asarray(image)


def read_image(path: str, backend: Union[VisionBackend, str, int] = "cv") -> np.ndarray:
	"""Read image with the corresponding backend."""
	if isinstance(backend, (str, int)):
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
	

# MARK: - Modules


class ImageLoader:
	"""Image Loader retrieves and loads image(s) from a filepath, a pathname
	pattern, or directory.

	Attributes:
		data (str):
			Data source. Can be a path to an image file or a directory.
			It can be a pathname pattern to images.
		batch_size (int):
			Number of samples in one forward & backward pass.
		image_files (list):
			List of image files found in the data source.
		num_images (int):
			Total number of images.
		index (int):
			Current index.
	"""

	# MARK: Magic Functions

	def __init__(self, data: str, batch_size: int = 1):
		super().__init__()
		self.data        = data
		self.batch_size  = batch_size
		self.image_files = []
		self.num_images  = -1
		self.index       = 0
		
		self.init_image_files(data=self.data)

	def __len__(self):
		"""Return the number of images in the `image_files`."""
		return self.num_images  # Number of images
	
	def __iter__(self):
		"""Return an iterator starting at index 0."""
		self.index = 0
		return self

	def __next__(self):
		"""Next iterator.
		
		Examples:
			>>> video_stream = ImageLoader("cam_1.mp4")
			>>> for index, image in enumerate(video_stream):
		
		Returns:
			images (np.ndarray):
				List of image file from opencv with `np.ndarray` type.
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
				
				file     = self.image_files[self.index]
				rel_path = file.replace(self.data, "")
				image    = cv2.imread(self.image_files[self.index])
				image    = image[:, :, ::-1]  # BGR to RGB
				
				images.append(image)
				indexes.append(self.index)
				files.append(file)
				rel_paths.append(rel_path)

				self.index += 1

			return np.array(images), indexes, files, rel_paths
	
	# MARK: Configure
	
	def init_image_files(self, data: str):
		"""Initialize list of image files in data source.
		
		Args:
			data (str):
				Data source. Can be a path to an image file or a directory.
				It can be a pathname pattern to images.
		"""
		if is_image_file(data):
			self.image_files = [data]
		elif os.path.isdir(data):
			self.image_files = [
				img for img in glob(os.path.join(data, "**/*"), recursive=True)
				if is_image_file(img)
			]
		elif isinstance(data, str):
			self.image_files = [img for img in glob(data) if is_image_file(img)]
		else:
			raise IOError("Error when reading input image files.")
		self.num_images = len(self.image_files)

	def list_image_files(self, data: str):
		"""Alias of `init_image_files()`."""
		self.init_image_files(data=data)


class ImageWriter:
	"""Video Writer saves images to a destination directory.

	Attributes:
		dst (str):
			Output directory or filepath.
		extension (str):
			Image file extension. One of [`.jpg`, `.jpeg`, `.png`, `.bmp`].
		index (int):
			Current index. Default: `0`.
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

	def write_image(self, image: np.ndarray, image_file: Optional[str] = None):
		"""Write image.

		Args:
			image (np.ndarray):
				Image.
			image_file (str, optional):
				Image file.
		"""
		if is_channel_first(image):
			image = to_channel_last(image)
		
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

	def write_images(self, images: Arrays, image_files: Optional[list[str]] = None):
		"""Write batch of images.

		Args:
			images (Arrays):
				Images.
			image_files (list[str], optional):
				Image files.
		"""
		images = to_4d_array(images)
		
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
