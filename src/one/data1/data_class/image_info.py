#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class for storing image info.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Union

from PIL import Image

from one.core import get_exif_size
from one.core import Int2Or3T
from one.core import Int3T

"""
from one.core import error_console
try:
	import pyvips
except ImportError:
	error_console.log(f"Cannot import `pyvips`.")
"""

__all__ = [
	"ImageInfo"
]


# MARK: - ImageInfo

@dataclass
class ImageInfo:
	"""ImageInfo is a data class for storing image information.
	
	Attributes:
		id (ID):
			Image ID. This attribute is useful for batch processing but you
			want to keep the objects in the correct frame sequence.
		name (str):
			Image name with extension.
		path (str):
			Image path.
		height0 (int):
			Original image height.
		width0 (int):
			Original image width.
		height (int):
			Resized image height.
		width (int):
			Resized image width.
		depth (int):
			Image channels.
		
	References:
		https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4
	"""

	id     : Union[int, str] = uuid.uuid4().int
	name   : str = ""
	path   : str = ""
	height0: int = 0
	width0 : int = 0
	height : int = 0
	width  : int = 0
	depth  : int = 3

	# MARK: Configure

	@staticmethod
	def from_file(image_path: str, info: Optional[ImageInfo] = None) -> ImageInfo:
		"""Parse image info from image file.

		Args:
			image_path (str):
				Image path.
			info (ImageInfo, optional):
				`ImageInfo` object.
				
		Returns:
			info (ImageInfo):
				`ImageInfo` object.
		"""
		# NOTE: Get image shape
		from one import is_bmp_file
		from one import VISION_BACKEND
		from one import VisionBackend
		
		if is_bmp_file(image_path) or VISION_BACKEND == VisionBackend.PIL:
			# NOTE: Using PIL = 315 ms ± 8.76 ms per loop
			image  = Image.open(image_path)
			image.verify()  # PIL verify
			shape0 = get_exif_size(image)  # Image size (height, width)
		# elif VISION_BACKEND == VisionBackend.LIBVIPS:
			# NOTE: Using VIPS = 69.1 ms ± 31.3 µs per loop
			# image  = pyvips.Image.new_from_file(image_path)
			# shape0 = (image.height, image.width)  # H, W

		if (shape0[0] <= 1) or (shape0[1] <= 1):
			raise ValueError(f"Image size (height and width) must > 1 pixel."
			                 f"But got: {shape0}.")
		
		# NOTE: Parse image info
		path = Path(image_path)
		stem = str(path.stem)
		name = str(path.name)
		
		info         = ImageInfo() if info is None               else info
		info.id      = stem        if info.id      != stem       else info.id
		info.name    = name        if info.name    != name       else info.name
		info.path    = image_path  if info.path    != image_path else info.path
		info.height0 = shape0[0]   if info.height0 != shape0[0]  else info.height0
		info.width0  = shape0[1]   if info.width0  != shape0[1]  else info.width0
		info.height  = shape0[0]   if info.height  != shape0[0]  else info.height
		info.width   = shape0[1]   if info.width   != shape0[1]  else info.width
		return info
	
	# MARK: Properties
	
	@property
	def shape0(self) -> Int3T:
		"""Return the image's original shape [H, W, C]."""
		return self.height0, self.width0, self.depth
	
	@shape0.setter
	def shape0(self, value: Int2Or3T):
		"""Assign the image's original shape."""
		if len(value) == 3:
			self.height0, self.width0, self.depth = value[0], value[1], value[2]
		elif len(value) == 2:
			self.height0, self.width0 = value[0], value[1]
	
	@property
	def shape(self) -> Int3T:
		"""Return the image's resized shape [H, W, C]."""
		return self.height, self.width, self.depth
	
	@shape.setter
	def shape(self, value: Int2Or3T):
		"""Assign the image's resized shape [H, W, C]."""
		if len(value) == 3:
			self.height, self.width, self.depth = value[0], value[1], value[2]
		elif len(value) == 2:
			self.height, self.width = value[0], value[1]
