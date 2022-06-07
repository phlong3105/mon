#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A flip is a motion in geometry in which an object is turned over a straight
line to form a mirror image. Every point of an object and the corresponding
point on the image are equidistant from the flip line. A flip is also called
a reflection.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms.functional import hflip
from torchvision.transforms.functional import vflip

from one.core import get_image_center4
from one.core import TensorOrArray
from one.core import TRANSFORMS
from one.imgproc.spatial.box import hflip_box
from one.imgproc.spatial.box import vflip_box

__all__ = [
	"hflip",
	"hflip_image_box",
	"vflip",
	"vflip_image_box",
	"Hflip",
	"HflipImageBox",
	"Vflip",
	"VflipImageBox",
]


# MARK: - Functional

def hflip_image_box(
	image: TensorOrArray, box: TensorOrArray = ()
) -> tuple[TensorOrArray, TensorOrArray]:
	center = get_image_center4(image)
	if isinstance(image, Tensor):
		return F.hflip(image), hflip_box(box, center)
	elif isinstance(image, np.ndarray):
		return np.fliplr(image), hflip_box(box, center)
	else:
		raise ValueError(f"Do not support: {type(image)}")


def vflip_image_box(
	image: TensorOrArray, box: TensorOrArray = ()
) -> tuple[TensorOrArray, TensorOrArray]:
	center = get_image_center4(image)
	if isinstance(image, Tensor):
		return F.vflip(image), vflip_box(box, center)
	elif isinstance(image, np.ndarray):
		return np.flipud(image), vflip_box(box, center)
	else:
		raise ValueError(f"Do not support: {type(image)}")
	

# MARK: - Modules

@TRANSFORMS.register(name="hflip")
@TRANSFORMS.register(name="horizontal_flip")
class Hflip(nn.Module):
	"""Horizontally flip a tensor image or a batch of tensor images. Input must
	be a tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].

	Examples:
		>>> hflip = Hflip()
		>>> input = torch.tensor([[[
		...    [0., 0., 0.],
		...    [0., 0., 0.],
		...    [0., 1., 1.]
		... ]]])
		>>> hflip(input)
		image([[[[0., 0., 0.],
				  [0., 0., 0.],
				  [1., 1., 0.]]]])
	"""
	
	# MARK: Magic Functions
	
	def __repr__(self):
		return self.__class__.__name__
	
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		return hflip(image)


@TRANSFORMS.register(name="hflip_image_box")
@TRANSFORMS.register(name="horizontal_flip_image_box")
class HflipImageBox(nn.Module):

	# MARK: Magic Functions
	
	def __repr__(self):
		return self.__class__.__name__
	
	# MARK: Forward Pass
	
	def forward(
		self, image: TensorOrArray, box: TensorOrArray
	) -> tuple[TensorOrArray, TensorOrArray]:
		return hflip_image_box(image=image, box=box)
	

@TRANSFORMS.register(name="vflip")
@TRANSFORMS.register(name="vertical_flip")
class Vflip(nn.Module):
	"""Vertically flip a tensor image or a batch of tensor images. Input must
	be a tensor of shape [C, H, W] or a batch of tensors [*, C, H, W].

	Examples:
		>>> vflip = Vflip()
		>>> input = torch.tensor([[[
		...    [0., 0., 0.],
		...    [0., 0., 0.],
		...    [0., 1., 1.]
		... ]]])
		>>> vflip(input)
		image([[[[0., 1., 1.],
				  [0., 0., 0.],
				  [0., 0., 0.]]]])
	"""
	
	# MARK: Magic Functions
	
	def __repr__(self):
		return self.__class__.__name__
	
	# MARK: Forward Pass
	
	def forward(self, image: Tensor) -> Tensor:
		return vflip(image)
	
	
@TRANSFORMS.register(name="vflip_image_box")
@TRANSFORMS.register(name="vertical_flip_image_box")
class VflipImageBox(nn.Module):
	
	# MARK: Magic Functions
	
	def __repr__(self):
		return self.__class__.__name__
	
	# MARK: Forward Pass
	
	def forward(
		self, image: TensorOrArray, box: TensorOrArray
	) -> tuple[TensorOrArray, TensorOrArray]:
		return vflip_image_box(image=image, box=box)
