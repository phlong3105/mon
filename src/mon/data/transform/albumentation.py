#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements data augmentation functions by extending
:mod:`albumentations` package. These functions are mainly applied to
:class:`numpy.ndarray` images.
"""

from __future__ import annotations

from typing import Literal

# noinspection PyPackageRequirements,PyUnresolvedReferences
import albumentations
from albumentations import *
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.core.pydantic import InterpolationType, ProbabilityType
from albumentations.core.transforms_interface import BaseTransformInitSchema, DualTransform
from albumentations.core.types import (BoxInternalType, KeypointInternalType, Targets)
from pydantic import Field


# region Crop

class CropPatch(DualTransform):
	"""Crop a patch of the image according to
	`<https://github.com/ZhendongWang6/Uformer/blob/main/dataset/dataset_denoise.py>__`
	"""
	
	def __init__(
		self,
		patch_size  : int   = 128,
		always_apply: bool  = False,
		p           : float = 0.5,
	):
		super().__init__(
			always_apply = always_apply,
			p            = p,
		)
		self.patch_size = patch_size
		self.r = 0
		self.c = 0
	
	def apply(self, img: np.ndarray, r: int, c: int, **params) -> np.ndarray:
		return img[r:r + self.patch_size, c:c + self.patch_size, :]
	
	def apply_to_mask(self, img: np.ndarray, r: int, c: int, **params) -> np.ndarray:
		return img[r:r + self.patch_size, c:c + self.patch_size, :]
	
	@property
	def targets_as_params(self) -> list[str]:
		return ["image"]
	
	def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
		image   = params["image"]
		h, w, c = image.shape
		if h - self.patch_size == 0:
			r = 0
			c = 0
		else:
			r = np.random.randint(0, h - self.patch_size)
			c = np.random.randint(0, w - self.patch_size)
		return {"r": r, "c": c}

# endregion


# region Normalize

class NormalizeImageMeanStd(DualTransform):
	"""Normalize image by given :attr:`mean` and :attr:`std`."""
	
	def __init__(
		self,
		mean        : Sequence[float] = [0.485, 0.456, 0.406],
		std         : Sequence[float] = [0.229, 0.224, 0.225],
		always_apply: bool            = True,
		p           : float           = 1.0,
	):
		super().__init__(
			always_apply = always_apply,
			p            = p,
		)
		self.mean = mean
		self.std  = std
	
	def apply(self, img: np.ndarray, **params) -> np.ndarray:
		return (img - self.mean) / self.std
	
	def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
		return (img - self.mean) / self.std
		
# endregion


# region Resize

class ResizeMultipleOf(DualTransform):
	"""Resize the input to the given height and width and ensure that they are
	constrained to be multiple of a given number.
	
    Args:
        height: Desired height of the output.
        width: Desired width of the output.
        keep_aspect_ratio: If ``True``, keep the aspect ratio of the input sample.
            Output sample might not have the given width and height, and
            resize behaviour depends on the parameter :param:`resize_method`.
            Default: ``False``.
        multiple_of: Output height and width are constrained to be
            multiple of this parameter. Default: ``1``.
        resize_method: Resize method.
            ``"lower_bound"``: Output will be at least as large as the given size.
            ``"upper_bound"``: Output will be at max as large as the given size. (Output size might be smaller than given size.)
            ``"minimal"``    : Scale as least as possible. (Output size might be smaller than given size.)
            Default: ``"lower_bound"``.
        interpolation: Flag that is used to specify the interpolation algorithm.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
            cv2.INTER_AREA, cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.
        p: Probability of applying the transform. Default: 1.
	
    Targets:
        image, mask, bboxes, keypoints.
	
    Image types:
        uint8, float32
    """
	
	_targets = (Targets.IMAGE, Targets.MASK, Targets.KEYPOINTS, Targets.BBOXES)
	
	class InitSchema(BaseTransformInitSchema):
		height           : int  = Field(ge=1,                 description="Desired height of the output.")
		width            : int  = Field(ge=1,                 description="Desired width of the output.")
		keep_aspect_ratio: bool = Field(False,         description="Keep the aspect ratio of the input sample.")
		multiple_of      : int  = Field(1,             description="Output height and width are constrained to be multiple of this parameter.")
		resize_method	 : str  = Field("lower_bound", description="Resize method.")
		interpolation    : InterpolationType = cv2.INTER_AREA
		p                : ProbabilityType   = 1
	
	def __init__(
		self,
		height           : int,
		width            : int,
		keep_aspect_ratio: bool        = False,
		multiple_of      : int         = 1,
		resize_method    : Literal["lower_bound", "upper_bound", "minimal"] = "lower_bound",
		interpolation    : int         = cv2.INTER_AREA,
		always_apply     : bool | None = None,
		p                : float       = 1,
	):
		super().__init__(p, always_apply)
		self.height            = height
		self.width             = width
		self.keep_aspect_ratio = keep_aspect_ratio
		self.multiple_of       = multiple_of
		self.resize_method     = resize_method
		self.interpolation     = interpolation
	
	def constrain_to_multiple_of(self, x, min_val: int = 0, max_val: int | None = None):
		y = (np.round(x / self.multiple_of) * self.multiple_of).astype(int)
		if max_val and y > max_val:
			y = (np.floor(x / self.multiple_of) * self.multiple_of).astype(int)
		if y < min_val:
			y = (np.ceil(x / self.multiple_of) * self.multiple_of).astype(int)
		return y
	
	def get_size(self, height: int, width: int) -> tuple[int, int]:
		# Determine new height and width
		scale_height = self.height / height
		scale_width  = self.width  / width
		
		if self.keep_aspect_ratio:
			if self.resize_method == "lower_bound":
				# Scale such that output size is lower bound
				if scale_width > scale_height:
					# Fit width
					scale_height = scale_width
				else:
					# Fit height
					scale_width = scale_height
			elif self.resize_method == "upper_bound":
				# Scale such that output size is upper bound
				if scale_width < scale_height:
					# Fit width
					scale_height = scale_width
				else:
					# Fit height
					scale_width = scale_height
			elif self.resize_method == "minimal":
				# Scale as least as possible
				if abs(1 - scale_width) < abs(1 - scale_height):
					# Fit width
					scale_height = scale_width
				else:
					# Fit height
					scale_width = scale_height
			else:
				raise ValueError(f":param:`resize_method` {self.resize_method} not implemented")
		
		if self.resize_method == "lower_bound":
			new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.height)
			new_width  = self.constrain_to_multiple_of(scale_width  * width,  min_val=self.width)
		elif self.resize_method == "upper_bound":
			new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.height)
			new_width  = self.constrain_to_multiple_of(scale_width  * width,  max_val=self.width)
		elif self.resize_method == "minimal":
			new_height = self.constrain_to_multiple_of(scale_height * height)
			new_width  = self.constrain_to_multiple_of(scale_width  * width)
		else:
			raise ValueError(f"resize_method {self.resize_method} not implemented")
		
		return new_height, new_width
	
	def apply(self, img: np.ndarray, interpolation: int, **params: Any) -> np.ndarray:
		height, width = self.get_size(img.shape[0], img.shape[1])
		return fgeometric.resize(img, height=height, width=width, interpolation=interpolation)
	
	def apply_to_bbox(self, bbox: BoxInternalType, **params: Any) -> BoxInternalType:
		# Bounding box coordinates are scale invariant
		return bbox
	
	def apply_to_keypoint(self, keypoint: KeypointInternalType, **params: Any) -> KeypointInternalType:
		height, width = self.get_size(params["rows"], params["cols"])
		scale_x       = self.width  / width
		scale_y       = self.height / height
		return fgeometric.keypoint_scale(keypoint, scale_x, scale_y)
	
	def get_transform_init_args_names(self) -> tuple[str, ...]:
		return "height", "width", "interpolation"
	
# endregion
