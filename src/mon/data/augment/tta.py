#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements test-time data augmentation (TTA). These functions
are mainly applied to :class:`torch.Tensor` images.
"""

from __future__ import annotations

__all__ = [
	"BlendInput",
	"SelfEnsembleX8",
	"TestTimeAugment",
]

from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import skimage
import torch

from mon import core


# region Base

class TestTimeAugment(ABC):
	"""The base class for all Test-Time Augmentation techniques."""
	
	_to_tensor	  : bool = False
	_requires_post: bool = False
	
	def __init__(self):
		super().__init__()
		
	def __call__(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		return self.apply(input, *args, **kwargs)
	
	@classmethod
	@property
	def to_tensor(cls) -> bool:
		return cls._to_tensors
	
	@classmethod
	@property
	def requires_post(cls) -> bool:
		return cls._requires_post
	
	@abstractmethod
	def apply(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		"""Apply augmentation to the given :param:`input`."""
		pass
	
	@abstractmethod
	def postprocess(
		self,
		input : torch.Tensor,
		output: torch.Tensor,
		*args, **kwargs
	) -> torch.Tensor:
		"""Apply post-processing on the :param:`output`."""
		pass

# endregion


# region Ensemble

class SelfEnsembleX8(TestTimeAugment):
	"""The Enhanced Prediction technique proposed in `Seven ways to improve
	example-based single image super resolution
	<https://arxiv.org/abs/1511.02228>`__
	
	Given an input image, we first rotate it :math:`0, 90, 180, 270`
	respectively. Then each intermediate image is vertically flipped.
	Finally, we obtain 8 images (hence, the name X8). After processing all
	8 images, we reverse the transformations and then average the outputs to
	create the final image.
	"""
	
	_to_tensor	 : bool = False
	_requires_post: bool = True
	
	def apply(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor | Sequence[torch.Tensor]:
		"""Generate 8 instances from :attr:`input` and then stack them together.
		to create a single :class:`torch.Tensor` or :class:`np.ndarray`.
		
		Args:
			input: An input image in :math:`[B, C, H, W]` format.
		
		Returns:
			An augmented image of shape :math:`[8 \times B, C, H, W]` format.
		"""
		r0    = input
		r90   = torch.rot90(input, k=1, dims=[-1, -2])
		r180  = torch.rot90(input, k=2, dims=[-1, -2])
		r270  = torch.rot90(input, k=3, dims=[-1, -2])
		r0f   = r0.flip(-2)
		r90f  = r90.flip(-2)
		r180f = r180.flip(-2)
		r270f = r270.flip(-2)
		if self._to_tensor:
			return torch.concat([r0, r90, r180, r270, r0f, r90f, r180f, r270f], dim=0)
		else:
			return [r0, r90, r180, r270, r0f, r90f, r180f, r270f]
	
	def postprocess(
		self,
		input : torch.Tensor,
		output: torch.Tensor | Sequence[torch.Tensor],
		*args, **kwargs
	) -> torch.Tensor:
		"""Reverse transforms 8 instances of from :attr:`output`.
		
		Args:
			input: An input image in :math:`[B, C, H, W]` format.
			output: Processed image in :math:`[8 \times B, C, H, W]` format.
		
		Returns:
			An augmented image of shape :math:`[B, C, H, W]` format.
		"""
		if isinstance(output, torch.Tensor):
			c0, c1, c2, c3, c4, c5, c6, c7 = torch.chunk(output, 8, 0)
		else:
			c0, c1, c2, c3, c4, c5, c6, c7 = output
		r0    = c0
		r90   = torch.rot90(c1, k=3, dims=[-1, -2])
		r180  = torch.rot90(c2, k=2, dims=[-1, -2])
		r270  = torch.rot90(c3, k=1, dims=[-1, -2])
		r0f   = c4.flip(-2)
		r90f  = torch.rot90(c5.flip(-2), k=3, dims=[-1, -2])
		r180f = torch.rot90(c6.flip(-2), k=2, dims=[-1, -2])
		r270f = torch.rot90(c7.flip(-2), k=1, dims=[-1, -2])
		cat   = torch.stack([r0, r90, r180, r270, r0f, r90f, r180f, r270f])
		mean  = torch.mean(cat, dim=0, keepdim=True)[0]
		return mean


class BlendInput(TestTimeAugment):
	"""Blend the output of the model with the input image to simulate the
	non-ideal ground-truth.
	"""
	
	_to_tensor	  : bool = False
	_requires_post: bool = True
	
	def apply(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		"""Do nothing and return the input as it is."""
		return input
	
	def postprocess(self, input: torch.Tensor, output: torch.Tensor, *args, **kwargs) -> torch.Tensor:
		"""
		
		Args:
			input: An input image in :math:`[B, C, H, W]` format.
			output: Processed image in :math:`[B, C, H, W]` format.
		
		Returns:
			An augmented image of shape :math:`[B, C, H, W]` format.
		"""
		input  = input.detach().cpu()
		output = output.detach().cpu()
		input  = core.to_image_nparray(input, False, True)
		output = skimage.util.img_as_ubyte(output)
		output = cv2.addWeighted(output, 0.9, input, 0.1, 0)
		return output
	
# endregion
