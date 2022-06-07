#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import functools
from typing import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from one.core import Tensors
from one.core import Weights

__all__ = [
    "convert_to_one_hot",
	"reduce_loss",
	"weight_reduce_loss",
	"weighted_loss",
	"weighted_sum",
]


# MARK: - Functional

def convert_to_one_hot(input: Tensor, num_classes: int) -> Tensor:
	"""This function converts target class indices to one-hot vectors, given
	the number of classes.

	Args:
		input (Tensor):
			Ground-truth label of the prediction with shape (N, 1).
		num_classes (int):
			Number of classes.

	Returns:
		one_hot_targets (Tensor):
			Processed loss values.
	"""
	if torch.max(input).item() >= num_classes:
		raise ValueError(f"Class Index must be < number of classes. "
		                 f"But got: {torch.max(input).item()} >= {num_classes}")
	
	one_hot_targets = torch.zeros(
		(input.shape[0], num_classes), dtype=torch.long, device=input.device
	)
	one_hot_targets.scatter_(1, input.long(), 1)
	return one_hot_targets


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
	"""Reduce loss as specified.
	
	Args:
		loss (Tensor):
			Elementwise loss image.
		reduction (str):
			One of: [`none`, `mean`, `sum`].
	
	Returns:
		loss (Tensor):
			Reduced loss image.
	"""
	reduction_enum = F._Reduction.get_enum(reduction)
	# NOTE: none: 0, elementwise_mean: 1, sum: 2
	if reduction_enum == 0:
		return loss
	elif reduction_enum == 1:
		return loss.mean()
	elif reduction_enum == 2:
		return loss.sum()


def weight_reduce_loss(
	loss     : Tensor,
	weight   : Optional[Weights] = None,
	reduction: str               = "mean",
) -> Tensor:
	"""Apply element-wise weight and reduce loss.
	
	Args:
		loss (Tensor):
			Element-wise loss.
		weight (Weights, optional):
			Element-wise weights.
		reduction (str):
			Same as built-in losses of PyTorch.
	
	Returns:
		loss (Tensor):
			Processed loss values.
	"""
	# If weight is specified, apply element-wise weight
	if weight is not None:
		if isinstance(weight, (tuple, list)):
			weight = torch.Tensor(weight).to(torch.float)
		if weight.dim() != loss.dim():
			raise ValueError(f"`weight` and `loss` must have the same ndim."
			                 f" But got: {weight.dim()} != {loss.dim()}")
		if not (weight.size()[1] == 1 or weight.size()[1] == loss.size()[1]):
			raise ValueError()
		loss *= weight
	# If weight is not specified or reduction is sum, just reduce the loss
	if weight is None or reduction == "sum":
		loss = reduce_loss(loss, reduction)
	# If reduction is mean, then compute mean over weight region
	elif reduction == "mean":
		if weight.size()[1] > 1:
			weight = weight.sum()
		else:
			weight = weight.sum() * loss.size()[1]
		loss = loss.sum() / weight
	
	return loss


def weighted_loss(loss_func: Callable):
	"""Create a weighted version of a given loss function. To use this
	decorator, the loss function must have the signature like
	`loss_func(pred, target, **kwargs)`.

	Function only needs to compute element-wise loss without any
	reduction. This decorator will add weight and reduction arguments to the
	function. Decorated function will have the signature like
	`loss_func(pred, target, weight=None, reduction='mean', avg_factor=None, **kwargs)`.
	
	Example:
	>>> import torch
	>>> @weighted_loss
	>>> def l1_loss(pred, target):
	>>>     return (pred - target).abs()
	>>> pred   = Tensor([0, 2, 3])
	>>> target = Tensor([1, 1, 1])
	>>> weight = Tensor([1, 0, 1])
	>>> l1_loss(pred, target)
	image(1.3333)
	>>> l1_loss(pred, target, weight)
	image(1.)
	>>> l1_loss(pred, target, reduction='none')
	image([1., 1., 2.])
	>>> l1_loss(pred, target, weight, avg_factor=2)
	image(1.5000)
	"""

	@functools.wraps(loss_func)
	def wrapper(
		input    : Tensor,
		target   : Optional[Tensor]  = None,
		weight   : Optional[Weights] = None,
		reduction: str               = "mean",
		**kwargs
	):
		"""
		
		Args:
			input (Tensor):
				Model prediction.
			target (Tensor):
				Target of each prediction
			weight (Weights, optional):
				Element-wise weights.
			reduction (str):
				Same as built-in losses of PyTorch.

		Returns:
			loss (Tensor):
				Processed loss values.
		"""
		# NOTE: Get element-wise loss
		loss = loss_func(input, target, **kwargs)
		loss = weight_reduce_loss(loss, weight, reduction)
		return loss

	return wrapper


def weighted_sum(input: Tensors, weight: Optional[Weights] = None) -> Tensor:
	"""Calculate the weighted sum of the given input.
	
	Args:
		input (Tensors):
			Can be a `Tensor` or a collection of `Tensor`.
		weight (Weights, optional):
			Weight for each element in `input`.

	Returns:
		sum (Tensor):
			The weighted sum.
	"""
	if isinstance(input, (list, tuple, dict)):
		input = list(input.values()) if isinstance(input, dict) else input
		
		if weight is None:
			weight = torch.ones(len(input)).to(torch.float)
		elif isinstance(weight, (tuple, list)):
			weight = torch.Tensor(weight).to(torch.float)
		elif len(input) != weight.shape[0]:
			raise ValueError(f"weight must have the same length as input."
			                 f"But got: {len(input)} != {weight.shape[0]}")
		if weight.device != input[0].device:
			weight = weight.to(input[0].device)
			
		return sum([i * w for i, w in zip(input, weight)])
	else:
		return input
