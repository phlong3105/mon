#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create a criterion that computes a loss based on the SSIM measurement.

https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/ssim.html
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from one.core import LOSSES
from one.core import Tensors
from one.core import Weights
from one.nn.loss.utils import weighted_loss
from one.nn.loss.utils import weighted_sum
from one.nn.metric import ssim

__all__ = [
    "elementwise_ssim_loss",
    "ssim_loss",
    "SSIMLoss",
]


# MARK: - Functional

@weighted_loss
def elementwise_ssim_loss(
    input      : Tensor,
    target     : Tensor,
    window_size: int,
    max_val    : float = 1.0,
    eps        : float = 1e-12
) -> Tensor:
    """Apply elementwise weight and reduce loss between a batch of input and
    a batch of target.
    
    Args:
    	input (Tensor):
    	    Either the prediction or the original input (unsupervised learning)
    	    of shape [B, C, H, W]
    	target (Tensor):
    	    Ground-truth of shape [B, C, H, W]. Sometimes can be `None`
    	    (unsupervised learning).
    	window_size (int):
    	    Size of the gaussian kernel to smooth the images.
    	max_val (float):
    	    Dynamic range of the images. Default: `1.0`.
    	eps (float):
    	    Small value for numerically stability when dividing.
            Default: `1e-12`.
            
    Returns:
    	loss (Tensor):
    	    Single reduced loss value.
    """
    # Compute the ssim map
    ssim_map = ssim(input, target, window_size, max_val, eps)
    # Compute and reduce the loss
    return torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)


def ssim_loss(
    input             : Tensors,
    target            : Tensor,
    window_size       : int,
    max_val           : float             = 1.0,
    eps               : float             = 1e-12,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean"
) -> Tensor:
    """Measures the loss value.

    Args:
        input (Tensors):
            Either the prediction or the original input (unsupervised learning).
            Can be a single/collection of batches of shape [B, C, H, W].
        target (Tensor):
            Ground-truth of shape [B, C, H, W]. Sometimes can be `None`
    	    (unsupervised learning).
        window_size (int):
    	    Size of the gaussian kernel to smooth the images.
    	max_val (float):
    	    Dynamic range of the images. Default: `1.0`.
    	eps (float):
    	    Small value for numerically stability when dividing.
            Default: `1e-12`.
        input_weight (Weights, optional):
            If `input` is a single batch, then set to `None` (or `1.0`).
            If `input` is a collection of batches, apply weighted sum on the
            returned loss values. Default: `None`.
        elementwise_weight (Weights, optional):
            Elementwise weights for each `input` batch of shape [B, C, H, W].
            Default: `None`.
        reduction (str):
            Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`].
            - `none`: No reduction will be applied.
            - `mean`: The sum of the output will be divided by the number of
                      elements in the output.
            - `sum`: The output will be summed.
            Default: `mean`.
    
    Returns:
    	loss (Tensor):
    	    Loss value.
    """
    if isinstance(input, Tensor):  # Single output
        input = [input]
    elif isinstance(input, dict):
        input = list(input.values())
    if not isinstance(input, (list, tuple)):
        raise ValueError(f"`input` must be a `list` or `tuple`. But got: {type(input)}.")

    losses = [
        elementwise_ssim_loss(
            input=i, target=target, window_size=window_size, max_val=max_val,
            eps=eps, weight=elementwise_weight, reduction=reduction,
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)


# MARK: - Modules

@LOSSES.register(name="ssim_loss")
class SSIMLoss(_Loss):
    """Create a criterion that computes a loss based on the SSIM measurement.
    Supports both single output and multi-outputs input.

    Attributes:
        name (str):
            Name of the loss. Default: `ssim_loss`.
        max_val (float):
            Dynamic range of the images. Default: `1.0`.
        eps (float):
            Small value for numerically stability when dividing.
            Default: `1e-12`.
        loss_weight (Weights, optional):
			Some loss function is the sum of other loss functions.
			This provides weight for each loss component. Default: `1.0`.
        reduction (str):
            reduction (str):
            Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`].
            - `none`: No reduction will be applied.
            - `mean`: The sum of the output will be divided by the number of
                      elements in the output.
            - `sum`: The output will be summed.
            Default: `mean`.
    """

    reductions = ["none", "mean", "sum"]

    # MARK: Magic Functions
    
    def __init__(
        self,
        window_size: int,
        max_val    : float             = 1.0,
        eps        : float             = 1e-12,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean",
    ):
        super().__init__(reduction=reduction)
        self.name        = "ssim_loss"
        self.window_size = window_size
        self.max_val     = max_val
        self.eps         = eps
        self.loss_weight = loss_weight

        if self.reduction not in self.reductions:
            raise ValueError(f"`reduction` must be one of: {self.reductions}. "
                             f"But got: {self.reduction}.")
        
    # MARK: Forward Pass
    
    def forward(
        self,
        input             : Tensors,
        target            : Tensor,
        input_weight      : Optional[Weights] = None,
        elementwise_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        """Measures the loss value.

        Args:
            input (Tensors):
                Either the prediction or the original input (unsupervised learning).
                Can be a single/collection of batches of shape [B, C, H, W].
            target (Tensor):
                Ground-truth of shape [B, C, H, W]. Sometimes can be `None`
                (unsupervised learning).
            input_weight (Weights, optional):
                If `input` is a single batch, then set to `None` (or `1.0`).
                If `input` is a collection of batches, apply weighted sum on the
                returned loss values. Default: `None`.
            elementwise_weight (Weights, optional):
                Elementwise weights for each `input` batch of shape [B, C, H, W].
                Default: `None`.
      
        Returns:
            loss (Tensor):
                Loss value.
    """
        return self.loss_weight * ssim_loss(
            input              = input,
            target             = target,
            window_size        = self.window_size,
            max_val            = self.max_val,
            eps                = self.eps,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
