#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Charbonnier Loss.
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

__all__ = [
    "charbonnier_loss",
    "elementwise_charbonnier_loss",
    "CharbonnierLoss",
]


# MARK: - Functional

@weighted_loss
def elementwise_charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
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
    	eps (float):
    	    Small value for numerically stability when dividing.
            Default: `1e-3`.
            
    Returns:
    	loss (Tensor):
    	    Single reduced loss value.
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


def charbonnier_loss(
    input             : Tensors,
    target            : Tensor,
    eps               : float             = 1e-3,
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
        eps (float):
    	    Small value for numerically stability when dividing.
            Default: `1e-3`.
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
        elementwise_charbonnier_loss(
            input     = i,
            target    = target,
            eps       = eps,
            weight    = elementwise_weight,
            reduction = reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)


# MARK: - Modules

@LOSSES.register(name="charbonnier_loss")
class CharbonnierLoss(_Loss):
    """Charbonnier Loss.
    
    Attributes:
        name (str):
            Name of the loss. Default: `charbonnier_loss`.
        eps (float):
            Small value for numerically stability when dividing.
            Default: `1e-3`.
        loss_weight (Weights, optional):
			Some loss function is the sum of other loss functions.
			This provides weight for each loss component. Default: `1.0`.
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
        eps        : float             = 1e-3,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "charbonnier_loss"
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
                Either the prediction or the original input (unsupervised
                learning). Can be a single/collection of batches of shape
                [B, C, H, W].
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
                loss value.
        """
        # diff = input - target
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        # loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return self.loss_weight * charbonnier_loss(
			input              = input,
            target             = target,
            eps                = self.eps,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
		)
