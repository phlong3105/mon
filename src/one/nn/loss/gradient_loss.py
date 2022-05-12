#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Gradient Loss.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from one.core import LOSSES
from one.core import Tensors
from one.core import Weights
from one.nn.loss.utils import weighted_sum

__all__ = [
    "gradient_loss",
    "GradientLoss",
]


# MARK: - Functional

def gradient_loss(
    input       : Tensors,
    input_weight: Optional[Weights] = None
) -> Tensor:
    """Measures the loss value.
    
    Args:
        input (Tensors):
            Either the prediction or the original input (unsupervised learning).
            Can be a single/collection of batches of shape [B, C, H, W].
        input_weight (Weights, optional):
            If `input` is a single batch, then set to `None` (or `1.0`).
            If `input` is a collection of batches, apply weighted sum on the
            returned loss values. Default: `None`.
    
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

    losses = []
    for inp in input:
        gradient_a_x = torch.abs(inp[:, :, :, :-1] - inp[:, :, :, 1:])
        gradient_a_y = torch.abs(inp[:, :, :-1, :] - inp[:, :, 1:, :])
        losses.append(torch.mean(gradient_a_x) + torch.mean(gradient_a_y))

    return weighted_sum(losses, input_weight)


# MARK: - Modules

@LOSSES.register(name="gradient_loss", force=True)
class GradientLoss(_Loss):
    """L1 loss on the gradient of the image.

    Attributes:
        name (str):
            Name of the loss. Default: `gradient_loss`.
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
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "gradient_loss"
        self.loss_weight = loss_weight

        if self.reduction not in self.reductions:
            raise ValueError(f"`reduction` must be one of: {self.reductions}. "
                             f"But got: {self.reduction}.")
        
    # MARK: Forward Pass
    
    def forward(
        self,
        input       : Tensors,
        input_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        """Measures the loss value.

        Args:
            input (Tensors):
                Either the prediction or the original input (unsupervised
                learning). Can be a single/collection of batches of shape
                [B, C, H, W].
            input_weight (Weights, optional):
                If `input` is a single batch, then set to `None` (or `1.0`).
                If `input` is a collection of batches, apply weighted sum on the
                returned loss values. Default: `None`.

        Returns:
        	loss (Tensor):
        	    Loss value.
        """
        return self.loss_weight * gradient_loss(input=input, input_weight=input_weight)
