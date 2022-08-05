#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MSE loss.
Modified based on BasicSR: https://github.com/xinntao/BasicSR
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from one.core import LOSSES
from one.core import Tensors
from one.core import Weights
from one.nn.loss.utils import weighted_loss
from one.nn.loss.utils import weighted_sum

__all__ = [
    "elementwise_l2_loss",
    "elementwise_mse_loss",
    "l2_loss",
    "mse_loss",
    "L2Loss",
    "MSELoss",
]


# MARK: - Functional

@weighted_loss
def elementwise_mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """Apply elementwise weight and reduce loss between a batch of input and
    a batch of target.
    
    Args:
    	input (Tensor):
    	    Either the prediction or the original input (unsupervised learning)
    	    of shape [B, C, H, W]
    	target (Tensor):
    	    Ground-truth of shape [B, C, H, W]. Sometimes can be `None`
    	    (unsupervised learning).
     
    Returns:
    	loss (Tensor):
    	    Single reduced loss value.
    """
    return F.mse_loss(input, target, reduction="none")


def mse_loss(
    input             : Tensors,
    target            : Tensor,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
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
        elementwise_mse_loss(
            input=i, target=target, weight=elementwise_weight,
            reduction=reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)


# MARK: - Modules

@LOSSES.register(name="mse_loss", force=True)
class MSELoss(_Loss):
    """MSE (Mean Squared Error or L2) loss.
    
    Attributes:
        name (str):
            Name of the loss. Default: `mse_loss`.
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
        self.name        = "mse_loss"
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
        return self.loss_weight * mse_loss(
            input              = input,
            target             = target,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )


# MARK: - Alias

elementwise_l2_loss = elementwise_mse_loss
l2_loss             = mse_loss
L2Loss              = MSELoss


# MARK: - Register

LOSSES.register(name="l2_loss", module=L2Loss)
