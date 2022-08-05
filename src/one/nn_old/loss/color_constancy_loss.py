#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Color Constancy Loss.

A color constancy loss to correct the potential color deviations in the enhanced
image and also build the relations among the three adjusted channels.

References:
    https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
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
    "color_constancy_loss",
    "elementwise_color_constancy_loss",
    "ColorConstancyLoss",
]


# MARK: - Functional

@weighted_loss
def elementwise_color_constancy_loss(
    input: Tensor, target: Optional[Tensor] = None
) -> Tensor:
    """Apply elementwise weight and reduce loss between a batch of input and
    a batch of target.
    
    Args:
        input (Tensor):
            Either the prediction or the original input (unsupervised learning)
            of shape [B, C, H, W].
        target (Tensor, optional):
            Ground-truth of shape [B, C, H, W]. Sometimes can be `None`
            (unsupervised learning). Default: `None`.
            
    Returns:
        k (Tensor):
            Single reduced loss value.
    """
    x          = input
    mean_rgb   = torch.mean(x, [2, 3], keepdim=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    d_rg       = torch.pow(mr - mg, 2)
    d_rb       = torch.pow(mr - mb, 2)
    d_gb       = torch.pow(mb - mg, 2)
    k = torch.pow(
        torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5
    )
    return k
    

def color_constancy_loss(
    input             : Tensors,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
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
            Loss values.
    """
    if isinstance(input, Tensor):  # Single output
        input = [input]
    elif isinstance(input, dict):
        input = list(input.values())
    if not isinstance(input, (list, tuple)):
        raise ValueError("`input` must be a `list` or `tuple`. "
                         f"But got: {type(input)}.")
    
    losses = [
        elementwise_color_constancy_loss(
            input     = i,
            target    = None,
            weight    = elementwise_weight,
            reduction = reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)


# MARK: - Modules

@LOSSES.register(name="color_constancy_loss")
class ColorConstancyLoss(_Loss):
    """Exposure Control Loss.

    Attributes:
        name (str):
            Name of the loss. Default: `color_constancy_loss`.
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
        self.name        = "color_constancy_loss"
        self.loss_weight = loss_weight
        
        if self.reduction not in self.reductions:
            raise ValueError(f"`reduction` must be one of: {self.reductions}. "
                             f"But got: {self.reduction}.")
 
    # MARK: Forward Pass
    
    def forward(
        self,
        input             : Tensors,
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
            input_weight (Weights, optional):
                If `input` is a single batch, then set to `None` (or `1.0`).
                If `input` is a collection of batches, apply weighted sum on the
                returned loss values. Default: `None`.
            elementwise_weight (Weights, optional):
                Elementwise weights for each `input` batch of shape [B, C, H, W].
                Default: `None`.

        Returns:
            loss (Tensor):
                Loss values.
        """
        return self.loss_weight * color_constancy_loss(
            input              = input,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )
