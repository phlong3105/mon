#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perceptual loss.
"""

from __future__ import annotations

from typing import Optional

from torch import nn
from torch import Tensor
from torch.nn.modules.loss import _Loss

from one.core import LOSSES
from one.core import Tensors
from one.core import Weights
from one.nn.loss.mae_loss import L1Loss
from one.nn.loss.perceptual_loss import PerceptualLoss

__all__ = [
    "PerceptualL1Loss",
]


# MARK: - Modules

@LOSSES.register(name="perceptual_l1_Loss")
class PerceptualL1Loss(_Loss):
    """Loss = weights[0] * Perceptual Loss + weights[1] * L1 Loss.
    
    Attributes:
        name (str):
            Name of the loss. Default: `perceptual_l1_Loss`.
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
    
    def __init__(
        self,
        vgg        : nn.Module,
        loss_weight: Optional[Weights] = Tensor([1.0, 1.0]),
        reduction  : str               = "mean",
    ):
        super().__init__(reduction=reduction)
        self.name        = "perceptual_l1_Loss"
        self.loss_weight = loss_weight
        self.per_loss    = PerceptualLoss(vgg=vgg, reduction=reduction)
        self.l1_loss     = L1Loss(reduction=reduction)
        self.layer_name_mapping = {
            "3" : "relu1_2",
            "8" : "relu2_2",
            "15": "relu3_3"
        }
        
        if self.loss_weight is None:
            self.loss_weight = Tensor([1.0, 1.0])
        elif len(self.loss_weight) != 2:
            raise ValueError(f"Length of `loss_weight` must be 2. "
                             f"But got: {len(self.loss_weight)}." )
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
        l0 = self.loss_weight[0] * self.per_loss(input, target, input_weight, elementwise_weight)
        l1 = self.loss_weight[1] * self.l1_loss(input, target, input_weight, elementwise_weight)
        return l0 + l1
