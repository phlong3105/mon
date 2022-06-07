#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Perceptual loss.
"""

from __future__ import annotations

from typing import Optional

from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from one.core import LOSSES
from one.core import Tensors
from one.core import Weights
from one.nn.loss.utils import weighted_sum

__all__ = [
    "PerceptualLoss",
]


# MARK: - Modules

@LOSSES.register(name="perceptual_Loss")
class PerceptualLoss(_Loss):
    """Charbonnier Loss.
    
    Attributes:
        name (str):
            Name of the loss. Default: `charbonnier_loss`.
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
        vgg        : nn.Module,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "perceptual_Loss"
        self.loss_weight = loss_weight
        self.vgg         = vgg
        self.vgg.freeze()
        
        if self.reduction not in self.reductions:
            raise ValueError(f"`reduction` must be one of: {self.reductions}. "
                             f"But got: {self.reduction}.")

    # MARK: Forward Pass

    def forward(
        self,
        input       : Tensors,
        target      : Tensor,
        input_weight: Optional[Weights] = None,
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
             
        Returns:
            loss (Tensor):
                loss value.
        """
        if isinstance(input, Tensor):  # Single output
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        if not isinstance(input, (list, tuple)):
            raise ValueError(f"`input` must be a `list` or `tuple`. But got: {type(input)}.")

        if self.vgg.device != input[0].device:
            self.vgg = self.vgg.to(input[0].device)

        losses = []
        for i in input:
            input_features  = self.vgg.forward_features(i)
            target_features = self.vgg.forward_features(target)
            loss = [
                F.mse_loss(i_feature, t_feature)
                for i_feature, t_feature in zip(input_features, target_features)
            ]
            losses.append(sum(loss) / len(loss))
            
        return self.loss_weight * weighted_sum(losses, input_weight)
