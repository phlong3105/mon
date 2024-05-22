#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements loss functions for classification tasks."""

from __future__ import annotations

__all__ = [
    "DiceLoss",
]

from typing import Literal

import torch

from mon.globals import LOSSES
from mon.nn.loss import base
from mon.nn.loss.base import reduce_loss


# region Utils

def dice_coefficient(
    input       : torch.Tensor,
    target      : torch.Tensor,
    reduce_batch: bool  = False,
    epsilon     : float = 1e-6
) -> torch.Tensor:
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch
    sum_dim  = (-1, -2) if input.dim() == 2 or not reduce_batch else (-1, -2, -3)
    inter    = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice     = (inter + epsilon) / (sets_sum + epsilon)
    return dice


def multiclass_dice_coefficient(
    input       : torch.Tensor,
    target      : torch.Tensor,
    reduce_batch: bool  = False,
    epsilon     : float = 1e-6
) -> torch.Tensor:
    # Average of Dice coefficient for all classes
    return dice_coefficient(
        input        = input.flatten(0 , 1),
        target       = target.flatten(0, 1),
        reduce_batch = reduce_batch,
        epsilon      = epsilon,
    )

# endregion


# region Loss

@LOSSES.register(name="dice_loss")
class DiceLoss(base.Loss):
    """Dice loss for binary or multiclass classification tasks."""
    
    def __init__(
        self,
        loss_weight : float = 1.0,
        reduction   : Literal["none", "mean", "sum"] = "mean",
        reduce_batch: bool  = True,
        multiclass  : bool  = False,
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.reduce_batch = reduce_batch
        self.multiclass   = multiclass
        self.fn = multiclass_dice_coefficient if multiclass else dice_coefficient
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input.shape == target.shape
        loss  = 1 - self.fn(input=input, target=target, reduce_batch=self.reduce_batch)
        loss  = reduce_loss(loss=loss, reduction=self.reduction)
        loss  = self.loss_weight * loss
        return loss

# endregion
