#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for all loss functions, and the
corresponding helper functions.
"""

from __future__ import annotations

__all__ = [
    "Loss",
    "WeightedLoss",
    "reduce_loss",
]

from abc import ABC, abstractmethod

import humps
import torch
from torch.nn.modules.loss import _Loss

from mon.globals import Reduction


# region Base Loss

def reduce_loss(
    loss     : torch.Tensor,
    weight   : torch.Tensor | None = None,
    reduction: Reduction    | str  = "mean",
) -> torch.Tensor:
    """Reduces the loss tensor.

    Args:
        loss: Elementwise loss tensor.
        reduction: Reduction value to use.
        weight: Element-wise weights. Default: ``None``.
        
    Returns:
        Reduced loss.
    """
    reduction = Reduction.from_value(reduction)
    if reduction == Reduction.NONE:
        return loss
    if reduction == Reduction.MEAN:
        return torch.mean(loss)
    if reduction == Reduction.SUM:
        return torch.sum(loss)
    if reduction == Reduction.WEIGHTED_SUM:
        if weight is None:
            return torch.sum(loss)
        else:
            if weight.devices != loss.device:
                weight.to(loss.device)
            if weight.ndim != loss.ndim:
                raise ValueError(
                    f"'weight' and 'loss' must have the same ndim."
                    f" But got: {weight.dim()} != {loss.dim()}"
                )
            loss *= weight
            return torch.sum(loss)


class Loss(_Loss, ABC):
    """The base class for all loss functions.
    
    Args:
        reduction: Specifies the reduction to apply to the output.
            One of: ``'none'``, ``'mean'``, ``'sum'``, or ``'weighted_sum'``.
            
            - ``None``: No reduction will be applied.
            - ``'mean'``: The sum of the output will be divided by the number of
              elements in the output.
            - ``'sum'``: The output will be summed.
            - Default: ``'mean'``.
    """
    
    # If your loss function only supports some custom reduction.
    # Consider overwriting this value.
    reductions = ["none", "mean", "sum", "weighted_sum"]
    
    def __init__(self, reduction: Reduction | str = "mean"):
        reduction = str(Reduction.from_value(value=reduction).value)
        super().__init__(reduction=reduction)
        
        if self.reduction not in self.reductions:
            raise ValueError(
                f"'reduction' must be one of: {self.reductions}. "
                f"But got: {self.reduction}."
            )
    
    def __str__(self):
        return humps.depascalize(self.__class__.__name__).lower()
    
    @abstractmethod
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pass


class WeightedLoss(Loss, ABC):
    """The base class for weighted loss functions."""
    
    def __init__(
        self,
        weight   : torch.Tensor | None = None,
        reduction: Reduction | str     = "mean",
    ):
        super().__init__(reduction=reduction)
        self.register_buffer("weight", weight)
        self.weight: torch.Tensor | None

# endregion
