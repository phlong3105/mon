#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base loss."""

__all__ = [
    "BaseLoss",
]

import abc
from typing import Literal

import torch
from torch.nn.modules.loss import _Loss

from mon.core.utils import depascalize


# ----- Base Loss -----
class BaseLoss(_Loss, abc.ABC):
    """The base class for all loss functions.
    
    Args:
        reduction: Specifies the reduction to apply to the output. One of:
            - ``'none'``: No reduction will be applied.
            - ``'mean'``: The sum of the output will be divided by the number of
                elements in the output.
            - ``'sum'``: The output will be summed.
            - Default: ``'mean'``.
    """
    
    reductions = ["none", "mean", "sum"]
    
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__(reduction=reduction)
        if self.reduction not in self.reductions:
            raise ValueError(f"``reduction`` must be one of: {self.reductions}, got {reduction}.")
        
    def __str__(self):
        return depascalize(self.__class__.__name__).lower()
    
    @abc.abstractmethod
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between input and target.
    
        Args:
            input: Input data as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in range :math:`[0.0, 1.0]`.
            target: Target data as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in range :math:`[0.0, 1.0]`.
    
        Returns:
            Loss value as a ``torch.Tensor``.
        """
        pass
    
    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduces the loss tensor.
    
        Args:
            loss: Elementwise loss tensor as a ``torch.Tensor``.
    
        Returns:
            Reduced loss valued as a ``torch.Tensor``.
        """
        return {"mean": torch.mean, "sum": torch.sum, "none": lambda x: x}[self.reduction](loss)
