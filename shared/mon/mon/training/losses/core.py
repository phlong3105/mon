#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registers ``torch``'s losses and implements basic loss functions."""

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CTCLoss",
    "CharbonnierLoss",
    "CosineEmbeddingLoss",
    "CosineSimilarityLoss",
    "CrossEntropyLoss",
    "ExtendedL1Loss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MSELoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "NLLLoss2d",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

import torch
from torch.nn.modules.loss import *  # Expose all losses from ``torch.nn.modules.loss``
import torch.nn.functional as F

from .base import BaseLoss


# ----- Basic Loss -----
class CharbonnierLoss(BaseLoss):
    """Computes the Charbonnier loss between input and target tensors.

    Args:
        eps: Small constant for numerical stability. Default: ``1e-3``.
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default: ``"mean"``.`.
    """
    
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.eps2 = eps ** 2
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (F.mse_loss(input, target, reduction="none") + self.eps2) ** 0.5
        loss = self.reduce(loss=loss)
        return loss
    

class CosineSimilarityLoss(BaseLoss):
    """Computes cosine similarity loss between input and target tensors.

    Args:
        dim: Dimension for cosine similarity. Default: ``1``.
        eps: Small constant for numerical stability. Default: ``1e-6``.
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default: ``"mean"``.
    """
    
    def __init__(self, dim: int = 1, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.cos = torch.nn.CosineSimilarity(dim=dim, eps=eps)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        x    = input.permute(0, 2, 3, 1).view(-1, c)
        y    = target.permute(0, 2, 3, 1).view(-1, c)
        loss = 1.0 - self.cos(x, y).sum() / (1.0 * b * h * w)
        loss = self.reduce(loss=loss)
        return loss
        

class ExtendedL1Loss(BaseLoss):
    """Computes extended L1 loss with mask normalization.

    Args:
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default: ``"mean"``.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.loss_l1 = L1Loss()
    
    # noinspection PyMethodOverriding
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        norm = self.loss_l1(mask, torch.zeros_like(mask))
        loss = self.loss_l1(mask * input, mask * target) / norm
        loss = self.reduce(loss=loss)
        return loss
