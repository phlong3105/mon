#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic loss functions from PyTorch.

This module provides various loss functions commonly used in training machine
learning models.
"""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import *  # Expose all losses from ``torch.nn.modules.loss``

from .base import BaseLoss


# ==============================================================================
# BASIC & ATOMIC LOSSES
# ==============================================================================

# --- Regression ---
class CharbonnierLoss(BaseLoss):
    """A differentiable variant of L1 loss.
    
    Attributes:
        eps2 (float): Small constant for numerical stability.
    """
    
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """Initializes the CharbonnierLoss instance.
        
        Args:
            eps: Small constant for numerical stability. Defaults to 1e-6.
            reduction: Reduction method to apply to the loss. Can be one
                of "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps2 = eps ** 2
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the Charbonnier loss between input and target.
        
        Args:
           input: Input (predictions), formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target (ground truth), formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        loss = (F.mse_loss(input, target, reduction="none") + self.eps2) ** 0.5
        loss = self.reduce(loss=loss)
        return loss
    

# --- Relational & Vector ---
class CosineSimilarityLoss(BaseLoss):
    """Cosine Similarity loss function.
    
    Attributes:
        cos (nn.CosineSimilarity): Cosine similarity module.
    """
    
    def __init__(self, dim: int = 1, eps: float = 1e-6, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            dim: Dimension along which to compute cosine similarity. Defaults to 1.
            eps: Small constant for numerical stability. Defaults to 1e-6.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.cos = nn.CosineSimilarity(dim=dim, eps=eps)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the Cosine Similarity loss between input and target.
        
        Args:
           input: Input (predictions), formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target (ground truth), formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        b, c, h, w = input.shape
        x    = input.permute(0, 2, 3, 1).view(-1, c)
        y    = target.permute(0, 2, 3, 1).view(-1, c)
        loss = 1.0 - self.cos(x, y).sum() / (1.0 * b * h * w)
        loss = self.reduce(loss=loss)
        return loss


# --- Specialized & Masked ---
class ExtendedL1Loss(BaseLoss):
    """Extended L1 loss function that applies a mask to the input and target.
    
    Attributes:
        loss_l1 (L1Loss): L1 loss module.
    """
    
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.loss_l1 = L1Loss()
    
    # noinspection PyMethodOverriding
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        mask  : torch.Tensor
    ) -> torch.Tensor:
        """Calculate the Extended L1 loss between input and target using a mask.
        
        Args:
            input: Input (predictions), formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target (ground truth), formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            mask: Mask, formatted as a torch.Tensor of dimensions (B, 1, H, W)
                with binary values indicating the regions to consider in the
                loss calculation.
                
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        norm = self.loss_l1(mask, torch.zeros_like(mask))
        loss = self.loss_l1(mask * input, mask * target) / norm
        loss = self.reduce(loss=loss)
        return loss
