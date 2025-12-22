#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for basic loss functions.

This module provides various loss functions commonly used in training machine
learning models, particularly in computer vision tasks. Each loss function is
implemented as a class that can be instantiated and used to compute the loss
between predicted outputs and target values.
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
from torch.nn.modules.loss import *  # Expose all losses from ``torch.nn.modules.loss``
import torch.nn.functional as F

from .base import BaseLoss


# --- Basic Loss ---
class CharbonnierLoss(BaseLoss):
    """A Charbonnier loss function, a differentiable variant of L1 loss.
    
    Attributes:
        eps2 (float): Small constant for numerical stability.
    """
    
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """Initializes the CharbonnierLoss instance.
        
        Args:
            eps (float): Small constant for numerical stability. Defaults to 1e-6.
            reduction (str): Reduction method to apply to the loss. Can be one
                of "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps2 = eps ** 2
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the Charbonnier loss between input and target.
        
        Args:
            input (torch.Tensor): Input tensor (predictions) of shape (B, C, H, W)
                with pixel values in the range [0.0, 1.0].
            target (torch.Tensor): Target tensor (ground truth) of shape (B, C, H, W)
                with pixel values in the range [0.0, 1.0].
        
        Returns:
            torch.Tensor: Calculated Charbonnier loss.
        """
        loss = (F.mse_loss(input, target, reduction="none") + self.eps2) ** 0.5
        loss = self.reduce(loss=loss)
        return loss
    

class CosineSimilarityLoss(BaseLoss):
    """A Cosine Similarity loss function.
    
    Attributes:
        cos (torch.nn.CosineSimilarity): Cosine similarity module.
    """
    
    def __init__(self, dim: int = 1, eps: float = 1e-6, reduction: str = "mean"):
        """Initializes the CosineSimilarityLoss instance.
        
        Args:
            dim (int): Dimension along which to compute cosine similarity. Defaults to 1.
            eps (float): Small constant for numerical stability. Defaults to 1e-6.
            reduction (str): Reduction method to apply to the loss. Can be one
                of "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.cos = torch.nn.CosineSimilarity(dim=dim, eps=eps)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the Cosine Similarity loss between input and target.
        
        Args:
            input (torch.Tensor): Input tensor (predictions) of shape (B, C, H, W)
                with pixel values in the range [0.0, 1.0].
            target (torch.Tensor): Target tensor (ground truth) of shape (B, C, H, W)
                with pixel values in the range [0.0, 1.0].
        
        Returns:
            torch.Tensor: Calculated Cosine Similarity loss.
        """
        b, c, h, w = input.shape
        x    = input.permute(0, 2, 3, 1).view(-1, c)
        y    = target.permute(0, 2, 3, 1).view(-1, c)
        loss = 1.0 - self.cos(x, y).sum() / (1.0 * b * h * w)
        loss = self.reduce(loss=loss)
        return loss
        

class ExtendedL1Loss(BaseLoss):
    """An Extended L1 loss function that applies a mask to the input and target.
    
    Attributes:
        loss_l1 (L1Loss): L1 loss module.
    """
    
    def __init__(self, reduction: str = "mean"):
        """Initializes the ExtendedL1Loss instance.
        
        Args:
            reduction (str): Reduction method to apply to the loss. Can be one
                of "none", "mean", or "sum". Defaults to "mean".
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
        """Calculates the Extended L1 loss between input and target using a mask.
        
        Args:
            input (torch.Tensor): Input tensor (predictions) of shape (B, C, H, W)
                with pixel values in the range [0.0, 1.0].
            target (torch.Tensor): Target tensor (ground truth) of shape (B, C, H, W)
                with pixel values in the range [0.0, 1.0].
            mask (torch.Tensor): Mask tensor of shape (B, 1, H, W) with binary
                values indicating the regions to consider in the loss calculation.
                
        Returns:
            torch.Tensor: Calculated Extended L1 loss.
        """
        norm = self.loss_l1(mask, torch.zeros_like(mask))
        loss = self.loss_l1(mask * input, mask * target) / norm
        loss = self.reduce(loss=loss)
        return loss
