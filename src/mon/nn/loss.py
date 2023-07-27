#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements loss functions for training machine learning models.
"""

from __future__ import annotations

__all__ = [
    "BCELoss", "BCEWithLogitsLoss", "CTCLoss", "CharbonnierLoss",
    "CosineEmbeddingLoss", "CrossEntropyLoss", "ExtendedL1Loss",
    "ExtendedMAELoss", "GaussianNLLLoss", "HingeEmbeddingLoss", "HuberLoss",
    "KLDivLoss", "L1Loss", "L2Loss", "Loss", "MAELoss", "MSELoss",
    "MarginRankingLoss", "MultiLabelMarginLoss", "MultiLabelSoftMarginLoss",
    "MultiMarginLoss", "NLLLoss", "NLLLoss2d", "PoissonNLLLoss", "reduce_loss",
    "SmoothL1Loss", "SmoothMAELoss", "SoftMarginLoss", "TripletMarginLoss",
    "TripletMarginWithDistanceLoss", "WeightedLoss",
]

from abc import ABC, abstractmethod

import humps
import torch
from torch import nn
from torch.nn import functional
from torch.nn.modules.loss import _Loss

from mon.globals import LOSSES, Reduction


# region Base Loss

def reduce_loss(
    loss     : torch.Tensor,
    weight   : torch.Tensor | None = None,
    reduction: Reduction | str     = "mean",
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
    
    # If your loss function only supports some custom reduction. Consider
    # overwriting this value.
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


# region General Loss

@LOSSES.register(name="charbonnier_loss")
class CharbonnierLoss(Loss):
    
    def __init__(self, reduction: Reduction | str = "mean", eps: float = 1e-3):
        super().__init__(reduction=reduction)
        self.eps = eps
    
    def __str__(self) -> str:
        return f"charbonnier_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.sqrt((input - target) ** 2 + (self.eps * self.eps))
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="l1_loss")
class L1Loss(Loss):
    """L1 Loss or Mean Absolute Error."""
    
    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
    
    def __str__(self) -> str:
        return f"l1_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        loss = functional.l1_loss(
            input     = input,
            target    = target,
            reduction = self.reduction
        )
        return loss


@LOSSES.register(name="l2_loss")
class L2Loss(Loss):
    """L2 Loss or Mean Squared Error."""
    
    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
    
    def __str__(self) -> str:
        return f"l2_loss"
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        loss = functional.mse_loss(
            input     = input,
            target    = target,
            reduction = self.reduction
        )
        return loss


@LOSSES.register(name="extended_l1_loss")
class ExtendedL1Loss(Loss):
    """Also pays attention to the mask, to be relative to its size."""
    
    def __init__(self, reduction: Reduction | str = "mean"):
        super().__init__(reduction=reduction)
        self.l1 = L1Loss()
    
    def __str__(self) -> str:
        return f"extended_l1_loss"
    
    # noinspection PyMethodOverriding
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        mask  : torch.Tensor,
    ) -> torch.Tensor:
        norm = self.l1(mask, torch.zeros(mask.shape).to(mask.device))
        loss = self.l1(mask * input, mask * target) / norm
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return loss


BCELoss                       = nn.BCELoss
BCEWithLogitsLoss             = nn.BCEWithLogitsLoss
CosineEmbeddingLoss           = nn.CosineEmbeddingLoss
CrossEntropyLoss              = nn.CrossEntropyLoss
CTCLoss                       = nn.CTCLoss
GaussianNLLLoss               = nn.GaussianNLLLoss
HingeEmbeddingLoss            = nn.HingeEmbeddingLoss
HuberLoss                     = nn.HuberLoss
KLDivLoss                     = nn.KLDivLoss
MAELoss                       = L1Loss
MarginRankingLoss             = nn.MarginRankingLoss
MSELoss                       = L2Loss
MultiLabelMarginLoss          = nn.MultiLabelMarginLoss
MultiLabelSoftMarginLoss      = nn.MultiLabelSoftMarginLoss
MultiMarginLoss               = nn.MultiMarginLoss
NLLLoss                       = nn.NLLLoss
NLLLoss2d                     = nn.NLLLoss2d
ExtendedMAELoss               = ExtendedL1Loss
PoissonNLLLoss                = nn.PoissonNLLLoss
SmoothL1Loss                  = nn.SmoothL1Loss
SmoothMAELoss                 = SmoothL1Loss
SoftMarginLoss                = nn.SoftMarginLoss
TripletMarginLoss             = nn.TripletMarginLoss
TripletMarginWithDistanceLoss = nn.TripletMarginWithDistanceLoss

LOSSES.register(name="bce_loss"                         , module=BCELoss)
LOSSES.register(name="bce_with_logits_loss"             , module=BCEWithLogitsLoss)
LOSSES.register(name="cosine_embedding_loss"            , module=CosineEmbeddingLoss)
LOSSES.register(name="cross_entropy_loss"               , module=CrossEntropyLoss)
LOSSES.register(name="ctc_loss"                         , module=CTCLoss)
LOSSES.register(name="gaussian_nll_loss"                , module=GaussianNLLLoss)
LOSSES.register(name="hinge_embedding_loss"             , module=HingeEmbeddingLoss)
LOSSES.register(name="huber_loss"                       , module=HuberLoss)
LOSSES.register(name="kl_div_loss"                      , module=KLDivLoss)
LOSSES.register(name="mae_loss"                         , module=MAELoss)
LOSSES.register(name="margin_ranking_loss"              , module=MarginRankingLoss)
LOSSES.register(name="mae_loss"                         , module=MSELoss)
LOSSES.register(name="multi_label_margin_loss"          , module=MultiLabelMarginLoss)
LOSSES.register(name="multi_label_soft_margin_loss"     , module=MultiLabelSoftMarginLoss)
LOSSES.register(name="multi_margin_loss"                , module=MultiMarginLoss)
LOSSES.register(name="nll_loss"                         , module=NLLLoss)
LOSSES.register(name="nll_loss2d"                       , module=NLLLoss2d)
LOSSES.register(name="extended_mae_loss"                , module=ExtendedMAELoss)
LOSSES.register(name="poisson_nll_loss"                 , module=PoissonNLLLoss)
LOSSES.register(name="smooth_l1_loss"                   , module=SmoothL1Loss)
LOSSES.register(name="smooth_mae_loss"                  , module=SmoothMAELoss)
LOSSES.register(name="soft_margin_loss"                 , module=SoftMarginLoss)
LOSSES.register(name="triplet_margin_loss"              , module=TripletMarginLoss)
LOSSES.register(name="triplet_margin_with_distance_Loss", module=TripletMarginWithDistanceLoss)

# endregion
