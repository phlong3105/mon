#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements basic loss functions."""

from __future__ import annotations

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CTCLoss",
    "CharbonnierLoss",
    "CosineEmbeddingLoss",
    "CrossEntropyLoss",
    "ExtendedL1Loss",
    "ExtendedMAELoss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "L2Loss",
    "MAELoss",
    "MSELoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "NLLLoss2d",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SmoothMAELoss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

import torch
from torch import nn
from torch.nn import functional as F

from mon.globals import LOSSES, Reduction
from mon.nn.loss import base


# region General Loss

@LOSSES.register(name="charbonnier_loss")
class CharbonnierLoss(base.Loss):
    
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
        # loss = torch.sqrt((input - target) ** 2 + (self.eps * self.eps))
        diff = input - target
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="l1_loss")
class L1Loss(base.Loss):
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
        loss = F.l1_loss(
            input     = input,
            target    = target,
            reduction = self.reduction
        )
        return loss


@LOSSES.register(name="l2_loss")
class L2Loss(base.Loss):
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
        loss = F.mse_loss(
            input     = input,
            target    = target,
            reduction = self.reduction
        )
        return loss


@LOSSES.register(name="extended_l1_loss")
class ExtendedL1Loss(base.Loss):
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
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
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
