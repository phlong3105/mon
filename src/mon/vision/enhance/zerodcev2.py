#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCEv2 models."""

from __future__ import annotations

__all__ = [

]

from typing import Any

import torch
from torch import nn

from mon.coreml import loss
from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.enhance import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Loss

class CombinedLoss(loss.Loss):
    """Loss = SpatialConsistencyLoss
              + ExposureControlLoss
              + ColorConstancyLoss
              + IlluminationSmoothnessLoss
              + ChannelConsistencyLoss
    """
    
    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tv_weight     : float = 1600.0,
        channel_weight: float = 5.0,
        reduction     : str   = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight     = spa_weight
        self.exp_weight     = exp_weight
        self.col_weight     = col_weight
        self.tv_weight      = tv_weight
        self.channel_weight = channel_weight
        
        self.loss_spa = loss.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = loss.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col     = loss.ColorConstancyLoss(reduction=reduction)
        self.loss_tv      = loss.IlluminationSmoothnessLoss(reduction=reduction)
        self.loss_channel = loss.ChannelConsistencyLoss(reduction=reduction)
    
    def __str__(self) -> str:
        return f"combined_loss"
    
    def forward(
        self,
        input : torch.Tensor | list[torch.Tensor],
        target: list[torch.Tensor],
        **_
    ) -> torch.Tensor:
        if isinstance(target, list | tuple):
            a       = target[-2]
            enhance = target[-1]
        else:
            raise TypeError()
        loss_spa     = self.loss_spa(input=enhance, target=input)
        loss_exp     = self.loss_exp(input=enhance)
        loss_col     = self.loss_col(input=enhance)
        loss_tv      = self.loss_tv(input=a)
        loss_channel = self.loss_channel(input=enhance, target=input)
        loss = self.spa_weight * loss_spa \
               + self.exp_weight * loss_exp \
               + self.col_weight * loss_col \
               + self.tv_weight * loss_tv \
               + self.channel_weight * loss_channel
        return loss

# endregion


# region Model

@MODELS.register(name="zerodcev2")
class ZeroDCEv2(base.ImageEnhancementModel):
    """Zero-DCEv2 model.
    
    Improvements over :class:`mon.vision.enhance.zerodce.ZeroDCE` model are:
        - Add FFC layers for global attention mechanism.
        - Dark Channel Prior (DCP), Bright Channel Prior (BCP), Contradict Channel Prior (CCP).
        - Components from NAFNet: SimpleGate, Simplified Channel Attention (SCA).
        
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {
        "backbone": {},
        "head"    : {},
    }
    
    def __init__(
        self,
        config: Any = "zerodcev2.yaml",
        loss  : Any = CombinedLoss(),
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
    
    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        pass
        
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape NCHW.
            target: A ground-truth of shape NCHW. Defaults to None.
            
        Return:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        loss = self.loss(input, pred) if self.loss else None
        return pred[-1], loss

# endregion
