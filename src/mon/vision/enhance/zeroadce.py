#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCE models."""

from __future__ import annotations

__all__ = [
    "ZeroADCE", "ZeroADCEJIT",
]

from functools import partial
from typing import Any

import torch

from mon.coreml import loss as mloss, layer as mlayer
from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.enhance import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Loss

class CombinedLoss(mloss.Loss):
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
        edge_weight   : float = 5.0,
        reduction     : str   = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight     = spa_weight
        self.exp_weight     = exp_weight
        self.col_weight     = col_weight
        self.tv_weight      = tv_weight
        self.channel_weight = channel_weight
        self.edge_weight    = edge_weight
        
        self.loss_spa = mloss.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = mloss.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col     = mloss.ColorConstancyLoss(reduction=reduction)
        self.loss_tv      = mloss.IlluminationSmoothnessLoss(reduction=reduction)
        self.loss_channel = mloss.ChannelConsistencyLoss(reduction=reduction)
        self.loss_edge    = mloss.EdgeLoss(reduction=reduction)
    
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
        loss_edge    = self.loss_edge(input=enhance, target=input)
        loss = self.spa_weight * loss_spa \
               + self.exp_weight * loss_exp \
               + self.col_weight * loss_col \
               + self.tv_weight * loss_tv \
               + self.channel_weight * loss_channel \
               + self.edge_weight * loss_edge
        return loss

# endregion


# region Model

@MODELS.register(name="zeroadce")
class ZeroADCE(base.ImageEnhancementModel):
    """Zero-ADCE (Zero-Reference Attention Deep Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(
        self,
        config: Any = "zeroadce-a.yaml",
        loss  : Any = CombinedLoss(),
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
    
    def init_weights(self, m: torch.nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
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
        pred  = self.forward(input=input, *args, **kwargs)
        loss  = self.loss(input, pred) if self.loss else None
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
    
    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in self.model.modules():
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss


@MODELS.register(name="zeroadce-jit")
class ZeroADCEJIT(base.ImageEnhancementModel):
    """Zero-ADCE (Zero-Reference Attention Deep Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(
        self,
        config: Any = "zeroadce-a",
        loss  : Any = CombinedLoss(),
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
        if config in ["zero-adce-a"]:
            conv       = partial(mlayer.ABSConv2dS, act2=mlayer.HalfInstanceNorm2d)
            final_conv = mlayer.Conv2d
        elif config in ["zero-adce-b"]:
            conv       = partial(mlayer.ABSConv2dS, ac1=mlayer.HalfInstanceNorm2d, act2=mlayer.HalfInstanceNorm2d)
            final_conv = mlayer.Conv2d
        elif config in ["zero-adce-c"]:
            conv       = partial(mlayer.ABSConv2dS, ac1=mlayer.HalfInstanceNorm2d, act2=mlayer.HalfInstanceNorm2d)
            final_conv = partial(mlayer.ABSConv2dS, ac1=mlayer.HalfInstanceNorm2d, act2=mlayer.HalfInstanceNorm2d)
        else:
            raise ValueError(
                f"config must be one of: `zero-adce-[a, b, c, d, e]`. "
                f"But got: {config}."
            )
           
        self.relu  = mlayer.ReLU(inplace=True)
        self.conv1 = conv(
            in_channels  = 3,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv2 = conv(
            in_channels  = 32,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv3 = mlayer.ABSConv2dS(
            in_channels  = 32,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv4 = conv(
            in_channels  = 32,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv5 = conv(
            in_channels  = 32 * 2,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv6 = conv(
            in_channels  = 32 * 2,
            out_channels = 32,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        self.conv7 = final_conv(
            in_channels  = 32 * 2,
            out_channels = 3,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            dilation     = 1,
            groups       = 1,
            bias         = True,
            padding_mode = "zeros",
            device       = None,
            dtype        = None,
        )
        # Load pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: torch.nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
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
        pred  = self.forward(input=input, *args, **kwargs)
        loss  = self.loss(input, pred) if self.loss else None
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
    
    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in self.model.modules():
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss
    
# endregion
