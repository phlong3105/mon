#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCEv2 models."""

from __future__ import annotations

__all__ = [
    "ZeroDCEv2",
]

from abc import ABC
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.enhance import base
from mon.vision.ml import layer, loss

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
        
        self.loss_spa     = loss.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp     = loss.ExposureControlLoss(
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
        loss         = self.spa_weight * loss_spa \
                       + self.exp_weight * loss_exp \
                       + self.col_weight * loss_col \
                       + self.tv_weight * loss_tv # \
                       # + self.channel_weight * loss_channel
        # print(loss_spa, loss_exp, loss_col, loss_tv, loss_channel)
        # print(loss_channel)
        return loss

# endregion


# region Model

@MODELS.register(name="zerodcev2")
class ZeroDCEv2(base.ImageEnhancementModel, ABC):
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
        config: Any = None,  # "zerodcev2.yaml",
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
    

@MODELS.register(name="zerodcev2-a")
class ZeroDCEv2A(ZeroDCEv2):
    """Zero-DCEv2-A model.
    
    See Also: :class:`mon.vision.enhance.zerodcev2.ZeroDCEv2`
    """
    
    def __init__(
        self,
        num_channels: int   = 32,
        scale_factor: float = 1.0,
        ratio       : float = 0.5,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        self.ratio        = ratio
        
        self.relu     = layer.ReLU(inplace=True)
        self.upsample = layer.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.conv0    = layer.DSConv2d(3, self.num_channels, 3, dw_stride=1, dw_padding=1)
        self.conv1    = layer.FFConv2dNormAct(self.num_channels * 2, self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv2    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv3    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv4    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv5    = layer.DSConv2d(self.num_channels,                          self.num_channels, 3, dw_stride=1, dw_padding=1)
        self.conv6    = layer.DSConv2d(self.num_channels + self.num_channels // 2, self.num_channels, 3, dw_stride=1, dw_padding=1)
        self.conv7    = layer.DSConv2d(self.num_channels + self.num_channels // 2, 3, 3, dw_stride=1, dw_padding=1)
        self.le_curve = layer.PixelwiseHigherOrderLECurve(n=8)
        
    def forward(
        self,
        input    : torch.Tensor,
        augment  : bool = False,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        if self.scale_factor == 1.0:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode="bilinear")
        
        x0         = self.relu(self.conv0(x_down))
        x1_l, x1_g = self.conv1([x0, x0])
        x2_l, x2_g = self.conv2([x1_l, x1_g])
        x3_l, x3_g = self.conv3([x2_l, x2_g])
        x4_l, x4_g = self.conv4([x3_l, x3_g])
        x1         = x1_l + x1_g
        x2         = x2_l + x2_g
        x3         = x3_l + x3_g
        x4         = x4_l + x4_g
        x5         = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6         = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        a          = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            a = a
        else:
            a = self.upsample(a)
        (a, y) = self.le_curve([a, x])
        return a, y


@MODELS.register(name="zerodcev2-b")
class ZeroDCEv2B(ZeroDCEv2):
    """Zero-DCEv2-B model.
    
    See Also: :class:`mon.vision.enhance.zerodcev2.ZeroDCEv2`
    """
    
    def __init__(
        self,
        num_channels: int   = 32,
        scale_factor: float = 1.0,
        ratio       : float = 0.5,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        self.ratio        = ratio
        
        self.relu     = layer.ReLU(inplace=True)
        self.upsample = layer.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.conv0    = layer.DSConv2d(3, self.num_channels, 3, dw_stride=1, dw_padding=1)
        self.conv1    = layer.FFConv2dNormAct(self.num_channels * 2, self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv2    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv3    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv4    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv5    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv6    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv7    = layer.FFConv2dNormAct(self.num_channels,     self.num_channels, 1, self.ratio, self.ratio, stride=1, padding=0, dilation=1, padding_mode="reflect", norm_layer=layer.BatchNorm2d, act_layer=layer.ReLU)
        self.conv8    = layer.DSConv2d(self.num_channels // 2, 3, 3, dw_stride=1, dw_padding=1)
        self.le_curve = layer.PixelwiseHigherOrderLECurve(n=8)
        
    def forward(
        self,
        input    : torch.Tensor,
        augment  : bool = False,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        if self.scale_factor == 1.0:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1/self.scale_factor, mode="bilinear")
        
        x0           = self.relu(self.conv0(x_down))
        x1_l, x1_g   = self.conv1([x0, x0])
        x2_l, x2_g   = self.conv2([x1_l, x1_g])
        x3_l, x3_g   = self.conv3([x2_l, x2_g])
        x4_l, x4_g   = self.conv4([x3_l, x3_g])
        x34_l, x34_g = x3_l + x4_l, x3_g + x4_g
        x5_l, x5_g   = self.conv5([x34_l, x34_g])
        x25_l, x25_g = x2_l + x5_l, x2_g + x5_g
        x6_l, x6_g   = self.conv6([x25_l, x25_g])
        x16_l, x16_g = x1_l + x6_l, x1_g + x6_g
        a_l, a_g     = self.conv7([x6_l, x6_g])
        a            = F.tanh(self.conv8(a_l + a_g))
        
        if self.scale_factor == 1:
            a = a
        else:
            a = self.upsample(a)
        (a, y) = self.le_curve([a, x])
        return a, y

# endregion
