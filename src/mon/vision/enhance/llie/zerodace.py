#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DACE models."""

from __future__ import annotations

__all__ = ["ZeroDACE"]

from typing import Any

import torch
import kornia
from mon import nn
from mon.core import math, pathlib
from mon.globals import LAYERS, MODELS
from mon.nn import functional as F
from mon.vision import loss
from mon.vision.enhance.llie import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Module

@LAYERS.register()
class HalfInstanceNorm2dBlock(nn.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        num_features       : int,
        relu_slope         : float = 0.2,
        eps                : float = 1e-5,
        momentum           : float = 0.1,
        affine             : bool  = True,
        track_running_stats: bool  = False,
        device             : Any   = None,
        dtype              : Any   = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels  = num_features,
            out_channels = num_features,
            kernel_size  = 3,
            padding      = 1,
            bias         = True,
        )
        self.conv2 = nn.Conv2d(
            in_channels  = num_features,
            out_channels = num_features,
            kernel_size  = 3,
            padding      = 1,
            bias         = True,
        )
        self.norm = nn.InstanceNorm2d(
            num_features        = math.ceil(num_features / 2),
            eps                 = eps,
            momentum            = momentum,
            affine              = affine,
            track_running_stats = track_running_stats,
            device              = device,
            dtype               = dtype,
        )
        self.identity = nn.Conv2d(
            in_channels  = num_features,
            out_channels = num_features,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
        )
        self.act1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.act2 = nn.LeakyReLU(relu_slope, inplace=False)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.conv1(x)
        if y.dim() == 3:
            y1, y2 = torch.chunk(y, 2, dim=0)
            y      = torch.cat([self.norm(y1), y2], dim=0)
        elif x.dim() == 4:
            y1, y2 = torch.chunk(y, 2, dim=1)
            y      = torch.cat([self.norm(y1), y2], dim=1)
        else:
            raise ValueError
        y  = self.act1(y)
        y  = self.act2(self.conv2(y))
        y += self.identity(x)
        return y
   
# endregion


# region Loss

class CombinedLoss(nn.Loss):
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
        
        self.loss_spa = loss.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = loss.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col     = loss.ColorConstancyLoss(reduction=reduction)
        self.loss_tv      = loss.IlluminationSmoothnessLoss(reduction=reduction)
        self.loss_channel = loss.ChannelConsistencyLoss(reduction=reduction)  # nn.KLDivLoss(reduction="mean")
        self.loss_edge    = loss.EdgeLoss(reduction=reduction)
    
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
        loss         = (self.spa_weight * loss_spa
                        + self.exp_weight * loss_exp
                        + self.col_weight * loss_col
                        + self.tv_weight * loss_tv
                        + self.edge_weight * loss_edge)
                       # + self.channel_weight * loss_channel \
        return loss
        
# endregion


# region Model

@MODELS.register(name="zerodace")
class ZeroDACE(base.LowLightImageEnhancementModel):
    """Zero-DACE (Zero-Reference Deep Attention Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(
        self,
        config       : Any                     = None,
        loss         : Any                     = CombinedLoss(),
        variant      : int | str        | None = 0,
        num_iters    : int | str               = 8,
        unsharp_sigma: int | str | bool | None = None,
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
        variant            = int(variant) if variant.isdigit() else variant
        self.variant       = f"{variant:04d}" if isinstance(variant, int) else variant
        self.num_channels  = 32
        self.scale_factor  = 1.0
        self.num_iters     = int(num_iters)       if isinstance(num_iters, str)     and num_iters.isdigit()     else 8
        self.unsharp_sigma = float(unsharp_sigma) if isinstance(unsharp_sigma, str) and unsharp_sigma.isdigit() else None

        # 0000-0099: test Zero-DCE++ with new loss
        if self.variant == "0000":
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.DSConv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv2    = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv3    = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv4    = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv5    = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv6    = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv7    = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, dw_stride=1, dw_padding=1)
        # 0100-0199: test layers composition
        elif self.variant == "0100":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0101":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.Conv2d(    in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv2    = nn.Conv2d(    in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv3    = nn.Conv2d(    in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv4    = nn.Conv2d(    in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0102":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv6    = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv7    = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1)
        elif self.variant == "0103":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.DSConv2d(  in_channels=3,                     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv2    = nn.DSConv2d(  in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv3    = nn.DSConv2d(  in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv4    = nn.DSConv2d(  in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0104":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.DSConv2d(  in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv6    = nn.DSConv2d(  in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv7    = nn.DSConv2d(  in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, dw_stride=1, dw_padding=1)
        elif self.variant == "0105":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1)
        elif self.variant == "0106":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.DSConv2d(  in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, dw_stride=1, dw_padding=1)
        # 0200-0299: test Conv layer
        elif self.variant == "0200":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS1(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS1(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS1(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS1(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS1(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS1(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS1(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0201":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS2(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS2(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS2(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS2(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS2(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS2(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS2(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0202":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS3(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS3(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS3(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS3(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS3(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS3(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS3(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0203":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS4(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS4(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS4(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS4(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS4(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS4(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS4(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0204":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS5(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS5(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS5(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS5(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS5(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS5(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS5(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0205":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS6(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS6(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS6(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS6(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS6(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS6(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS6(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0206":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS7(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS7(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS7(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS7(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS7(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS7(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS7(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0207":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS8(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS8(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS8(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS8(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS8(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS8(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS8(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0208":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS9(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS9(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS9(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS9(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS9(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS9(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS9(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0209":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS10(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS10(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS10(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS10(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS10(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS10(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS10(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0210":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS11(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS11(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS11(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS11(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS11(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS11(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS11(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0211":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS12(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS12(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS12(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS12(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS12(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS12(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS12(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0212":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS13(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS13(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS13(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS13(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS13(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS13(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS13(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0213":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.GhostConv2dV2(in_channels=3,                     out_channels=self.num_channels)
            self.conv2    = nn.GhostConv2dV2(in_channels=self.num_channels,     out_channels=self.num_channels)
            self.conv3    = nn.GhostConv2dV2(in_channels=self.num_channels,     out_channels=self.num_channels)
            self.conv4    = nn.GhostConv2dV2(in_channels=self.num_channels,     out_channels=self.num_channels)
            self.conv5    = nn.GhostConv2dV2(in_channels=self.num_channels * 2, out_channels=self.num_channels)
            self.conv6    = nn.GhostConv2dV2(in_channels=self.num_channels * 2, out_channels=self.num_channels)
            self.conv7    = nn.GhostConv2dV2(in_channels=self.num_channels * 2, out_channels=3,               )
        # 0300-0399: test activation layer
        elif self.variant == "0300":
            self.act      = nn.Sigmoid()
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0301":
            self.act      = nn.Tanh()
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0302":
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0303":
            self.act      = nn.PReLU()
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0304":
            self.act      = nn.ELU()
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0305":
            self.act      = nn.SELU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        # 0400-0499: test normalization layer's num_features
        elif self.variant == "0400":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.1, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.1, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.1, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.1, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.1, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.1, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.1, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0401":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.2, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.2, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.2, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.2, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.2, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.2, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.2, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0402":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.3, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.3, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.3, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.3, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.3, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.3, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.3, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0403":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.4, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.4, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.4, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.4, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.4, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.4, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.4, norm2=nn.HalfInstanceNorm2d)
        elif self.variant == "0404":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.5, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.5, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.5, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.5, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.5, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.5, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.5, norm2=nn.HalfInstanceNorm2d)
        # 0500-0599: test normalization block
        elif self.variant == "0500":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=HalfInstanceNorm2dBlock)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=HalfInstanceNorm2dBlock)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=HalfInstanceNorm2dBlock)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=HalfInstanceNorm2dBlock)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=HalfInstanceNorm2dBlock)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=HalfInstanceNorm2dBlock)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=HalfInstanceNorm2dBlock)
        # 0600-0699: test skip connections
        elif self.variant == "0600":
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 4, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 5, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 6, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, norm2=nn.HalfInstanceNorm2d)
        
    @property
    def config_dir(self) -> pathlib.Path:
        return pathlib.Path(__file__).absolute().parent / "config"
    
    def init_weights(self, m: nn.Module):
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
            target: A ground-truth of shape NCHW. Default: None.
            
        Return:
            Predictions and loss value.
        """
        pred  = self.forward(input=input, *args, **kwargs)
        loss  = self.loss(input, pred) if self.loss else None
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.
                
        Return:
            Predictions.
        """
        x = input
        #
        if self.variant in [
            "0000",
            "0100", "0101", "0102", "0103", "0104", "0105", "0106",
            "0200", "0201", "0202", "0203", "0204", "0205", "0206", "0207", "0208", "0209", "0210", "0211", "0212", "0213",
            "0300", "0301", "0302", "0303", "0304", "0305",
            "0400", "0401", "0402", "0403", "0404",
            "0500",
        ]:
            if self.scale_factor == 1:
                x_down = x
            else:
                scale_factor = 1 / self.scale_factor
                x_down       = F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
            f1 = self.act(self.conv1(x_down))
            f2 = self.act(self.conv2(f1))
            f3 = self.act(self.conv3(f2))
            f4 = self.act(self.conv4(f3))
            f5 = self.act(self.conv5(torch.cat([f3, f4], 1)))
            f6 = self.act(self.conv6(torch.cat([f2, f5], 1)))
            f7 =   F.tanh(self.conv7(torch.cat([f1, f6], 1)))
            if self.scale_factor == 1:
                f7 = f7
            else:
                f7 = self.upsample(f7)
        elif self.variant in ["0600"]:
            if self.scale_factor == 1:
                x_down = x
            else:
                scale_factor = 1 / self.scale_factor
                x_down       = F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
            f1 = self.act(self.conv1(x_down))
            f2 = self.act(self.conv2(f1))
            f3 = self.act(self.conv3(f2))
            f4 = self.act(self.conv4(f3))
            f5 = self.act(self.conv5(torch.cat([f1, f2, f3, f4], 1)))
            f6 = self.act(self.conv6(torch.cat([f1, f2, f3, f4, f5], 1)))
            f7 =   F.tanh(self.conv7(torch.cat([f1, f2, f3, f4, f5, f6], 1)))
            if self.scale_factor == 1:
                f7 = f7
            else:
                f7 = self.upsample(f7)
        #
        y = x + f7 * (torch.pow(x, 2) - x)
        for i in range(self.num_iters - 1):
            y = y + f7 * (torch.pow(y, 2) - y)
        #
        if self.unsharp_sigma is not None:
            y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))
            # y = self.unsharp_mask(y)
        #
        return f7, y
    
    def unsharp_mask(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.unsharp_sigma, (int, float)):
            return image
        
        c = image.shape[-3]
        kernel = (
            torch.tensor(
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]]
            ) + self.unsharp_sigma * torch.tensor(
                [[ 0, -1,  0],
                 [-1,  4, -1],
                 [ 0, -1,  0]]
            ) / 4
        )
        kernel = kernel.to(image.device)
        image  = F.conv2d(image, kernel.repeat(c, 1, 1, 1), padding=1, groups=c)
        image  = image.clamp(0, 1)
        return image
        
    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in [
            self.conv1, self.conv2, self.conv3, self.conv4,
            self.conv5, self.conv6, self.conv7
        ]:
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss

# endregion
