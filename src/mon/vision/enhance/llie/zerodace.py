#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DACE models."""

from __future__ import annotations

__all__ = ["ZeroDACE"]

from typing import Any

import torch

from mon import nn
from mon.core import pathlib
from mon.globals import MODELS
from mon.nn import functional as F
from mon.vision import loss
from mon.vision.enhance.llie import base

_current_dir = pathlib.Path(__file__).absolute().parent


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
        self.loss_channel = nn.KLDivLoss(reduction="mean")  # loss.ChannelConsistencyLoss(reduction=reduction)
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
        loss         = self.spa_weight * loss_spa \
                       + self.exp_weight * loss_exp \
                       + self.col_weight * loss_col \
                       + self.tv_weight * loss_tv \
                       + self.channel_weight * loss_channel \
                       + self.edge_weight * loss_edge
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
        config: Any = None,
        loss  : Any = CombinedLoss(),
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
        self.variant      = 1
        self.num_channels = 32
        self.scale_factor = 1.0
        self.num_iters    = 8
        
        if self.variant == 0:
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.DSConv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv2    = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv3    = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv4    = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv5    = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv6    = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv7    = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, dw_stride=1, dw_padding=1)
        elif self.variant == 1:
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
        elif self.variant == 2:
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.Conv2d(    in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv2    = nn.Conv2d(    in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv3    = nn.Conv2d(    in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv4    = nn.Conv2d(    in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
        elif self.variant == 3:
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, stride=1, padding=1)
            self.conv6    = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, stride=1, padding=1)
            self.conv7    = nn.Conv2d(    in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1)
        elif self.variant == 4:
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.DSConv2d(  in_channels=3,                     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv2    = nn.DSConv2d(  in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv3    = nn.DSConv2d(  in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv4    = nn.DSConv2d(  in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv5    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
        elif self.variant == 5:
            self.act      = nn.LeakyReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
            self.conv1    = nn.ABSConv2dS(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, stride=1, padding=1, p=0.25, act2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.DSConv2d(  in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv6    = nn.DSConv2d(  in_channels=self.num_channels * 2, out_channels=32,                kernel_size=3, dw_stride=1, dw_padding=1)
            self.conv7    = nn.DSConv2d(  in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, dw_stride=1, dw_padding=1)
        
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
        if self.variant in [0, 1, 2, 3, 4, 5]:
            if self.scale_factor == 1:
                x_down = x
            else:
                scale_factor = 1 / self.scale_factor
                x_down       = F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
            f1  = self.act(self.conv1(x_down))
            f2  = self.act(self.conv2(f1))
            f3  = self.act(self.conv3(f2))
            f4  = self.act(self.conv4(f3))
            f5  = self.act(self.conv5(torch.cat([f3, f4], 1)))
            f6  = self.act(self.conv6(torch.cat([f2, f5], 1)))
            f7  = F.tanh(self.conv7(torch.cat([f1, f6], 1)))
            if self.scale_factor == 1:
                f7 = f7
            else:
                f7 = self.upsample(f7)
            y = x + f7 * (torch.pow(x, 2) - x)
            for i in range(self.num_iters - 1):
                y = y + f7 * (torch.pow(y, 2) - y)
        #
        return f7, y
        
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
