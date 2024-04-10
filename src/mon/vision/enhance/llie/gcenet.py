#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GCENet models.

./run.sh gcenet none none train 100 sice-zerodce all vision/enhance/llie no last
"""

from __future__ import annotations

__all__ = [
    "GCENet",
]

from typing import Any, Literal

import kornia
import torch

from mon import core, nn, proc
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance.llie import base

console = core.console


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        exp_patch_size : int   = 16,
        exp_mean_val   : float = 0.6,
        spa_num_regions: Literal[4, 8, 16, 24] = 4,
        spa_patch_size : int   = 4,
        weight_col     : float = 5,
        weight_exp     : float = 10,
        weight_spa     : float = 1,
        weight_tva     : float = 1600,
        reduction      : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_col  = weight_col
        self.weight_exp  = weight_exp
        self.weight_spa  = weight_spa
        self.weight_tva  = weight_tva
        
        self.loss_col    = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_exp    = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_spa    = nn.SpatialConsistencyLoss(
            num_regions = spa_num_regions,
            patch_size  = spa_patch_size,
            reduction   = reduction,
        )
        self.loss_tva    = nn.TotalVariationLoss(reduction=reduction)
    
    def forward(
        self,
        input   : torch.Tensor,
        adjust  : torch.Tensor,
        enhance : torch.Tensor,
        **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_col  = self.loss_col(input=enhance)               if self.weight_col  > 0 else 0
        loss_exp  = self.loss_exp(input=enhance)               if self.weight_exp  > 0 else 0
        loss_spa  = self.loss_spa(input=enhance, target=input) if self.weight_spa  > 0 else 0
        loss_tva  = self.loss_tva(input=adjust)                if self.weight_tva  > 0 else 0
        loss = (
              self.weight_col * loss_col
            + self.weight_exp * loss_exp
            + self.weight_tva * loss_tva
            + self.weight_spa * loss_spa
        )
        return loss
        
# endregion


# region Module

class DnCNN(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        num_channels: int = 64,
        num_layers  : int = 17,
        kernel_size : int = 3,
        padding     : int = 1,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.num_channels = num_channels
        self.num_layers   = num_layers
        self.kernel_size  = kernel_size
        self.padding      = padding
        
        layers      = []
        layers.append(nn.Conv2d(self.in_channels, self.num_channels, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(self.num_layers - 2):
            layers.append(nn.Conv2d(self.num_channels, self.num_channels, kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(self.num_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(self.num_channels, self.in_channels, kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x        = input
        residual = self.dncnn(x)
        y        = x - residual
        return y
    
# endregion


# region Model

@MODELS.register(name="gcenet")
class GCENet(base.LowLightImageEnhancementModel):
    """Guidance Curve Estimation (GCENet) model.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default: ``64``.
        num_iters: The number of convolutional layers in the model.
            Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
        gamma: Gamma value for dark channel prior. Default: ``2.8``.
        
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}
    
    def __init__(
        self,
        in_channels : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 8,
        scale_factor: int   = 1,
        gamma       : float = 2.8,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "gcenet",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters"   , num_iters)
            scale_factor = self.weights.get("scale_factor", scale_factor)
            gamma        = self.weights.get("gamma"       , gamma)
        self.in_channels  = in_channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.scale_factor = scale_factor
        self.gamma        = gamma
        
        # Construct model
        self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
        self.enc1     = nn.DSConv2d(self.in_channels,      self.num_channels, 3, 1, 1, bias=True)
        self.enc2     = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.enc3     = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.mid      = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.dec3     = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.dec2     = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.dec1     = nn.DSConv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
        self.act      = nn.PReLU()
        
        # Loss
        self._loss    = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = self.forward(input=input, *args, **kwargs)
        adjust, enhance = pred
        loss = self.loss(input, adjust, enhance)
        return enhance, loss
        
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        # Denoising
        x = kornia.filters.bilateral_blur(x, (3, 3), 0.1, (1.5, 1.5))
        # Downsampling
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        enc1   = self.act(self.enc1(x_down))
        enc2   = self.act(self.enc2(enc1))
        enc3   = self.act(self.enc3(enc2))
        mid    = self.act(self.mid(enc3))
        dec3   = self.act(self.dec3(torch.cat([mid,  enc3], dim=1)))
        dec2   = self.act(self.dec2(torch.cat([dec3, enc2], dim=1)))
        dec1   =   F.tanh(self.dec1(torch.cat([dec2, enc1], dim=1)))
        l      = dec1
        # Upsampling
        if self.scale_factor != 1:
            l = self.upsample(l)
        # Enhancement
        if not self.predicting:
            y = x
            for _ in range(self.num_iters):
                y = y + l * (torch.pow(y, 2) - y)
        else:
            y = x
            g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
            for _ in range(self.num_iters):
                b = y * (1 - g)
                d = y * g
                y = b + d + l * (torch.pow(d, 2) - d)
        return l, y
    
# endregion
