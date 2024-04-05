#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCE++ models."""

from __future__ import annotations

__all__ = [
    "ZeroDCEPP",
]

from typing import Any, Literal

import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance.llie import base

console = core.console


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tva_weight    : float = 1600.0,
        reduction     : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tva_weight = tva_weight
        
        self.loss_spa = nn.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_tva = nn.TotalVariationALoss(reduction=reduction)
    
    def forward(
        self,
        input  : torch.Tensor,
        adjust : torch.Tensor,
        enhance: torch.Tensor,
        **_
    ) -> torch.Tensor:
        loss_spa = self.loss_spa(input=enhance, target=input)
        loss_exp = self.loss_exp(input=enhance)
        loss_col = self.loss_col(input=enhance)
        loss_tva = self.loss_tva(input=adjust)
        loss     = (
              self.spa_weight * loss_spa
            + self.exp_weight * loss_exp
            + self.col_weight * loss_col
            + self.tva_weight * loss_tva
        )
        return loss

# endregion


# region Model

@MODELS.register(name="zero_dce++")
class ZeroDCEPP(base.LowLightImageEnhancementModel):
    """Zero-DCE++ (Zero-Reference Deep Curve Estimation) model.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: The number of input and output channels for subsequent
            layers. Default: ``32``.
        num_iters: The number of convolutional layers in the model.
            Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
        
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE_extension>`__

    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {
        "sice_mix" : {
            "url"         : None,
            "path"        : "zero_dce++/zero_dce++_sice_mix.pt",
            "channels"    : 3,
            "num_channels": 32,
            "num_iters"   : 8,
            "scale_factor": 1.0,
            "map": {},
        },
    }

    def __init__(
        self,
        channels    : int   = 3,
        num_channels: int   = 32,
        num_iters   : int   = 8,
        scale_factor: float = 1.0,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "zero_dce++",
            channels = channels,
            weights  = weights,
            *args, **kwargs
        )
        assert num_iters <= 8
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels     = self.weights.get("channels",     channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters",    num_iters)
            scale_factor = self.weights.get("scale_factor", scale_factor)
        
        self._channels     = channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.scale_factor = scale_factor
        
        # Construct model
        self.relu     = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.e_conv1  = nn.DSConv2d(self.channels,         self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv2  = nn.DSConv2d(self.num_channels,     self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv3  = nn.DSConv2d(self.num_channels,     self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv4  = nn.DSConv2d(self.num_channels,     self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv5  = nn.DSConv2d(self.num_channels * 2, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv6  = nn.DSConv2d(self.num_channels * 2, self.num_channels, kernel_size=3, stride=1, padding=1)
        self.e_conv7  = nn.DSConv2d(self.num_channels * 2, 3,     kernel_size=3, stride=1, padding=1)
        
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m: torch.nn.Module):
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
        
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        x1  =  self.relu(self.e_conv1(x_down))
        x2  =  self.relu(self.e_conv2(x1))
        x3  =  self.relu(self.e_conv3(x2))
        x4  =  self.relu(self.e_conv4(x3))
        x5  =  self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6  =  self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        if self.scale_factor != 1:
            x_r = self.upsample(x_r)

        y = x
        for i in range(0, self.num_iters):
            y = y + x_r * (torch.pow(y, 2) - y)
       
        return x_r, y

# endregion
