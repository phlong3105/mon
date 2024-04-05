#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements RapidLight (RapidLight: Real-Time Unsupervised
Low-Light Image Enhancement) models.
"""

from __future__ import annotations

__all__ = [
    "RapidLightV00",
    "RapidLightV01",
]

from typing import Any, Literal

import torch

from mon import core, nn, proc
from mon.core import _callable
from mon.globals import MODELS, Scheme
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

# endregion


# region Dev-Model

@MODELS.register(name="rapidlight_v00")
class RapidLightV00(base.LowLightImageEnhancementModel):
    """RapidLight (RapidLight: Real-Time Unsupervised Low-Light Image Enhancement) model.
    
    Baseline model starting from ZeroDCE.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: The number of input and output channels for subsequent
            layers. Default: ``32``.
        num_iters: The number of progressive loop. Default: ``8``.
    
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}

    def __init__(
        self,
        channels    : int = 3,
        num_channels: int = 64,
        num_iters   : int = 8,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "rapidlight",
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
            
        self._channels    = channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        
        # Construct model
        self.l_conv1 = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
        self.l_conv2 = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.l_conv3 = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.l_conv4 = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.l_conv5 = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.l_conv6 = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.l_conv7 = nn.Conv2d(self.num_channels * 2, 3,     3, 1, 1, bias=True)
        self.l_relu  = nn.ReLU(inplace=True)
        
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

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
        x  = input
        
        l1 = self.l_relu(self.l_conv1(x))
        l2 = self.l_relu(self.l_conv2(l1))
        l3 = self.l_relu(self.l_conv3(l2))
        l4 = self.l_relu(self.l_conv4(l3))
        l5 = self.l_relu(self.l_conv5(torch.cat([l3, l4], 1)))
        l6 = self.l_relu(self.l_conv6(torch.cat([l2, l5], 1)))
        l  =  torch.tanh(self.l_conv7(torch.cat([l1, l6], 1)))
        
        y  = x
        for i in range(0, self.num_iters):
            y = y + l * (torch.pow(y, 2) - y)
        
        return l, y


@MODELS.register(name="rapidlight_v01")
class RapidLightV01(base.LowLightImageEnhancementModel):
    """RapidLight (RapidLight: Real-Time Unsupervised Low-Light Image Enhancement) model.
    
    Testing the effect of extra input channel.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: The number of input and output channels for subsequent
            layers. Default: ``32``.
        num_iters: The number of progressive loop. Default: ``8``.
    
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}

    def __init__(
        self,
        channels    : int = 3,
        num_channels: int = 64,
        num_iters   : int = 8,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "rapidlight",
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
            
        self._channels    = channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        
        # Construct model
        self.l_conv1 = nn.DSConv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
        self.l_conv2 = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.l_conv3 = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.l_conv4 = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.l_conv5 = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.l_conv6 = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.l_conv7 = nn.DSConv2d(self.num_channels * 2, 3,     3, 1, 1, bias=True)
        self.l_relu  = nn.ReLU(inplace=True)
        
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
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
        x   = input
        #
        l1  = self.l_relu(self.l_conv1(x))
        l2  = self.l_relu(self.l_conv2(l1))
        l3  = self.l_relu(self.l_conv3(l2))
        l4  = self.l_relu(self.l_conv4(l3))
        l5  = self.l_relu(self.l_conv5(torch.cat([l3, l4], 1)))
        l6  = self.l_relu(self.l_conv6(torch.cat([l2, l5], 1)))
        l_u =  torch.tanh(self.l_conv7(torch.cat([l1, l6], 1)))
        #
        xx  = 1 - x
        l1  = self.l_relu(self.l_conv1(xx))
        l2  = self.l_relu(self.l_conv2(l1))
        l3  = self.l_relu(self.l_conv3(l2))
        l4  = self.l_relu(self.l_conv4(l3))
        l5  = self.l_relu(self.l_conv5(torch.cat([l3, l4], 1)))
        l6  = self.l_relu(self.l_conv6(torch.cat([l2, l5], 1)))
        l_d =  torch.tanh(self.l_conv7(torch.cat([l1, l6], 1)))
        #
        l = (l_u + l_d) / 2
        #
        if not self.predicting:
            y = x
            for _ in range(self.num_iters):
                y = y + l * (torch.pow(y, 2) - y)
        else:
            y = x
            g = proc.get_guided_brightness_enhancement_map_prior(x, 2.8, 9)
            for _ in range(self.num_iters):
                b = y * (1 - g)
                d = y * g
                y = b + d + l * (torch.pow(d, 2) - d)
                
        return l, y
    
# endregion
