#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCE models."""

from __future__ import annotations

__all__ = [
    "ZeroDCE",
]

from typing import Any

import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision.enhance.llie import base

console = core.console


# region Loss

class Loss(nn.Loss):
    """Loss = SpatialConsistencyLoss
              + ExposureControlLoss
              + ColorConstancyLoss
              + IlluminationSmoothnessLoss
    """
    
    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tv_weight     : float = 200.0,
        reduction     : str   = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tv_weight  = tv_weight
        
        self.loss_spa = nn.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_tv  = nn.IlluminationSmoothnessLoss(reduction=reduction)
    
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
        loss_spa = self.loss_spa(input=enhance, target=input)
        loss_exp = self.loss_exp(input=enhance)
        loss_col = self.loss_col(input=enhance)
        loss_tv  = self.loss_tv(input=a)
        loss     = self.spa_weight * loss_spa \
                   + self.exp_weight * loss_exp \
                   + self.col_weight * loss_col \
                   + self.tv_weight * loss_tv
        return loss

# endregion


# region Model

@MODELS.register(name="zerodce")
class ZeroDCE(base.LowLightImageEnhancementModel):
    """Zero-DCE (Zero-Reference Deep Curve Estimation) model.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: The number of input and output channels for subsequent
            layers. Default: ``32``.
        num_filters: The number of convolutional layers in the model.
            Default: ``8``.
        
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE>`__

    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {
        "sice_mix": {
            "url"         : None,
            "path"        : "zerodce/zerodce_best.pth",
            "channels"    : 3,
            "num_channels": 32,
            "num_iters"   : 8,
            "map": {},
        },
    }

    def __init__(
        self,
        channels    : int = 3,
        num_channels: int = 32,
        num_iters   : int = 8,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "zerodce",
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
        self.relu     = nn.ReLU(inplace=True)
        self.e_conv1  = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(self.num_channels * 2, 24,    3, 1, 1, bias=True)
        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
       
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
        loss = self.loss(input, pred) if self.loss else None
        return pred[-1], loss

    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x   = input
        x1  = self.relu(self.e_conv1(x))
        x2  = self.relu(self.e_conv2(x1))
        x3  = self.relu(self.e_conv3(x2))
        x4  = self.relu(self.e_conv4(x3))
        x5  = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        x_rs = torch.split(x_r, 3, dim=1)
        y    = x
        for i in range(0, self.num_iters):
            y = y + x_rs[i] * (torch.pow(y, 2) - y)

        return x_r, y

# endregion
