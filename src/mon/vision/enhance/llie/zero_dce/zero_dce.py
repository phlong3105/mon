#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-DCE

This module implements the paper: "Zero-Reference Deep Curve Estimation for
Low-Light Image Enhancement".

References:
    https://github.com/Li-Chongyi/Zero-DCE
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE_RE",
]

from typing import Any, Literal

import torch

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

class Loss(nn.Loss):

    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tva_weight    : float = 200.0,
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
        self.loss_tva = nn.TotalVariationLoss(reduction=reduction)
    
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

@MODELS.register(name="zero_dce_re", arch="zero_dce")
class ZeroDCE_RE(base.ImageEnhancementModel):
    """Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB
            image.
        num_channels: The number of input and output channels for subsequent
            layers. Default: ``32``.
        num_iters: The number of progressive loop. Default: ``8``.
        
    References:
        https://github.com/Li-Chongyi/Zero-DCE
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "zero_dce"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 32,
        num_iters   : int = 8,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "zero_dce_re",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        assert num_iters <= 8
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters"   , num_iters)
        self.in_channels  = in_channels  or self.in_channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.out_channels = self.in_channels * self.num_iters
        
        # Construct model
        self.relu     = nn.ReLU(inplace=True)
        self.e_conv1  = nn.Conv2d(self.in_channels,      self.num_channels, 3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
        self.maxpool  = nn.MaxPool2d(2, 2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Loss
        self.loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Loss
        image    = datapoint.get("image")
        enhanced = outputs.get("enhanced")
        adjust   = outputs.pop("adjust")
        outputs["loss"] = self.loss(image, adjust, enhanced)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x       = datapoint.get("image")
        x1      =  self.relu(self.e_conv1(x))
        x2      =  self.relu(self.e_conv2(x1))
        x3      =  self.relu(self.e_conv3(x2))
        x4      =  self.relu(self.e_conv4(x3))
        x5      =  self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6      =  self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r     = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        x_rs    = torch.split(x_r, 3, dim=1)
        y       = x
        outputs = {}
        for i in range(0, self.num_iters):
            y = y + x_rs[i] * (torch.pow(y, 2) - y)
            outputs[f"adjust_{i}"] = x_rs[i]
        outputs["adjust"]   = x_r
        outputs["enhanced"] = y
        return outputs
    
# endregion
