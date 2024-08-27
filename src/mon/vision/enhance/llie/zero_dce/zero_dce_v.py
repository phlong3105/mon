#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-DCE-V.

This module implements our idea. Base on Zero-DCE, we use the V-channel of the
HSV color space to enhance the low-light image.
"""

from __future__ import annotations

__all__ = [
    "ZeroDCEV",
]

from typing import Any, Literal

import numpy as np
import torch
from torch.nn import functional as F

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import filtering
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
        exp_mean_val  : float = 0.8,
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
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
            reduction  = reduction,
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

@MODELS.register(name="zero_dce_v", arch="zero_dce")
class ZeroDCEV(base.ImageEnhancementModel):

    model_dir: core.Path    = current_dir
    arch     : str          = "zero_dce"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int = 1,
        num_channels: int = 32,
        num_iters   : int = 15,
        down_size   : int = 256,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = "zero_dce_v",
            weights = weights,
            *args, **kwargs
        )
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            num_channels = self.weights.get("num_channels", num_channels)
            num_iters    = self.weights.get("num_iters"   , num_iters)
            down_size    = self.weights.get("down_size"   , down_size)
        self.in_channels  = in_channels  or self.in_channels
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.out_channels = self.in_channels * self.num_iters
        self.down_size    = down_size
        
        # Construct model
        self.relu     = nn.ReLU(inplace=True)
        self.e_conv1  = nn.Conv2d(self.in_channels,      self.num_channels, 3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
        
        # Loss
        self.loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        image    = datapoint.get("image")
        adjust   = outputs.pop("adjust")
        enhanced = outputs.get("enhanced")
        outputs["loss"] = self.loss(image, adjust, enhanced)
        # Return
        return outputs
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        image_rgb  = datapoint.get("image")
        image_hsv  = core.rgb_to_hsv(image_rgb)
        image_v    = core.rgb_to_v(image_rgb)
        image_v_lr = self.interpolate_image(image_v)
        
        x    = image_v_lr
        x1   =  self.relu(self.e_conv1(x))
        x2   =  self.relu(self.e_conv2(x1))
        x3   =  self.relu(self.e_conv3(x2))
        x4   =  self.relu(self.e_conv4(x3))
        x5   =  self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6   =  self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r  = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        x_rs = torch.split(x_r, self.in_channels, dim=1)
        
        y = image_v_lr
        for i in range(0, self.num_iters):
            y = y + x_rs[i] * (torch.pow(y, 2) - y)

        image_v_fixed_lr = y
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed  = self.replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed  = core.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        return {
            "image_v"          : image_v,
            "image_v_lr"       : image_v_lr,
            "adjust"           : x_r,
            "image_v_fixed_lr" : image_v_fixed_lr,
            "image_v_fixed"    : image_v_fixed,
            "enhanced"         : image_rgb_fixed,
        }
        
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size))
    
    @staticmethod
    def filter_up(
        x_lr  : torch.Tensor,
        y_lr  : torch.Tensor,
        x_hr  : torch.Tensor,
        radius: int = 1
    ):
        """Applies the guided filter to upscale the predicted image. """
        gf   = filtering.FastGuidedFilter(radius=radius)
        y_hr = gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0, 1)
        return y_hr
    
    @staticmethod
    def replace_v_component(
        image_hsv: torch.Tensor, v_new: torch.Tensor
    ) -> torch.Tensor:
        """Replaces the `V` component of an HSV image `[1, 3, H, W]`."""
        image_hsv[:, -1, :, :] = v_new
        return image_hsv
    
    def infer(
        self,
        datapoint    : dict,
        epochs       : int   = 200,
        lr           : float = 1e-5,
        weight_decay : float = 3e-4,
        reset_weights: bool  = True,
        *args, **kwargs
    ) -> dict:
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
        else:
            optimizer = nn.Adam(
                self.parameters(),
                lr           = lr,
                betas        = (0.9, 0.999),
                weight_decay = weight_decay,
            )
        
        # Pre-processing
        self.assert_datapoint(datapoint)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Training
        for _ in range(epochs):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
 
        # Forward
        self.eval()
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint=datapoint)
        timer.tock()
        self.assert_outputs(outputs)
    
        # Return
        outputs["time"] = timer.avg_time
        return outputs
    
# endregion
