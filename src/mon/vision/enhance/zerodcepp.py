#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCE++ models."""

from __future__ import annotations

__all__ = [
    "ZeroDCEPP", "ZeroDCEPPVanilla",
]

from typing import Any

import torch
from torch.nn import functional as F

from mon.coreml import loss as mloss, model as mmodel, layer as mlayer
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
    """
    
    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tv_weight     : float = 1600.0,
        reduction     : str   = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tv_weight  = tv_weight
        
        self.loss_spa = mloss.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = mloss.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col = mloss.ColorConstancyLoss(reduction=reduction)
        self.loss_tv  = mloss.IlluminationSmoothnessLoss(reduction=reduction)
    
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
        loss = self.spa_weight * loss_spa \
               + self.exp_weight * loss_exp \
               + self.col_weight * loss_col \
               + self.tv_weight * loss_tv
        return loss

# endregion


# region Model

@MODELS.register(name="zerodcepp")
@MODELS.register(name="zerodce++")
class ZeroDCEPP(base.ImageEnhancementModel):
    """Zero-DCE++ (Zero-Reference Deep Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "zerodce++-lol" : {
            "name"       : "lol",
            "path"       : "",
            "file_name"  : "zerodce++-lol.pth",
            "num_classes": None,
        },
        "zerodce++-sice": {
            "name"       : "sice",
            "path"       : "",
            "file_name"  : "zerodce++-sice.pth",
            "num_classes": None,
        },
    }
    map_weights = {
        "backbone": {
            "2.dw_conv.weight" : "e_conv1.depth_conv.weight",
            "2.dw_conv.bias"   : "e_conv1.depth_conv.bias",
            "2.pw_conv.weight" : "e_conv1.point_conv.weight",
            "2.pw_conv.bias"   : "e_conv1.point_conv.bias",
            "4.dw_conv.weight" : "e_conv2.depth_conv.weight",
            "4.dw_conv.bias"   : "e_conv2.depth_conv.bias",
            "4.pw_conv.weight" : "e_conv2.point_conv.weight",
            "4.pw_conv.bias"   : "e_conv2.point_conv.bias",
            "6.dw_conv.weight" : "e_conv3.depth_conv.weight",
            "6.dw_conv.bias"   : "e_conv3.depth_conv.bias",
            "6.pw_conv.weight" : "e_conv3.point_conv.weight",
            "6.pw_conv.bias"   : "e_conv3.point_conv.bias",
            "8.dw_conv.weight" : "e_conv4.depth_conv.weight",
            "8.dw_conv.bias"   : "e_conv4.depth_conv.bias",
            "8.pw_conv.weight" : "e_conv4.point_conv.weight",
            "8.pw_conv.bias"   : "e_conv4.point_conv.bias",
            "11.dw_conv.weight": "e_conv5.depth_conv.weight",
            "11.dw_conv.bias"  : "e_conv5.depth_conv.bias",
            "11.pw_conv.weight": "e_conv5.point_conv.weight",
            "11.pw_conv.bias"  : "e_conv5.point_conv.bias",
            "14.dw_conv.weight": "e_conv6.depth_conv.weight",
            "14.dw_conv.bias"  : "e_conv6.depth_conv.bias",
            "14.pw_conv.weight": "e_conv6.point_conv.weight",
            "14.pw_conv.bias"  : "e_conv6.point_conv.bias",
            "17.dw_conv.weight": "e_conv7.depth_conv.weight",
            "17.dw_conv.bias"  : "e_conv7.depth_conv.bias",
            "17.pw_conv.weight": "e_conv7.point_conv.weight",
            "17.pw_conv.bias"  : "e_conv7.point_conv.bias",
        },
        "head"    : {},
    }
    
    def __init__(
        self,
        config: Any = "zerodce++.yaml",
        loss  : Any = CombinedLoss(),
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
    
    def init_weights(self, m: torch.nn.Module):
        """Initialize model's weights."""
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
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict) \
            and self.weights["name"] in ["sice"]:
            state_dict = mmodel.load_state_dict_from_path(
                model_dir=self.zoo_dir, **self.weights
            )
            state_dict       = state_dict["params"]
            model_state_dict = self.model.state_dict()
            """
            for k in self.model.state_dict().keys():
                print(f"\"{k}\": ")
            for k in state_dict.keys():
                print(f"\"{k}\"")
            """
            for k, v in self.map_weights["backbone"].items():
                model_state_dict[k] = state_dict[v]
            if self.weights["num_classes"] == self.num_classes:
                for k, v in self.map_weights["head"].items():
                    model_state_dict[k] = state_dict[v]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()
    
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


class DSConv(torch.nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depth_conv = mlayer.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            groups       = in_channels,
        )
        self.point_conv = mlayer.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = 1,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.depth_conv(input)
        output = self.point_conv(output)
        return output
    
    
@MODELS.register(name="zerodcepp-vanilla")
@MODELS.register(name="zerodce++-vanilla")
class ZeroDCEPPVanilla(torch.nn.Module):
    """Original implementation of ZeroDCE++.
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE_extension
    """
    
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        number_f   = 32
        self.relu         = mlayer.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample     = mlayer.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.conv1        = DSConv(3,            number_f)
        self.conv2        = DSConv(number_f,     number_f)
        self.conv3        = DSConv(number_f,     number_f)
        self.conv4        = DSConv(number_f,     number_f)
        self.conv5        = DSConv(number_f * 2, number_f)
        self.conv6        = DSConv(number_f * 2, number_f)
        self.conv7        = DSConv(number_f * 2, 3       )
    
    def enhance(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        
        x1  = self.relu(self.conv1(x_down))
        x2  = self.relu(self.conv2(x1))
        x3  = self.relu(self.conv3(x2))
        x4  = self.relu(self.conv4(x3))
        x5  = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        a = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            a = a
        else:
            a = self.upsample(a)
        x = self.enhance(x, a)
        return x

# endregion
