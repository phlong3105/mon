#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCE models."""

from __future__ import annotations

__all__ = [
    "ZeroDCE", "ZeroDCEVanilla",
]

from typing import Any

import torch

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
        tv_weight     : float = 200.0,
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

@MODELS.register(name="zerodce")
class ZeroDCE(base.ImageEnhancementModel):
    """Zero-DCE (Zero-Reference Deep Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "zerodce-lol": {
            "name"       : "lol",
            "path"       : "",
            "file_name"  : "zerodce-lol.pth",
            "num_classes": None,
        },
        "zerodce-sice": {
            "name"       : "sice",
            "path"       : "",
            "file_name"  : "zerodce-sice.pth",
            "num_classes": None,
        },
    }
    map_weights = {
        "backbone": {
            "1.weight" : "e_conv1.weight",
            "1.bias"   : "e_conv1.bias",
            "3.weight" : "e_conv2.weight",
            "3.bias"   : "e_conv2.bias",
            "5.weight" : "e_conv3.weight",
            "5.bias"   : "e_conv3.bias",
            "7.weight" : "e_conv4.weight",
            "7.bias"   : "e_conv4.bias",
            "10.weight": "e_conv5.weight",
            "10.bias"  : "e_conv5.bias",
            "13.weight": "e_conv6.weight",
            "13.bias"  : "e_conv6.bias",
            "16.weight": "e_conv7.weight",
            "16.bias"  : "e_conv7.bias",
        },
        "head"    : {},
    }
    
    def __init__(
        self,
        config: Any = "zerodce.yaml",
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
            else:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
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


@MODELS.register(name="zerodce-vanilla")
class ZeroDCEVanilla(torch.nn.Module):
    """Original implementation of Zero-DCE.
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE
    """
    
    def __init__(self):
        super().__init__()
        number_f   = 32
        self.relu  = mlayer.ReLU(inplace=True)
        self.conv1 = mlayer.Conv2d(3,            number_f, 3, 1, 1, bias=True)
        self.conv2 = mlayer.Conv2d(number_f,     number_f, 3, 1, 1, bias=True)
        self.conv3 = mlayer.Conv2d(number_f,     number_f, 3, 1, 1, bias=True)
        self.conv4 = mlayer.Conv2d(number_f,     number_f, 3, 1, 1, bias=True)
        self.conv5 = mlayer.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv6 = mlayer.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv7 = mlayer.Conv2d(number_f * 2, 24,       3, 1, 1, bias=True)
    
    def enhance(self, x: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x   = input
        x1  = self.relu(self.conv1(x))
        x2  = self.relu(self.conv2(x1))
        x3  = self.relu(self.conv3(x2))
        x4  = self.relu(self.conv4(x3))
        x5  = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        x = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return x
    
# endregion
