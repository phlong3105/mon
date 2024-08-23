#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MobileNetV2.

This module implements MobileNetV2 models.
"""

from __future__ import annotations

__all__ = [
    "MobileNetV2",
]

from typing import Any

from torchvision.models import mobilenet_v2

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console = core.console


# region Model

@MODELS.register(name="mobilenet_v2", arch="mobilenet")
class MobileNetV2(nn.ExtraModel, base.ImageClassificationModel):
    """MobileNetV2 models from the paper: "MobileNetV2: Inverted Residuals
    and Linear Bottlenecks"
    
    References:
        https://arxiv.org/abs/1801.04381
    """
    
    arch   : str  = "mobilenet"
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v2/imagenet1k_v1/mobilenet_v2_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v2/imagenet1k_v2/mobilenet_v2_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "mobilenet_v2",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        width_mult : float = 1.0,
        dropout    : float = 0.2,
        weights    : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            num_classes = num_classes,
            weights     = weights,
            *args, **kwargs
        )
        
        if isinstance(self.weights, dict):
            in_channels = self.weights.get("in_channels", in_channels)
            num_classes = self.weights.get("num_classes", num_classes)
            width_mult  = self.weights.get("width_mult" , width_mult )
            dropout     = self.weights.get("dropout"    , dropout)
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes
        self.width_mult   = width_mult
        self.dropout      = dropout
        
        self.model = mobilenet_v2(
            num_classes = self.num_classes,
            width_mult  = self.width_mult,
            dropout     = self.dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}

# endregion
