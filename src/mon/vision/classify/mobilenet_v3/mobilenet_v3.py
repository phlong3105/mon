#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MobileNetV3.

This module implements MobileNetV3 models.
"""

from __future__ import annotations

__all__ = [
    "MobileNetV3Large",
    "MobileNetV3Small",
]

from abc import ABC
from typing import Any

from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class MobileNetV3(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """MobileNetV3 models from the paper: "Searching for MobileNetV3".
    
    References:
        https://arxiv.org/abs/1905.02244
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "mobilenet"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}
    

@MODELS.register(name="mobilenet_v3_large", arch="mobilenet")
class MobileNetV3Large(MobileNetV3):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v3_large/imagenet1k_v1/mobilenet_v3_large_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v3_large/imagenet1k_v2/mobilenet_v3_large_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "mobilenet_v3_large",
        in_channels: int   = 3,
        num_classes: int   = 1000,
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
            dropout     = self.weights.get("dropout"    , dropout)
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes
        self.dropout      = dropout
        
        self.model = mobilenet_v3_large(
            num_classes = self.num_classes,
            dropout     = self.dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="mobilenet_v3_small", arch="mobilenet")
class MobileNetV3Small(MobileNetV3):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v3_small/imagenet1k_v1/mobilenet_v3_small_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "mobilenet_v3_small",
        in_channels: int   = 3,
        num_classes: int   = 1000,
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
            dropout     = self.weights.get("dropout"    , dropout)
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes
        self.dropout      = dropout
        
        self.model = mobilenet_v3_small(
            num_classes = self.num_classes,
            dropout     = self.dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

# endregion
