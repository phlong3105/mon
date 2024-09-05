#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ShuffleNetV2.

This module implements ShuffleNetV2 models.
"""

from __future__ import annotations

__all__ = [
    "ShuffleNetV2_X1_0",
    "ShuffleNetV2_X1_5",
    "ShuffleNetV2_X2_0",
    "ShuffleNetV2_x0_5",
]

from abc import ABC
from typing import Any

from torchvision.models import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class ShuffleNetV2(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """ShuffleNetV2 models from the paper: "ShuffleNet V2: Practical Guidelines
    for Efficient CNN Architecture Design"
    
    References:
        https://arxiv.org/abs/1807.11164
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "shufflenet"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}


@MODELS.register(name="shufflenet_v2_x0_5", arch="shufflenet")
class ShuffleNetV2_x0_5(ShuffleNetV2):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            "path"       : ZOO_DIR / "vision/classify/shufflenet/shufflenet_v2_x0_5/imagenet1k_v1/shufflenet_v2_x0_5_x0_5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "shufflenet_v2_x0_5",
        in_channels: int = 3,
        num_classes: int = 1000,
        weights    : Any = None,
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
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes

        self.model = shufflenet_v2_x0_5(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="shufflenet_v2_x1_0", arch="shufflenet")
class ShuffleNetV2_X1_0(ShuffleNetV2):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            "path"       : ZOO_DIR / "vision/classify/shufflenet/shufflenetv2_x1_0/imagenet1k_v1/shufflenetv2_x1_0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "shufflenet_v2_x1_0",
        in_channels: int = 3,
        num_classes: int = 1000,
        weights    : Any = None,
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
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes

        self.model = shufflenet_v2_x1_0(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="shufflenet_v2_x1_5", arch="shufflenet")
class ShuffleNetV2_X1_5(ShuffleNetV2):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
            "path"       : ZOO_DIR / "vision/classify/shufflenet/shufflenetv2_x1_5/imagenet1k_v1/shufflenetv2_x1_5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "shufflenet_v2_x1_5",
        in_channels: int = 3,
        num_classes: int = 1000,
        weights    : Any = None,
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
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes

        self.model = shufflenet_v2_x1_5(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="shufflenet_v2_x2_0", arch="shufflenet")
class ShuffleNetV2_X2_0(ShuffleNetV2):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
            "path"       : ZOO_DIR / "vision/classify/shufflenet/shufflenetv2_x2_0/imagenet1k_v1/shufflenetv2_x2_0_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name       : str = "shufflenet_v2_x2_0",
        in_channels: int = 3,
        num_classes: int = 1000,
        weights    : Any = None,
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
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes

        self.model = shufflenet_v2_x2_0(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
# endregion
