#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DenseNet.

This module implements DenseNet models.
"""

from __future__ import annotations

__all__ = [
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
]

from abc import ABC
from typing import Any

from torchvision.models import (
    densenet121, densenet161, densenet169, densenet201,
)

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console = core.console


# region Model

class DenseNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """DenseNet models from the paper: "Densely Connected Convolutional
    Networks".
    
    References:
        https://arxiv.org/pdf/1608.06993.pdf
    """
    
    arch   : str  = "densenet"
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}
    

@MODELS.register(name="densenet121", arch="densenet")
class DenseNet121(DenseNet):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet121-a639ec97.pth",
            "path"       : ZOO_DIR / "vision/classify/densenet/densenet121/imagenet1k_v1/densenet121_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "densenet121",
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
        
        self.model = densenet121(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="densenet161", arch="densenet")
class DenseNet161(DenseNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet161-8d451a50.pth",
            "path"       : ZOO_DIR / "vision/classify/densenet/densenet161/imagenet1k_v1/densenet161_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "densenet161",
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
        
        self.model = densenet161(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="densenet169", arch="densenet")
class DenseNet169(DenseNet):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            "path"       : ZOO_DIR / "vision/classify/densenet/densenet169/imagenet1k_v1/densenet169_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name       : str = "densenet169",
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
        
        self.model = densenet169(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="densenet201", arch="densenet")
class DenseNet201(DenseNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet201-c1103571.pth",
            "path"       : ZOO_DIR / "vision/classify/densenet/densenet201/imagenet1k_v1/densenet201_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "densenet201",
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
        
        self.model = densenet201(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
 
# endregion
