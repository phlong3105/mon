#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SqueezeNet.

This module implements SqueezeNet models.
"""

from __future__ import annotations

__all__ = [
    "SqueezeNet1_0",
    "SqueezeNet1_1",
]

from abc import ABC
from typing import Any

from torchvision.models import squeezenet1_0, squeezenet1_1

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class SqueezeNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """SqueezeNet models from the paper: "SqueezeNet: AlexNet-level accuracy
    with 50x fewer parameters and <0.5MB model size".
    
    References:
        https://arxiv.org/abs/1602.07360
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "squeezenet"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}

    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}


@MODELS.register(name="squeezenet1_0", arch="squeezenet")
class SqueezeNet1_0(SqueezeNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
            "path"       : ZOO_DIR / "vision/classify/squeezenet/squeezenet1_0/imagenet1k_v1/squeezenet1_0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "squeezenet1_0",
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout    : float = 0.5,
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
            dropout     = self.weights.get("dropout"    , dropout)
        self.in_channels  = in_channels or self.in_channels
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = squeezenet1_0(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
       

@MODELS.register(name="squeezenet1_1", arch="squeezenet")
class SqueezeNet1_1(SqueezeNet):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
            "path"       : ZOO_DIR / "vision/classify/squeezenet/squeezenet1_1/imagenet1k_v1/squeezenet1_1_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "squeezenet1_1",
        in_channels: int = 3,
        num_classes: int = 1000,
        dropout    : float = 0.5,
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
            dropout     = self.weights.get("dropout"    , dropout)
        self.in_channels  = in_channels or self.in_channels
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = squeezenet1_1(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
# endregion
