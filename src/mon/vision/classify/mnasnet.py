#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MNASNet.

This module implements MNASNet models.
"""

from __future__ import annotations

__all__ = [
    "MNASNet0_5",
    "MNASNet0_75",
    "MNASNet1_0",
    "MNASNet1_3",
]

from abc import ABC
from typing import Any

from torchvision.models import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console = core.console


# region Model

class MNASNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """MNASNet models from the paper: "MnasNet: Platform-Aware Neural
    Architecture Search for Mobile"
    
    References:
        https://arxiv.org/abs/1807.11626
    """
    
    arch   : str  = "mnasnet"
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}
    

@MODELS.register(name="mnasnet0_5", arch="mnasnet")
class MNASNet0_5(MNASNet):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
            "path"       : ZOO_DIR / "vision/classify/mnasnet/mnasnet0_5/imagenet1k_v1/mnasnet0_5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str    = "mnasnet0_5",
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
        
        self.model = mnasnet0_5(
            num_classes = self.num_classes,
            dropout     = self.dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="mnasnet0_75", arch="mnasnet")
class MNASNet0_75(MNASNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet0_75-7090bc5f.pth",
            "path"       : ZOO_DIR / "vision/classify/mnasnet/mnasnet0_75/imagenet1k_v1/mnasnet0_75_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "mnasnet0_75",
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
        
        self.model = mnasnet0_75(
            num_classes = self.num_classes,
            dropout     = self.dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
        
@MODELS.register(name="mnasnet1_0", arch="mnasnet")
class MNASNet1_0(MNASNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
            "path"       : ZOO_DIR / "vision/classify/mnasnet/mnasnet1_0/imagenet1k_v1/mnasnet1_0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "mnasnet1_0",
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
        
        self.model = mnasnet1_0(
            num_classes = self.num_classes,
            dropout     = self.dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
        
@MODELS.register(name="mnasnet1_3", arch="mnasnet")
class MNASNet1_3(MNASNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet1_3-a4c69d6f.pth",
            "path"       : ZOO_DIR / "vision/classify/mnasnet/mnasnet1_3/imagenet1k_v1/mnasnet1_3_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "mnasnet1_3",
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
        
        self.model = mnasnet1_3(
            num_classes = self.num_classes,
            dropout     = self.dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
# endregion
