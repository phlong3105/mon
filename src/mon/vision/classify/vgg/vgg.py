#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG.

This module implements VGG models.
"""

from __future__ import annotations

__all__ = [
    "VGG11",
    "VGG11_BN",
    "VGG13",
    "VGG13_BN",
    "VGG16",
    "VGG16_BN",
    "VGG19",
    "VGG19_BN",
]

from abc import ABC
from typing import Any

from torchvision.models import (
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
)

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class VGG(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """VGG models from the paper: "Very Deep Convolutional Networks for
    Large-Scale Image Recognition"
    
    References:
        https://arxiv.org/abs/1409.1556
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "vgg"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}


@MODELS.register(name="vgg11", arch="vgg")
class VGG11(VGG):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg11-8a719046.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg11/imagenet1k_v1/vgg11_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "vgg11",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
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
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = vgg11(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg11_bn", arch="vgg")
class VGG11_BN(VGG):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg11_bn/imagenet1k_v1/vgg11_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "vgg11_bn",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
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
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = vgg11_bn(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg13", arch="vgg")
class VGG13(VGG):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg13-19584684.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg13/imagenet1k_v1/vgg13_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "vgg13",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
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
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = vgg13(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg13_bn", arch="vgg")
class VGG13_BN(VGG):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg13_bn/imagenet1k_v1/vgg13_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "vgg13_bn",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
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
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = vgg13_bn(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
            

@MODELS.register(name="vgg16", arch="vgg")
class VGG16(VGG):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg16-397923af.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg16/imagenet1k_v1/vgg16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "vgg16",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
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
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = vgg16(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg16_bn", arch="vgg")
class VGG16_BN(VGG):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg16_bn/imagenet1k_v1/vgg16_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "vgg16_bn",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
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
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = vgg16_bn(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg19", arch="vgg")
class VGG19(VGG):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg19/imagenet1k_v1/vgg19_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "vgg19",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
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
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = vgg19(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg19_bn", arch="vgg")
class VGG19_BN(VGG):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg19_bn/imagenet1k_v1/vgg19_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "vgg19_bn",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
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
        self.out_channels = num_classes or self.out_channels
        self.dropout      = dropout

        self.model = vgg19_bn(
            num_classes = self.out_channels,
            dropout     = self.dropout
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
# endregion
