#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ConvNeXt.

This module implements ConNeXt models.
"""

from __future__ import annotations

__all__ = [
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
]

from abc import ABC
from typing import Any

from torchvision.models import (
    convnext_base, convnext_large, convnext_small, convnext_tiny,
)

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class ConvNeXt(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """ConvNeXt models from the paper: "A ConvNet for the 2020s".
    
    References:
        https://arxiv.org/abs/2201.03545
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "convnext"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}


@MODELS.register(name="convnext_base", arch="convnext")
class ConvNeXtBase(ConvNeXt):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
            "path"       : ZOO_DIR / "vision/classify/convnext/convnext_base/imagenet1k_v1/convnext_base_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "convnext_base",
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
        self.out_channels = num_classes or self.out_channels
        
        self.model = convnext_base(num_classes=self.out_channels)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="convnext_tiny", arch="convnext")
class ConvNeXtTiny(ConvNeXt):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
            "path"       : ZOO_DIR / "vision/classify/convnext/convnext_tiny/imagenet1k_v1/convnext_tiny_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "convnext_tiny",
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
        self.out_channels = num_classes or self.out_channels
        
        self.model = convnext_tiny(num_classes=self.out_channels)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="convnext_small", arch="convnext")
class ConvNeXtSmall(ConvNeXt):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_small-0c510722.pth",
            "path"       : ZOO_DIR / "vision/classify/convnext/convnext_small/imagenet1k_v1/convnext_small_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "convnext_small",
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
        self.out_channels = num_classes or self.out_channels
        
        self.model = convnext_small(num_classes=self.out_channels)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="convnext_large", arch="convnext")
class ConvNeXtLarge(ConvNeXt):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
            "path"       : ZOO_DIR / "vision/classify/convnext/convnext_large/imagenet1k_v1/convnext_large_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "convnext_large",
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
        self.out_channels = num_classes or self.out_channels
        
        self.model = convnext_large(num_classes=self.out_channels)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

# endregion
