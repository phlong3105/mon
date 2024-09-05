#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EfficientNet.

This module implements EfficientNet models.
"""

from __future__ import annotations

__all__ = [
    "EfficientNet_B0",
    "EfficientNet_B1",
    "EfficientNet_B2",
    "EfficientNet_B3",
    "EfficientNet_B4",
    "EfficientNet_B5",
    "EfficientNet_B6",
    "EfficientNet_B7",
    "EfficientNet_V2_L",
    "EfficientNet_V2_M",
    "EfficientNet_V2_S",
]

from abc import ABC
from typing import Any

from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    efficientnet_v2_l, efficientnet_v2_m, efficientnet_v2_s,
)

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class EfficientNet(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """EfficientNet models from the paper: "EfficientNet: Rethinking
    Model Scaling for Convolutional Neural Networks".
    
    References:
        https://arxiv.org/abs/1905.11946
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "efficientnet"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}
    

@MODELS.register(name="efficientnet_b0", arch="efficientnet")
class EfficientNet_B0(EfficientNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b0/imagenet1k_v1/efficientnet_b0_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_b0",
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
        
        self.model = efficientnet_b0(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b1", arch="efficientnet")
class EfficientNet_B1(EfficientNet):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b1/imagenet1k_v1/efficientnet_b1_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b1/imagenet1k_v2/efficientnet_b1_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_b1",
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
        
        self.model = efficientnet_b1(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b2", arch="efficientnet")
class EfficientNet_B2(EfficientNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b2/imagenet1k_v1/efficientnet_b2_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_b2",
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
        
        self.model = efficientnet_b2(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b3", arch="efficientnet")
class EfficientNet_B3(EfficientNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b3/imagenet1k_v1/efficientnet_b3_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_b3",
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
        
        self.model = efficientnet_b3(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b4", arch="efficientnet")
class EfficientNet_B4(EfficientNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b4/imagenet1k_v1/efficientnet_b4_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_b4",
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
        
        self.model = efficientnet_b4(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b5", arch="efficientnet")
class EfficientNet_B5(EfficientNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b5/imagenet1k_v1/efficientnet_b5_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_b5",
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
        
        self.model = efficientnet_b5(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_b6", arch="efficientnet")
class EfficientNet_B6(EfficientNet):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b6/imagenet1k_v1/efficientnet_b6_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_b6",
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
        
        self.model = efficientnet_b6(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
   
   
@MODELS.register(name="efficientnet_b7", arch="efficientnet")
class EfficientNet_B7(EfficientNet):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_b7/imagenet1k_v1/efficientnet_b7_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_b7",
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
        
        self.model = efficientnet_b7(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_v2_s", arch="efficientnet")
class EfficientNet_V2_S(EfficientNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_v2_s/imagenet1k_v1/efficientnet_v2_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_v2_s",
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
        
        self.model = efficientnet_v2_s(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_v2_m", arch="efficientnet")
class EfficientNet_V2_M(EfficientNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_v2_m/imagenet1k_v1/efficientnet_v2_m_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_v2_m",
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
        
        self.model = efficientnet_v2_m(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="efficientnet_v2_l", arch="efficientnet")
class EfficientNet_V2_L(EfficientNet):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
            "path"       : ZOO_DIR / "vision/classify/efficientnet/efficientnet_v2_l/imagenet1k_v1/efficientnet_v2_l_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "efficientnet_v2_l",
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
        
        self.model = efficientnet_v2_l(num_classes=self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
# endregion
