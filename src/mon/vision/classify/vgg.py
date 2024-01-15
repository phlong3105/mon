#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements VGG models."""

from __future__ import annotations

__all__ = [
    "VGG",
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
from typing import cast

import torch

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Model

class VGG(base.ImageClassificationModel, ABC):
    """VGG.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {}
    
    def __init__(
        self,
        features    : nn.Module,
        num_classes : int   = 1000,
        init_weights: bool  = True,
        dropout     : float = 0.5,
        name        : str   = "vgg",
        *args, **kwargs
    ):
        super().__init__(
            num_classes = num_classes,
            name        = name,
            *args, **kwargs
        )
        
        self.dropout    = dropout
        self.features   = features
        self.avgpool    = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, self.num_classes),
        )
        
        if init_weights:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y


def make_layers(cfg: list[str | int], batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: dict[str, list[str | int]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


@MODELS.register(name="vgg11")
class VGG11(VGG):
    """VGG-11 from `Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>`__.

    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/vgg11-8a719046.pth",
            "path"       : "vgg11-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "vgg",
        variant: str = "vgg11",
        *args, **kwargs
    ):
        super().__init__(
            features = make_layers(cfgs["A"], batch_norm=False),
            name     = name,
            variant  = variant,
            *args, **kwargs
        )


@MODELS.register(name="vgg11_bn")
class VGG11_BN(VGG):
    """VGG-11-B from `Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>`__.

    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            "path"       : "vgg11_bn-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "vgg",
        variant: str = "vgg11_bn",
        *args, **kwargs
    ):
        super().__init__(
            features = make_layers(cfgs["A"], batch_norm=True),
            name     = name,
            variant  = variant,
            *args, **kwargs
        )


@MODELS.register(name="vgg13")
class VGG13(VGG):
    """VGG-13 from `Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>`__.

    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/vgg13-19584684.pth",
            "path"       : "vgg13-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "vgg",
        variant: str = "vgg13",
        *args, **kwargs
    ):
        super().__init__(
            features = make_layers(cfgs["B"], batch_norm=False),
            name     = name,
            variant  = variant,
            *args, **kwargs
        )


@MODELS.register(name="vgg13_bn")
class VGG13_BN(VGG):
    """VGG-13-BN from `Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>`__.

    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            "path"       : "vgg13_bn-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "vgg",
        variant: str = "vgg13_bn",
        *args, **kwargs
    ):
        super().__init__(
            features = make_layers(cfgs["B"], batch_norm=True),
            name     = name,
            variant  = variant,
            *args, **kwargs
        )


@MODELS.register(name="vgg16")
class VGG16(VGG):
    """VGG-16 from `Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>`__.

    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/vgg16-397923af.pth",
            "path"       : "vgg16-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k-features": {
            "url"        : "https://download.pytorch.org/models/vgg16_features-amdegroot-88682ab5.pth",
            "path"       : "vgg16-imagenet1k-features.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "vgg",
        variant: str = "vgg16",
        *args, **kwargs
    ):
        super().__init__(
            features = make_layers(cfgs["D"], batch_norm=False),
            name     = name,
            variant  = variant,
            *args, **kwargs
        )


@MODELS.register(name="vgg16_bn")
class VGG16_BN(VGG):
    """VGG-16-BN from `Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>`__.

    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "path"       : "vgg16_bn-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "vgg",
        variant: str = "vgg16_bn",
        *args, **kwargs
    ):
        super().__init__(
            features = make_layers(cfgs["D"], batch_norm=True),
            name     = name,
            variant  = variant,
            *args, **kwargs
        )


@MODELS.register(name="vgg19")
class VGG19(VGG):
    """VGG-19 from `Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>`__.

    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "path"       : "vgg19-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "vgg",
        variant: str = "vgg19",
        *args, **kwargs
    ):
        super().__init__(
            features = make_layers(cfgs["E"], batch_norm=False),
            name     = name,
            variant  = variant,
            *args, **kwargs
        )


@MODELS.register(name="vgg19_bn")
class VGG19_BN(VGG):
    """VGG-19-BN from `Very Deep Convolutional Networks for Large-Scale Image
    Recognition <https://arxiv.org/abs/1409.1556>`__.

    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
            "path"       : "vgg19_bn-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "vgg",
        variant: str = "vgg19_bn",
        *args, **kwargs
    ):
        super().__init__(
            features = make_layers(cfgs["E"], batch_norm=True),
            name     = name,
            variant  = variant,
            *args, **kwargs
        )
        
# endregion
