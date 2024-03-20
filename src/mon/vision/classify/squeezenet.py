#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements SqueezeNet models."""

from __future__ import annotations

__all__ = [
    "SqueezeNet",
    "SqueezeNet1_0",
    "SqueezeNet1_1",
]

from abc import ABC
from typing import Any

import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision.classify import base

console = core.console


# region Module

class Fire(nn.Module):
    
    def __init__(
        self,
        inplanes        : int,
        squeeze_planes  : int,
        expand1x1_planes: int,
        expand3x3_planes: int,
        *args, **kwargs
    ):
        super().__init__()
        self.inplanes             = inplanes
        self.squeeze              = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation   = nn.ReLU(inplace=True)
        self.expand1x1            = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3            = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.squeeze_activation(self.squeeze(x))
        y = torch.cat(
            [
                self.expand1x1_activation(self.expand1x1(x)),
                self.expand3x3_activation(self.expand3x3(x))
            ], 1
        )
        return y
    
# endregion


# region Model

class SqueezeNet(base.ImageClassificationModel, ABC):
    """SqueezeNet.
    
    See Also: :class:`base.ImageClassificationModel`
    """
    
    _scheme: list[Scheme] = [Scheme.SUPERVISED]
    _zoo   : dict = {}

    def __init__(
        self,
        version    : str   = "1_0",
        channels   : int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
        weights    : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            channels    = channels,
            num_classes = num_classes,
            weights     = weights,
            *args, **kwargs
        )
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(self.channels, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(self.channels, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError(f"Unsupported SqueezeNet version {version}: ``1_0`` or ``1_1`` expected")
        
        # The final convolution is initialized differently from the rest
        final_conv      = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, model: nn.Module):
        pass
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        x = self.classifier(x)
        y = torch.flatten(x, 1)
        return y


@MODELS.register(name="squeezenet1_0")
class SqueezeNet1_0(SqueezeNet):
    """SqueezeNet model architecture from the `SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size
    <https://arxiv.org/abs/1602.07360>`_ paper.
    
    See Also: :class:`SqueezeNet`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
            "path"       : "squeezenet/squeezenet1_0_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name    = "squeezenet1_0",
            version = "1_0",
            *args, **kwargs
        )
       

@MODELS.register(name="squeezenet1_1")
class SqueezeNet1_1(SqueezeNet):
    """SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`__.
    
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    
    See Also: :class:`SqueezeNet`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
            "path"       : "squeezenet/squeezenet1_1_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name    = "squeezenet1_1",
            version = "1_1",
            *args, **kwargs
        )
        
# endregion
