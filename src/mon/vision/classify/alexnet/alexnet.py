#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AlexNet.

This module implements AlexNet models.
"""

from __future__ import annotations

__all__ = [
    "AlexNet",
]

from typing import Any

from torchvision.models import alexnet

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

@MODELS.register(name="alexnet", arch="alexnet")
class AlexNet(nn.ExtraModel, base.ImageClassificationModel):
    
    model_dir: core.Path    = current_dir
    arch     : str          = "alexnet"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "path"       : ZOO_DIR / "vision/classify/alexnet/alexnet/imagenet1k_v1/alexnet_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "alexnet",
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
        
        self.model = alexnet(num_classes=self.out_channels, dropout=self.dropout)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}

# endregion
