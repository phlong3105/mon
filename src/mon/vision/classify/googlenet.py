#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GoogLeNet (Inception v1).

This module implements GoogleNet models.
"""

from __future__ import annotations

__all__ = [
    "GoogleNet",
]

from typing import Any

from torchvision.models import googlenet

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console = core.console


# region Model

@MODELS.register(name="googlenet", arch="googlenet")
class GoogleNet(nn.ExtraModel, base.ImageClassificationModel):
    """GoogLeNet (Inception v1) models from the paper: "Going Deeper with
    Convolutions".
    
    References:
        https://arxiv.org/abs/1409.4842
    """
    
    arch   : str  = "googlenet"
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/googlenet-1378be20.pth",
            "path"       : ZOO_DIR / "vision/classify/googlenet/googlenet/imagenet1k_v1/googlenet_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str = "googlenet",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        aux_logits : bool  = True,
        dropout    : float = 0.2,
        dropout_aux: float = 0.7,
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
            aux_logits  = self.weights.get("aux_logits" , aux_logits)
            dropout     = self.weights.get("dropout"    , dropout)
            dropout_aux = self.weights.get("dropout_aux", dropout_aux)
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes
        self.aux_logits   = aux_logits
        self.dropout      = dropout
        self.dropout_aux  = dropout_aux
        
        self.model = googlenet(
            num_classes = self.num_classes,
            aux_logits  = self.aux_logits,
            dropout     = self.dropout,
            dropout_aux = self.dropout_aux,
        )
        
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
