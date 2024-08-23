#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Inception v3.

This module implements Inception models.
"""

from __future__ import annotations

__all__ = [
    "Inception3",
]

from typing import Any

from torchvision.models import inception_v3

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console = core.console


# region Model

@MODELS.register(name="inception_v3", arch="inception")
class Inception3(nn.ExtraModel, base.ImageClassificationModel):
    """Inception v3 model architecture from
    `Rethinking the Inception Architecture for Computer Vision <http://arxiv.org/abs/1512.00567>`_.

    Notes:
        **Important**: In contrast to the other models, the ``inception_v3``
        expects tensors with a size of `N x 3 x 299 x 299`, so ensure
        your images are sized accordingly.
    
    """
    
    arch   : str  = "inception"
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
            "path"       : ZOO_DIR / "vision/classify/inception/inception_v3/imagenet1k_v1/inception_v3_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name       : str   = "inception_v3",
        in_channels: int   = 3,
        num_classes: int   = 1000,
        aux_logits : bool  = True,
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
            aux_logits  = self.weights.get("aux_logits" , aux_logits)
            dropout     = self.weights.get("dropout"    , dropout)
        self.in_channels  = in_channels or self.in_channels
        self.num_channels = num_classes
        self.aux_logits   = aux_logits
        self.dropout      = dropout
        
        self.model = inception_v3(
            num_classes = self.num_classes,
            aux_logits  = self.aux_logits,
            dropout     = self.dropout,
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
