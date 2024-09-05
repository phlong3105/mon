#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Swin Transformer.

This module implements Swin Transformer models.
"""

from __future__ import annotations

__all__ = [
    "Swin_B",
    "Swin_S",
    "Swin_T",
    "Swin_V2_B",
    "Swin_V2_S",
    "Swin_V2_T",
]

from abc import ABC
from typing import Any

from torchvision.models import (
    swin_b, swin_s, swin_t, swin_v2_b, swin_v2_s, swin_v2_t,
)

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class SwinTransformer(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """Implements Swin Transformer from the paper: "Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows".
    
    References:
        https://arxiv.org/pdf/2103.14030
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "swin"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}
    

@MODELS.register(name="swin_t", arch="swin")
class Swin_T(SwinTransformer):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_t-704ceda3.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_t/imagenet1k_v1/swin_t_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "swin_t",
        in_channels      : int   = 3,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        weights          : Any   = None,
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
            in_channels       = self.weights.get("in_channels"      , in_channels)
            num_classes       = self.weights.get("num_classes"      , num_classes)
            dropout           = self.weights.get("dropout"          , dropout    )
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = swin_t(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="swin_s", arch="swin")
class Swin_S(SwinTransformer):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_s-5e29d889.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_s/imagenet1k_v1/swin_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "swin_s",
        in_channels      : int   = 3,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        weights          : Any   = None,
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
            in_channels       = self.weights.get("in_channels"      , in_channels)
            num_classes       = self.weights.get("num_classes"      , num_classes)
            dropout           = self.weights.get("dropout"          , dropout    )
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = swin_s(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="swin_b", arch="swin")
class Swin_B(SwinTransformer):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_b/imagenet1k_v1/swin_b_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "swin_b",
        in_channels      : int   = 3,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        weights          : Any   = None,
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
            in_channels       = self.weights.get("in_channels"      , in_channels)
            num_classes       = self.weights.get("num_classes"      , num_classes)
            dropout           = self.weights.get("dropout"          , dropout    )
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = swin_b(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="swin_v2_t", arch="swin")
class Swin_V2_T(SwinTransformer):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_v2_t/imagenet1k_v1/swin_v2_t_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "swin_v2_t",
        in_channels      : int   = 3,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        weights          : Any   = None,
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
            in_channels       = self.weights.get("in_channels"      , in_channels)
            num_classes       = self.weights.get("num_classes"      , num_classes)
            dropout           = self.weights.get("dropout"          , dropout    )
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = swin_v2_t(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="swin_v2_s", arch="swin")
class Swin_V2_S(SwinTransformer):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_v2_s/imagenet1k_v1/swin_v2_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "swin_v2_s",
        in_channels      : int   = 3,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        weights          : Any   = None,
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
            in_channels       = self.weights.get("in_channels"      , in_channels)
            num_classes       = self.weights.get("num_classes"      , num_classes)
            dropout           = self.weights.get("dropout"          , dropout    )
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = swin_v2_s(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="swin_v2_b", arch="swin")
class Swin_V2_B(SwinTransformer):
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_v2_b/imagenet1k_v1/swin_v2_b_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "swin_v2_b",
        in_channels      : int   = 3,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        weights          : Any   = None,
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
            in_channels       = self.weights.get("in_channels"      , in_channels)
            num_classes       = self.weights.get("num_classes"      , num_classes)
            dropout           = self.weights.get("dropout"          , dropout    )
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = swin_v2_b(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
# endregion
