#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ViT (Vision Transformer).

This module implements ViT (Vision Transformer) models.
"""

from __future__ import annotations

__all__ = [
    "ViT_B_16",
    "ViT_B_32",
    "ViT_H_14",
    "ViT_L_16",
    "ViT_L_32",
]

from abc import ABC
from typing import Any

from torchvision.models import vit_b_16, vit_b_32, vit_h_14, vit_l_16, vit_l_32

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class VisionTransformer(base.ImageClassificationModel, ABC):
    """Vision Transformer models from "An Image is Worth 16x16 Words:
    Transformers for Image Recognition at Scale"
    
    References:
        https://arxiv.org/abs/2010.11929
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "vit"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass

    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        return {"logits": y}


@MODELS.register(name="vit_b_16", arch="vit")
class ViT_B_16(VisionTransformer):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_b_16/imagenet1k_v1/vit_b_16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_b_16_swag/imagenet1k_v1/vit_b_16_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_b_16_lc_swag/imagenet1k_v1/vit_b_16_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "vit_b_16",
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
            dropout           = self.weights.get("dropout"          , dropout)
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = vit_b_16(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vit_b_32", arch="vit")
class ViT_B_32(VisionTransformer):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_b_32/imagenet1k_v1/vit_b_32_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "vit_b_32",
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
            dropout           = self.weights.get("dropout"          , dropout)
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = vit_b_32(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vit_l_16", arch="vit")
class ViT_L_16(VisionTransformer):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_l_16/imagenet1k_v1/vit_l_16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_l_16_swag/imagenet1k_v1/vit_l_16_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_l_16_lc_swag/imagenet1k_v1/vit_l_16_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "vit_l_16",
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
            dropout           = self.weights.get("dropout"          , dropout)
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = vit_l_16(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vit_l_32", arch="vit")
class ViT_L_32(VisionTransformer):

    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_l_32/imagenet1k_v1/vit_l_32_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "vit_l_32",
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
            dropout           = self.weights.get("dropout"          , dropout)
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = vit_l_32(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
      

@MODELS.register(name="vit_h_14", arch="vit")
class ViT_H_14(VisionTransformer):
    
    zoo: dict = {
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_h_14_swag/imagenet1k_v1/vit_h_14_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_h_14_lc_swag/imagenet1k_v1/vit_h_14_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        name             : str   = "vit_h_14",
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
            dropout           = self.weights.get("dropout"          , dropout)
            attention_dropout = self.weights.get("attention_dropout", attention_dropout)
        self.in_channels       = in_channels or self.in_channels
        self.num_channels      = num_classes
        self.dropout           = dropout
        self.attention_dropout = attention_dropout

        self.model = vit_h_14(
            num_classes       = self.out_channels,
            dropout           = self.dropout,
            attention_dropout = self.attention_dropout,
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
# endregion
