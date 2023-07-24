#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ConNeXt models."""

from __future__ import annotations

__all__ = [
    "ConNeXtBase", "ConNeXtLarge", "ConNeXtSmall", "ConNeXtTiny",
    "ConvNeXtBlock", "ConvNeXtLayer",
]

import functools
from abc import ABC
from typing import Callable

import torch
import torchvision.ops
from torch import nn

from mon.foundation import pathlib
from mon.globals import LAYERS, MODELS
from mon.vision.classify import base
from mon.vision.ml import layer, model

_current_dir = pathlib.Path(__file__).absolute().parent


# region Module

@LAYERS.register()
class ConvNeXtLayer(layer.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        dim                  : int,
        layer_scale          : float,
        stochastic_depth_prob: float,
        stage_block_id       : int | None = None,
        total_stage_blocks   : int | None = None,
        norm                 : Callable   = None,
    ):
        super().__init__()
        sd_prob = stochastic_depth_prob
        if (stage_block_id is not None) and (total_stage_blocks is not None):
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
        if norm is None:
            norm = functools.partial(layer.LayerNorm, eps=1e-6)
        self.block = torch.nn.Sequential(
            layer.Conv2d(
                in_channels  = dim,
                out_channels = dim,
                kernel_size  = 7,
                padding      = 3,
                groups       = dim,
                bias         = True,
            ),
            layer.Permute([0, 2, 3, 1]),
            norm(dim),
            layer.Linear(
                in_features  = dim,
                out_features = 4 * dim,
                bias         = True,
            ),
            layer.GELU(),
            layer.Linear(
                in_features  = 4 * dim,
                out_features = dim,
                bias         = True,
            ),
            layer.Permute([0, 3, 1, 2]),
        )
        self.layer_scale      = torch.nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = torchvision.ops.StochasticDepth(sd_prob, "row")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y  = self.layer_scale * self.block(x)
        y  = self.stochastic_depth(y)
        y += x
        return y


@LAYERS.register()
class ConvNeXtBlock(layer.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        dim                  : int,
        layer_scale          : float,
        stochastic_depth_prob: float,
        num_layers           : int,
        stage_block_id       : int,
        total_stage_blocks   : int | None = None,
        norm                 : Callable   = None,
    ):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                ConvNeXtLayer(
                    dim                   = dim,
                    layer_scale           = layer_scale,
                    stochastic_depth_prob = stochastic_depth_prob,
                    stage_block_id        = stage_block_id,
                    total_stage_blocks    = total_stage_blocks,
                    norm                  = norm,
                )
            )
            stage_block_id += 1
        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.block(input)

# endregion


# region Model

class ConNeXt(base.ImageClassificationModel, ABC):
    """ConNeXt.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def init_weights(self, m: torch.nn.Module):
        """Initialize model's weights."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                torch.nn.init.trunc_normal_(m.conv.weight, std=0.02)
                if m.conv.bias is not None:
                    torch.nn.init.zeros_(m.conv.bias)
            else:
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        elif classname.find("Linear") != -1:
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict) \
            and self.weights["name"] in ["imagenet"]:
            state_dict = model.load_state_dict_from_path(
                model_dir=self.zoo_dir, **self.weights
            )
            model_state_dict = self.model.state_dict()
            """
            for k in self.model.state_dict().keys():
                print(f"\"{k}\": ")
            for k in state_dict.keys():
                print(f"\"{k}\"")
            """
            for k, v in state_dict.items():
                if "features.0"     in k:
                    k = k.replace("features.",    "")
                elif "features.1"   in k:
                    k = k.replace("features.1",   "1.block")
                elif "features.2.0" in k:
                    k = k.replace("features.2.0", "2")
                elif "features.2.1" in k:
                    k = k.replace("features.2.1", "3")
                elif "features.3"   in k:
                    k = k.replace("features.3",   "4.block")
                elif "features.4.0" in k:
                    k = k.replace("features.4.0", "5")
                elif "features.4.1" in k:
                    k = k.replace("features.4.1", "6")
                elif "features.5"   in k:
                    k = k.replace("features.5",   "7.block")
                elif "features.6.0" in k:
                    k = k.replace("features.6.0", "8")
                elif "features.6.1" in k:
                    k = k.replace("features.6.1", "9")
                elif "features.7"   in k:
                    k = k.replace("features.7",   "10.block")
                elif "classifier"   in k:
                    continue
                model_state_dict[k] = v
            if self.weights["num_classes"] == self.num_classes:
                model_state_dict["12.norm.bias"]     = state_dict["classifier.0.bias"]
                model_state_dict["12.norm.weight"]   = state_dict["classifier.0.weight"]
                model_state_dict["12.linear.bias"]   = state_dict["classifier.2.bias"]
                model_state_dict["12.linear.weight"] = state_dict["classifier.2.weight"]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()


@MODELS.register(name="convnext-base")
class ConNeXtBase(ConNeXt):
    """ConNeXt-Base.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
            "file_name"  : "convnext-base-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "convnext-base.yaml",
            "name"   : "convnext",
            "variant": "convnext-base"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="convnext-tiny")
class ConNeXtTiny(ConNeXt):
    """ConNeXt-Tiny.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
            "file_name"  : "convnext-tiny-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "convnext-tiny.yaml",
            "name"   : "convnext",
            "variant": "convnext-tiny"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="convnext-small")
class ConNeXtSmall(ConNeXt):
    """ConNeXt-Small.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/convnext_small-0c510722.pth",
            "file_name"  : "convnext-small-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "convnext-small.yaml",
            "name"   : "convnext",
            "variant": "convnext-small"
        }
        super().__init__(*args, **kwargs)
        
        
@MODELS.register(name="convnext-large")
class ConNeXtLarge(ConNeXt):
    """ConNeXt-Large.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
            "file_name"  : "convnext-large-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "convnext-large.yaml",
            "name"   : "convnext",
            "variant": "convnext-large"
        }
        super().__init__(*args, **kwargs)
        
# endregion
