#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements SqueezeNet models."""

from __future__ import annotations

__all__ = [
    "Fire", "SqueezeNet1_0", "SqueezeNet1_1",
]

from abc import ABC

import torch

from mon.globals import LAYERS, MODELS
from mon.vision import core, nn
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Module

@LAYERS.register()
class Fire(nn.LayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        squeeze_planes  : int,
        expand1x1_planes: int,
        expand3x3_planes: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.squeeze     = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = squeeze_planes,
            kernel_size  = 1,
        )
        self.squeeze_activation = nn.ReLU(inplace = True)
        self.expand1x1 = nn.Conv2d(
            in_channels  = squeeze_planes,
            out_channels = expand1x1_planes,
            kernel_size  = 1,
        )
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(
            in_channels  = squeeze_planes,
            out_channels = expand3x3_planes,
            kernel_size  = 3,
            padding      = 1,
        )
        self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x     = input
        x     = self.squeeze_activation(self.squeeze(x))
        y_1x1 = self.expand1x1_activation(self.expand1x1(x))
        y_3x3 = self.expand3x3_activation(self.expand3x3(x))
        y     = torch.cat([y_1x1, y_3x3], dim=1)
        return y
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        expand1x1_planes = args[2]
        expand3x3_planes = args[3]
        c2 = expand1x1_planes + expand3x3_planes
        ch.append(c2)
        return args, ch

# endregion


# region Model

class SqueezeNet(base.ImageClassificationModel, ABC):
    """SqueezeNet.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict) \
            and self.weights["name"] in ["imagenet"]:
            state_dict = nn.load_state_dict_from_path(
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
                if "features." in k:
                    k = k.replace("features.", "")
                else:
                    continue
                model_state_dict[k] = v
            if self.weights["num_classes"] == self.num_classes:
                model_state_dict["13.conv.bias"]   = state_dict["classifier.1.bias"]
                model_state_dict["13.conv.weight"] = state_dict["classifier.1.weight"]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()


@MODELS.register(name="squeezenet-1.0")
class SqueezeNet1_0(SqueezeNet):
    """SqueezeNet-1.0.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
            "file_name"  : "squeezenet-1.0-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "squeezenet-1.0.yaml",
            "name"   : "squeezenet",
            "variant": "squeezenet-1.0"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="squeezenet-1.1")
class SqueezeNet1_1(SqueezeNet):
    """SqueezeNet-1.1.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
            "file_name"  : "squeezenet-1.1-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "squeezenet-1.1.yaml",
            "name"   : "squeezenet",
            "variant": "squeezenet-1.1"
        }
        super().__init__(*args, **kwargs)
# endregion
