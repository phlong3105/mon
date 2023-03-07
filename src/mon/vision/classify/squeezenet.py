#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements SqueezeNet models."""

from __future__ import annotations

__all__ = [
    "SqueezeNet1_0", "SqueezeNet1_1",
]

from abc import ABC

import torch

from mon.coreml import layer as mlayer, model as mmodel
from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.classify import base

_current_dir = pathlib.Path(__file__).absolute().parent


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
            state_dict = mmodel.load_state_dict_from_path(
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
