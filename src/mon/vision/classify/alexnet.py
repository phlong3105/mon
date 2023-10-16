#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements AlexNet models."""

from __future__ import annotations

__all__ = [
    "AlexNet",
]

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="alexnet")
class AlexNet(base.ImageClassificationModel):
    """AlexNet.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "file_name"  : "alexnet-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {
        "backbone": {
            "0.weight"         : "features.0.weight",
            "0.bias"           : "features.0.bias",
            "3.weight"         : "features.3.weight",
            "3.bias"           : "features.3.bias",
            "6.weight"         : "features.6.weight",
            "6.bias"           : "features.6.bias",
            "8.weight"         : "features.8.weight",
            "8.bias"           : "features.8.bias",
            "10.weight"        : "features.10.weight",
            "10.bias"          : "features.10.bias",
            "14.linear1.weight": "classifier.1.weight",
            "14.linear1.bias"  : "classifier.1.bias",
            "14.linear2.weight": "classifier.4.weight",
            "14.linear2.bias"  : "classifier.4.bias",
        },
        "head": {
            "14.linear3.weight": "classifier.6.weight",
            "14.linear3.bias"  : "classifier.6.bias",
        },
    }

    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "alexnet.yaml",
            "name"   : "alexnet",
            "variant": "alexnet"
        }
        super().__init__(*args, **kwargs)
    
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
            for k, v in self.map_weights["backbone"].items():
                model_state_dict[k] = state_dict[v]
            if self.weights["num_classes"] == self.num_classes:
                for k, v in self.map_weights["head"].items():
                    model_state_dict[k] = state_dict[v]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()
        
# endregion
