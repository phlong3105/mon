#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ShuffleNetv2 models."""

from __future__ import annotations

__all__ = [
    "ShuffleNetV2_x0_5", "ShuffleNetV2_x1_0", "ShuffleNetV2_x1_5",
    "ShuffleNetV2_x2_0",
]

from abc import ABC

from mon.coreml import model as mmodel
from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.classify import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Model

class ShuffleNetV2(base.ImageClassificationModel, ABC):
    """ShuffleNetV2.
    
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
                if "stage2.0" in k:
                    k = k.replace("stage2.0", "4")
                elif "stage2.1" in k:
                    k = k.replace("stage2.1", "5")
                elif "stage2.2" in k:
                    k = k.replace("stage2.2", "6")
                elif "stage2.3" in k:
                    k = k.replace("stage2.3", "7")
                elif "stage3.0" in k:
                    k = k.replace("stage3.0", "8")
                elif "stage3.1" in k:
                    k = k.replace("stage3.1", "9")
                elif "stage3.2" in k:
                    k = k.replace("stage3.2", "10")
                elif "stage3.3" in k:
                    k = k.replace("stage3.3", "11")
                elif "stage3.4" in k:
                    k = k.replace("stage3.4", "12")
                elif "stage3.5" in k:
                    k = k.replace("stage3.5", "13")
                elif "stage3.6" in k:
                    k = k.replace("stage3.6", "14")
                elif "stage3.7" in k:
                    k = k.replace("stage3.7", "15")
                elif "stage4.0" in k:
                    k = k.replace("stage4.0", "16")
                elif "stage4.1" in k:
                    k = k.replace("stage4.1", "17")
                elif "stage4.2" in k:
                    k = k.replace("stage4.2", "18")
                elif "stage4.3" in k:
                    k = k.replace("stage4.3", "19")
                elif "stage5.0" in k:
                    k = k.replace("stage5.0", "20")
                elif "stage5.1" in k:
                    k = k.replace("stage5.1", "21")
                else:
                    continue
                model_state_dict[k] = v
            model_state_dict["0.weight"]       = state_dict["conv1.0.weight"]
            model_state_dict["1.bias"]         = state_dict["conv1.1.bias"]
            model_state_dict["1.running_mean"] = state_dict["conv1.1.running_mean"]
            model_state_dict["1.running_var"]  = state_dict["conv1.1.running_var"]
            model_state_dict["1.weight"]       = state_dict["conv1.1.weight"]
            if self.weights["num_classes"] == self.num_classes:
                model_state_dict["23.linear.bias"]   = state_dict["fc.bias"]
                model_state_dict["23.linear.weight"] = state_dict["fc.weight"]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()


@MODELS.register(name="shufflenet-v2-x0.5")
class ShuffleNetV2_x0_5(ShuffleNetV2):
    """ShuffleNetV2-x0.5.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            "file_name"  : "shufflenet-v2-x0.5-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "shufflenet-v2-x0.5.yaml",
            "name"   : "shufflenet-v2",
            "variant": "shufflenet-v2-x0.5"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="shufflenet-v2-x1.0")
class ShuffleNetV2_x1_0(ShuffleNetV2):
    """ShuffleNetV2-x1.0.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            "file_name"  : "shufflenet-v2-x1.0-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "shufflenet-v2-x1.0.yaml",
            "name"   : "shufflenet-v2",
            "variant": "shufflenet-v2-x1.0"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="shufflenet-v2-x1.5")
class ShuffleNetV2_x1_5(ShuffleNetV2):
    """ShuffleNetV2-x1.5.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
            "file_name"  : "shufflenet-v2-x1.5-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "shufflenet-v2-x1.5.yaml",
            "name"   : "shufflenet-v2",
            "variant": "shufflenet-v2-x1.5"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="shufflenet-v2-x2.0")
class ShuffleNetV2_x2_0(ShuffleNetV2):
    """ShuffleNetV2-x2.0.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
            "file_name"  : "shufflenet-v2-x2.0-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "shufflenet-v2-x2.0.yaml",
            "name"   : "shufflenet-v2",
            "variant": "shufflenet-v2-x2.0"
        }
        super().__init__(*args, **kwargs)
 
# endregion
