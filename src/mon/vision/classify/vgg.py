#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements VGG models."""

from __future__ import annotations

__all__ = [
    "VGG11", "VGG11BN", "VGG13", "VGG13BN", "VGG16", "VGG16BN", "VGG19",
    "VGG19BN",
]

from abc import ABC

import torch

from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.classify import base
from mon.vision.ml import model

_current_dir = pathlib.Path(__file__).absolute().parent


# region Model

class VGG(base.ImageClassificationModel, ABC):
    """VGG.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def init_weights(self, m: torch.nn.Module):
        """Initialize model's weights."""
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, 0, 0.01)
            torch.nn.init.constant_(m.bias, 0)
    
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
                k = k.replace("features.", "")
                if k in model_state_dict:
                    model_state_dict[k] = v
            if "22.linear1.weight" in model_state_dict:  # vgg11
                model_state_dict["22.linear1.weight"] = state_dict["classifier.0.weight"]
                model_state_dict["22.linear1.bias"]   = state_dict["classifier.0.bias"]
                model_state_dict["22.linear2.weight"] = state_dict["classifier.3.weight"]
                model_state_dict["22.linear2.bias"]   = state_dict["classifier.3.bias"]
                if self.weights["num_classes"] == self.num_classes:
                    model_state_dict["22.linear3.weight"] = state_dict["classifier.6.weight"]
                    model_state_dict["22.linear3.bias"]   = state_dict["classifier.6.bias"]
            elif "30.linear1.weight" in model_state_dict:  # vgg11-bn
                model_state_dict["30.linear1.weight"] = state_dict["classifier.0.weight"]
                model_state_dict["30.linear1.bias"]   = state_dict["classifier.0.bias"]
                model_state_dict["30.linear2.weight"] = state_dict["classifier.3.weight"]
                model_state_dict["30.linear2.bias"]   = state_dict["classifier.3.bias"]
                if self.weights["num_classes"] == self.num_classes:
                    model_state_dict["30.linear3.weight"] = state_dict["classifier.6.weight"]
                    model_state_dict["30.linear3.bias"]   = state_dict["classifier.6.bias"]
            elif "26.linear1.weight" in model_state_dict:  # vgg13
                model_state_dict["26.linear1.weight"] = state_dict["classifier.0.weight"]
                model_state_dict["26.linear1.bias"]   = state_dict["classifier.0.bias"]
                model_state_dict["26.linear2.weight"] = state_dict["classifier.3.weight"]
                model_state_dict["26.linear2.bias"]   = state_dict["classifier.3.bias"]
                if self.weights["num_classes"] == self.num_classes:
                    model_state_dict["26.linear3.weight"] = state_dict["classifier.6.weight"]
                    model_state_dict["26.linear3.bias"]   = state_dict["classifier.6.bias"]
            elif "36.linear1.weight" in model_state_dict:  # vgg13-bn
                model_state_dict["36.linear1.weight"] = state_dict["classifier.0.weight"]
                model_state_dict["36.linear1.bias"]   = state_dict["classifier.0.bias"]
                model_state_dict["36.linear2.weight"] = state_dict["classifier.3.weight"]
                model_state_dict["36.linear2.bias"]   = state_dict["classifier.3.bias"]
                if self.weights["num_classes"] == self.num_classes:
                    model_state_dict["36.linear3.weight"] = state_dict["classifier.6.weight"]
                    model_state_dict["36.linear3.bias"]   = state_dict["classifier.6.bias"]
            elif "32.linear1.weight" in model_state_dict:  # vgg16
                model_state_dict["32.linear1.weight"] = state_dict["classifier.0.weight"]
                model_state_dict["32.linear1.bias"]   = state_dict["classifier.0.bias"]
                model_state_dict["32.linear2.weight"] = state_dict["classifier.3.weight"]
                model_state_dict["32.linear2.bias"]   = state_dict["classifier.3.bias"]
                if self.weights["num_classes"] == self.num_classes:
                    model_state_dict["32.linear3.weight"] = state_dict["classifier.6.weight"]
                    model_state_dict["32.linear3.bias"]   = state_dict["classifier.6.bias"]
            elif "45.linear1.weight" in model_state_dict:  # vgg16-bn
                model_state_dict["45.linear1.weight"] = state_dict["classifier.0.weight"]
                model_state_dict["45.linear1.bias"]   = state_dict["classifier.0.bias"]
                model_state_dict["45.linear2.weight"] = state_dict["classifier.3.weight"]
                model_state_dict["45.linear2.bias"]   = state_dict["classifier.3.bias"]
                if self.weights["num_classes"] == self.num_classes:
                    model_state_dict["45.linear3.weight"] = state_dict["classifier.6.weight"]
                    model_state_dict["45.linear3.bias"]   = state_dict["classifier.6.bias"]
            elif "38.linear1.weight" in model_state_dict:  # vgg19
                model_state_dict["38.linear1.weight"] = state_dict["classifier.0.weight"]
                model_state_dict["38.linear1.bias"]   = state_dict["classifier.0.bias"]
                model_state_dict["38.linear2.weight"] = state_dict["classifier.3.weight"]
                model_state_dict["38.linear2.bias"]   = state_dict["classifier.3.bias"]
                if self.weights["num_classes"] == self.num_classes:
                    model_state_dict["38.linear3.weight"] = state_dict["classifier.6.weight"]
                    model_state_dict["38.linear3.bias"]   = state_dict["classifier.6.bias"]
            elif "54.linear1.weight" in model_state_dict:  # vgg19-bn
                model_state_dict["54.linear1.weight"] = state_dict["classifier.0.weight"]
                model_state_dict["54.linear1.bias"]   = state_dict["classifier.0.bias"]
                model_state_dict["54.linear2.weight"] = state_dict["classifier.3.weight"]
                model_state_dict["54.linear2.bias"]   = state_dict["classifier.3.bias"]
                if self.weights["num_classes"] == self.num_classes:
                    model_state_dict["54.linear3.weight"] = state_dict["classifier.6.weight"]
                    model_state_dict["54.linear3.bias"]   = state_dict["classifier.6.bias"]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()


@MODELS.register(name="vgg11")
class VGG11(VGG):
    """VGG11.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/vgg11-8a719046.pth",
            "file_name"  : "vgg11-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "vgg11.yaml",
            "name"   : "vgg",
            "variant": "vgg11"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="vgg11-bn")
class VGG11BN(VGG):
    """VGG11-BN.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            "file_name"  : "vgg11-bn-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "vgg11-bn.yaml",
            "name"   : "vgg",
            "variant": "vgg11-bn"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="vgg13")
class VGG13(VGG):
    """VGG13.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/vgg13-19584684.pth",
            "file_name"  : "vgg13-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "vgg13.yaml",
            "name"   : "vgg",
            "variant": "vgg13"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="vgg13-bn")
class VGG13BN(VGG):
    """VGG13-BN.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            "file_name"  : "vgg13-bn-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "vgg13-bn.yaml",
            "name"   : "vgg",
            "variant": "vgg13-bn"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="vgg16")
class VGG16(VGG):
    """VGG16.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/vgg16-397923af.pth",
            "file_name"  : "vgg16-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "vgg16.yaml",
            "name"   : "vgg",
            "variant": "vgg16"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="vgg16-bn")
class VGG16BN(VGG):
    """VGG16-BN.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "file_name"  : "vgg16-bn-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "vgg16-bn.yaml",
            "name"   : "vgg",
            "variant": "vgg16-bn"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="vgg19")
class VGG19(VGG):
    """VGG19.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "file_name"  : "vgg19-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "vgg19.yaml",
            "name"   : "vgg",
            "variant": "vgg19"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="vgg19-bn")
class VGG19BN(VGG):
    """VGG19-BN.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
            "file_name"  : "vgg19-bn-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "vgg19-bn.yaml",
            "name"   : "vgg",
            "variant": "vgg19"
        }
        super().__init__(*args, **kwargs)
    
# endregion
