#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements DenseNet models."""

from __future__ import annotations

__all__ = [
    "DenseNet121", "DenseNet161", "DenseNet169", "DenseNet201",
]

from abc import ABC

import torch

from mon.coreml import layer as mlayer, model as mmodel
from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.classify import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Model

class DenseNet(base.ImageClassificationModel, ABC):
    """DenseNet.
    
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
                torch.nn.init.kaiming_normal_(m.conv.weight)
            else:
                torch.nn.init.kaiming_normal_(m.weight)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif classname.find("Linear") != -1:
            torch.nn.init.constant_(m.bias, 0)
        
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
                if "features.conv0" in k:
                    k = k.replace("features.conv0", "0")
                elif "features.norm0" in k:
                    k = k.replace("features.norm0", "1")
                elif "features.denseblock1" in k:
                    k = k.replace("features.denseblock1", "4")
                    k = k.replace("norm.", "norm")
                    k = k.replace("conv.", "conv")
                elif "features.transition1" in k:
                    k = k.replace("features.transition1", "5")
                elif "features.denseblock2" in k:
                    k = k.replace("features.denseblock2", "6")
                    k = k.replace("norm.", "norm")
                    k = k.replace("conv.", "conv")
                elif "features.transition2" in k:
                    k = k.replace("features.transition2", "7")
                elif "features.denseblock3" in k:
                    k = k.replace("features.denseblock3", "8")
                    k = k.replace("norm.", "norm")
                    k = k.replace("conv.", "conv")
                elif "features.transition3" in k:
                    k = k.replace("features.transition3", "9")
                elif "features.denseblock4" in k:
                    k = k.replace("features.denseblock4", "10")
                    k = k.replace("norm.", "norm")
                    k = k.replace("conv.", "conv")
                elif "features.norm5" in k:
                    k = k.replace("features.norm5", "11")
                elif "classifier" in k:
                    continue
                model_state_dict[k] = v
            if self.weights["num_classes"] == self.num_classes:
                model_state_dict["13.linear.bias"]   = state_dict["classifier.bias"]
                model_state_dict["13.linear.weight"] = state_dict["classifier.weight"]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()


@MODELS.register(name="densenet121")
class DenseNet121(DenseNet):
    """DenseNet121.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/densenet121-a639ec97.pth",
            "file_name"  : "densenet121-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "densenet121.yaml",
            "name"   : "densenet",
            "variant": "densenet121"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="densenet161")
class DenseNet161(DenseNet):
    """DenseNet161.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/densenet161-8d451a50.pth",
            "file_name"  : "densenet161-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "densenet161.yaml",
            "name"   : "densenet",
            "variant": "densenet161"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="densenet169")
class DenseNet169(DenseNet):
    """DenseNet169.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            "file_name"  : "densenet169-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "densenet169.yaml",
            "name"   : "densenet",
            "variant": "densenet169"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="densenet201")
class DenseNet201(DenseNet):
    """DenseNet201.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/densenet201-c1103571.pth",
            "file_name"  : "densenet201-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "densenet201.yaml",
            "name"   : "densenet",
            "variant": "densenet201"
        }
        super().__init__(*args, **kwargs)
 
# endregion
