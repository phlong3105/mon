#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ResNeXt models."""

from __future__ import annotations

__all__ = [
    "ResNeXt101_32X8D", "ResNeXt101_64X4D", "ResNeXt50_32X4D",
]

from abc import ABC

import torch

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.classify import base, resnet

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Model

class ResNeXt(base.ImageClassificationModel, ABC):
    """ResNeXt.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    zoo = {}

    def init_weights(self, m: torch.nn.Module):
        """Initialize model's weights."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                torch.nn.init.kaiming_normal_(m.conv.weight, mode="fan_out", nonlinearity="relu")
            else:
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif classname.find("GroupNorm") != -1:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

        zero_init_residual = self.cfg["zero_init_residual"]
        if zero_init_residual:
            if isinstance(m, resnet.ResNetBottleneck) and m.bn3.weight is not None:
                torch.nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, resnet.ResNetBottleneck) and m.bn2.weight is not None:
                torch.nn.init.constant_(m.bn2.weight, 0)
    
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
                if "layer1" in k:
                    k = k.replace("layer1", "4.convs")
                elif "layer2" in k:
                    k = k.replace("layer2", "5.convs")
                elif "layer3" in k:
                    k = k.replace("layer3", "6.convs")
                elif "layer4" in k:
                    k = k.replace("layer4", "7.convs")
                else:
                    continue
                model_state_dict[k] = v
            model_state_dict["0.weight"]       = state_dict["conv1.weight"]
            model_state_dict["1.weight"]       = state_dict["bn1.weight"]
            model_state_dict["1.bias"]         = state_dict["bn1.bias"]
            model_state_dict["1.running_mean"] = state_dict["bn1.running_mean"]
            model_state_dict["1.running_var"]  = state_dict["bn1.running_var"]
            if self.weights["num_classes"] == self.num_classes:
                model_state_dict["9.linear.weight"] = state_dict["fc.weight"]
                model_state_dict["9.linear.bias"]   = state_dict["fc.bias"]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()


@MODELS.register(name="resnext50-32x4d")
class ResNeXt50_32X4D(ResNeXt):
    """ResNeXt50-32x4d.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
            "file_name"  : "resnext50-32x4d-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "resnext50-32x4d.yaml",
            "name"   : "resnext",
            "variant": "resnext50-32x4d"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="resnext101-32x8d")
class ResNeXt101_32X8D(ResNeXt):
    """ResNeXt101-32X8D.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    zoo = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
            "file_name"  : "resnext101-32x8d-imagenet.pth",
            "num_classes": 1000,
        },
    }

    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "resnext101-32x8d.yaml",
            "name"   : "resnext",
            "variant": "resnext101-32x8d"
        }
        super().__init__(*args, **kwargs)


@MODELS.register(name="resnext101-64x4d")
class ResNeXt101_64X4D(ResNeXt):
    """ResNeXt101-64X4D.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "imagenet": {
            "name"       : "imagenet",
            "path"       : "https://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
            "file_name"  : "resnext101-64x4d-imagenet.pth",
            "num_classes": 1000,
        },
    }
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "resnext101-64x4d.yaml",
            "name"   : "resnext",
            "variant": "resnext101-64x4d"
        }
        super().__init__(*args, **kwargs)
        
# endregion
