#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileNetV3 model for image classification.

References:
    - Paper: https://arxiv.org/abs/1905.02244
"""

__all__ = [
    "MobileNetV3Large",
    "MobileNetV3Small",
]

import abc
from typing import Any

import box
from torchvision import models as tvm
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# --- Model ---
class MobileNetV3(tvm.MobileNetV3, nn.ModelMetadataMixin, abc.ABC):
    """MobileNetV3 model for image classification.

    References:
        - Paper: https://arxiv.org/abs/1905.02244
    """
    
    _arch     : str          = "mobilenet"
    _name     : str          = "mobilenet_v3"
    _tasks    : list[Task]   = [Task.CLASSIFY]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
    

@MODELS.register(name="mobilenet_v3_large", arch="mobilenet")
class MobileNetV3Large(MobileNetV3):
    
    _name: str  = "mobilenet_v3_large"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobilenet/mobilenet_v3_large/imagenet1k_v1/mobilenet_v3_large_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobilenet/mobilenet_v3_large/imagenet1k_v2/mobilenet_v3_large_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            last_channel              = last_channel,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )
        
        
@MODELS.register(name="mobilenet_v3_small", arch="mobilenet")
class MobileNetV3Small(MobileNetV3):
    """MobileNetV3-Small model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
        dropout: Dropout rate for the model. Default: ``0.2``.
    """
    
    _name: str  = "mobilenet_v3_small"
    _zoo : dict = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobilenet/mobilenet_v3_small/imagenet1k_v1/mobilenet_v3_small_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(self, weights: Any = "imagenet1k_v1", num_classes: int = 1000, *args, **kwargs):
        inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
        super().__init__(
            inverted_residual_setting = inverted_residual_setting,
            last_channel              = last_channel,
            weights                   = weights,
            num_classes               = num_classes,
            *args, **kwargs
        )
