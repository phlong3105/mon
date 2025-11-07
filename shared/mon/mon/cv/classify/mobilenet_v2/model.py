#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileNetV2 model for image classification.

References:
    - Paper: https://arxiv.org/abs/1801.04381
"""

__all__ = [
    "MobileNetV2",
]

from typing import Any

import box
from torchvision import models as tvm

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Model -----
@MODELS.register(name="mobilenet_v2", arch="mobilenet")
class MobileNetV2(tvm.MobileNetV2, nn.ModelMixin):
    """MobileNetV2 model for image classification.

    Args:
        num_classes: Number of output classes. Default: ``1000``.
        width_mult: Width multiplier for the network. Default: ``1.0``.
        dropout: Dropout rate for the model. Default: ``0.2``.
    
    References:
        - Paper: https://arxiv.org/abs/1801.04381
    """
    
    arch     : str          = "mobilenet"
    name     : str          = "mobilenet_v2"
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobilenet/mobilenet_v2/imagenet1k_v1/mobilenet_v2_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/mobilenet/mobilenet_v2/imagenet1k_v2/mobilenet_v2_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(
        self,
        weights    : Any   = "imagenet1k_v1",
        num_classes: int   = 1000,
        width_mult : float = 1.0,
        dropout    : float = 0.2,
        *args, **kwargs
    ):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(
            num_classes = num_classes,
            width_mult  = width_mult,
            dropout     = dropout,
            *args, **kwargs
        )
        if weights:
            self.load_state_dict(weights)
