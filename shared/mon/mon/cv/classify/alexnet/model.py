#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements AlexNet models for image classification."""

__all__ = [
    "AlexNet",
]

from typing import Any

import box
from torchvision import models as tvm

import mon.nn as nn
import mon.nn as nn
from mon.constants import MODELS, ROOT_DIR
from mon.core import MLType, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(name="alexnet", arch="alexnet")
class AlexNet(tvm.AlexNet, nn.ModelMixin):
    """AlexNet model for image classification.
    
    Args:
        num_classes: Number of output classes. Default: ``1000``.
        dropout: Dropout rate for the model. Default: ``0.5``.
    """
    
    arch     : str          = "alexnet"
    name     : str          = "alexnet",
    tasks    : list[Task]   = [Task.CLASSIFY]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box({
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "path"       : ROOT_DIR / "zoo/cv/classify/alexnet/alexnet/imagenet1k_v1/alexnet_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    })
    
    def __init__(
        self,
        weights    : Any   = "imagenet1k_v1",
        num_classes: int   = 1000,
        dropout    : float = 0.5,
        *args, **kwargs
    ):
        weights, path, num_classes = self.parse_weights(weights, num_classes)
        super().__init__(num_classes=num_classes, dropout=dropout, *args, **kwargs)
        if weights:
            self.load_state_dict(weights)
