#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the AlexNet model for image classification."""

__all__ = [
    "AlexNet",
]

from typing import Any

import box
from torchvision import models as tvm

from mon import nn
from mon.core import MLType, MODELS, Path, ROOT_DIR, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(variant="alexnet", name="alexnet")
class AlexNet(tvm.AlexNet, nn.ModelMetadataMixin):
    """AlexNet model for image classification.
    
    Args:
        num_classes: Number of output classes. Default: ``1000``.
        dropout: Dropout rate for the model. Default: ``0.5``.
    """
    
    _arch     : str          = "alexnet"
    _name     : str          = "alexnet",
    _tasks    : list[Task]   = [Task.CLASSIFY]
    _mltypes  : list[MLType] = [MLType.SUPERVISED]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box({
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
