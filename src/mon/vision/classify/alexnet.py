#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements AlexNet models."""

from __future__ import annotations

__all__ = [
    "AlexNet",
]

import torchvision.models

from mon.globals import MODELS
from mon.vision import core
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="alexnet")
class AlexNet(base.ImageClassificationModel, torchvision.models.AlexNet):
    """AlexNet.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    zoo = {
        "imagenet": {
            "url"        : "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "path"       : "alexnet-imagenet.pth",
            "num_classes": 1000,
        },
    }

    def __init__(
        self,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
        *args, **kwargs
    ):
        super().__init__(
            num_classes = num_classes,
            dropout     = dropout,
            *args, **kwargs
        )

        
# endregion
