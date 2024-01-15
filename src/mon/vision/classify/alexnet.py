#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements AlexNet models."""

from __future__ import annotations

__all__ = [
    "AlexNet",
]

import torch

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="alexnet")
class AlexNet(base.ImageClassificationModel):
    """AlexNet.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
            "path"       : "alexnet-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }

    def __init__(
        self,
        num_classes: int   = 1000,
        dropout    : float = 0.5,
        name       : str   = "alexnet",
        *args, **kwargs
    ):
        super().__init__(
            num_classes = num_classes,
            name        = name,
            *args, **kwargs
        )
        self.dropout  = dropout
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool    = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes),
        )

    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y


# endregion
