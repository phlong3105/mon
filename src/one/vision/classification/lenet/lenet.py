#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LeNet model.
"""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from torch import Tensor

from one.core import BACKBONES
from one.core import Indexes
from one.core import MODELS
from one.core import Pretrained
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "LeNet5"
]


# MARK: - LeNet5

@MODELS.register(name="lenet")
@MODELS.register(name="lenet5")
@BACKBONES.register(name="lenet")
@BACKBONES.register(name="lenet5")
class LeNet5(ImageClassifier):
    """`LeNet5 <https://en.wikipedia.org/wiki/LeNet>`. Input for LeNet-5
    is a 32x32 grayscale image.

    Args:
        basename (str, optional):
            Model basename. Default: `lenet`.
        name (str, optional):
            Name of the backbone. Default: `lenet`.
        num_classes (int, optional):
            Number of classes for classification. Default: `None`.
        out_indexes (Indexes):
            List of output tensors taken from specific layers' indexes.
            If `>= 0`, return the ith layer's output.
            If `-1`, return the final layer's output. Default: `-1`.
        pretrained (Pretrained):
            Use pretrained weights. If `True`, returns a model pre-trained on
            ImageNet. If `str`, load weights from saved file. Default: `True`.
            - If `True`, returns a model pre-trained on ImageNet.
            - If `str` and is a weight file(path), then load weights from
              saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
    """
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        basename   : Optional[str] = "lenet",
        name       : Optional[str] = "lenet",
        num_classes: Optional[int] = None,
        out_indexes: Indexes       = -1,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            basename    = basename,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        
        # NOTE: Features
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 120, kernel_size=(5, 5), stride=(1, 1)),
            nn.Tanh(),
        )
        
        # NOTE: Head (classifier)
        self.classifier = self.create_classifier(self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
    
    # MARK: Configure
    
    @staticmethod
    def create_classifier(num_classes: Optional[int]) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Sequential(
                nn.Linear(120, 84),
                nn.Tanh(),
                nn.Linear(84, num_classes),
            )
        else:
            classifier = nn.Identity()
        return classifier
    
    # MARK: Forward Pass
    
    def forward_once(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            x (Tensor):
                Input of shape [B, C, H, W].

        Returns:
            yhat (Tensor):
                Predictions.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
