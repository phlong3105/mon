#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SqueezeNet models.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.squeezenet import Fire

from one.core import BACKBONES
from one.core import Indexes
from one.core import MODELS
from one.core import Pretrained
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "SqueezeNet",
    "SqueezeNet1_0",
    "SqueezeNet1_1"
]


# MARK: - SqueezeNet

@MODELS.register(name="squeezenet")
@BACKBONES.register(name="squeezenet")
class SqueezeNet(ImageClassifier):
    """SqueezeNet model.

    Args:
        basename (str, optional):
            Model basename. Default: `squeezenet`.
        name (str, optional):
            Name of the backbone. Default: `squeezenet`.
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
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        version    : str           = "1_0",
        # BaseModel's args
        basename   : Optional[str] = "squeezenet",
        name       : Optional[str] = "squeezenet",
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
        self.features = self.create_features(version)
        
        # NOTE: Classifier layer
        self.classifier = self.create_classifier(self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.init_weights()
            
    # MARK: Configure
    
    @staticmethod
    def create_features(version: str) -> nn.Sequential:
        if version == "1_0":
            features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2)),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))
        return features
    
    @staticmethod
    def create_classifier(num_classes: Optional[int]) -> nn.Module:
        if num_classes and num_classes > 0:
            final_conv = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
            classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            classifier = nn.Identity()
        return classifier
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final_conv:
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
        x = torch.flatten(x, 1)
        return x
    
    
# MARK: - SqueezeNet1_0

@MODELS.register(name="squeezenet1_0")
@BACKBONES.register(name="squeezenet1_0")
class SqueezeNet1_0(SqueezeNet):
    """SqueezeNet model architecture from the `SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size -
    <https://arxiv.org/abs/1602.07360>`_ paper. Frequired minimum input size
    of the model is 21x21.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth",
            file_name="squeezenet_1_0_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "squeezenet1_0",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            version     = "1_0",
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - SqueezeNet1_1

@MODELS.register(name="squeezenet1_1")
@BACKBONES.register(name="squeezenet1_1")
class SqueezeNet1_1(SqueezeNet):
    """SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth",
            file_name="squeezenet_1_1_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "squeezenet1_1",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            version     = "1_1",
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
