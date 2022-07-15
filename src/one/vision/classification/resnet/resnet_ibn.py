#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNet with Detection-Batch Normalization models.
"""

from __future__ import annotations

import math
from typing import Optional
from typing import Type

import torch
import torch.nn as nn
from torch import Tensor

from one.core import BACKBONES
from one.core import Indexes
from one.core import Int2T
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "ResNet50_IBN",
    "ResNet101_IBN",
    "ResNet152_IBN",
    "ResNet_IBN"
]


# MARK: - Modules

class IBN(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, planes: int):
        super().__init__()
        half1 	  = int(planes / 2)
        self.half = half1
        half2 	  = planes - half1
        self.IN   = nn.InstanceNorm2d(half1, affine=True)
        self.BN   = nn.BatchNorm2d(half2)

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        split = torch.split(x, self.half, 1)
        out1  = self.IN(split[0].contiguous())
        out2  = self.BN(split[1].contiguous())
        out   = torch.cat((out1, out2), 1)
        return out


class Bottleneck_IBN(nn.Module):
    
    expansion = 4
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        inplanes: int,
        planes	: int,
        ibn		: bool   = False,
        stride	: Int2T = 1,
        downsample 	     = None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1, 1), bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=(3, 3), stride=(stride, stride),
            padding=1, bias=False
        )
        self.bn2   		= nn.BatchNorm2d(planes)
        self.conv3 	    = nn.Conv2d(planes, planes * self.expansion,
                                      kernel_size=(1, 1), bias=False)
        self.bn3   	    = nn.BatchNorm2d(planes * self.expansion)
        self.relu  	    = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride 	= stride
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


# MARK: - ResNet_IBN

cfgs = {
    "resnet50_ibn_a": {
        "block": Bottleneck_IBN, "layers": [3, 4, 6, 3], "norm_layer": None
    },
    "resnet101_ibn_a": {
        "block": Bottleneck_IBN, "layers": [3, 4, 23, 3], "norm_layer": None
    },
    "resnet152_ibn_a": {
        "block": Bottleneck_IBN, "layers": [3, 8, 36, 3], "norm_layer": None
    },
}


@MODELS.register(name="resnet_ibn")
@BACKBONES.register(name="resnet_ibn")
class ResNet_IBN(ImageClassifier):
    """ResNet with Detection-Batch Normalization model.
    
    Args:
        basename (str, optional):
            Model basename. Default: `resnet`.
        name (str, optional):
            Name of the backbone. Default: `resnet_ibn`.
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
        # Hyperparameters
        last_stride: int,
        block                             = Bottleneck_IBN,
        layers     : ListOrTupleAnyT[int] = (3, 4, 6, 3),
        norm_layer : Optional[nn.Module]  = None,
        # BaseModel's args
        basename   : Optional[str] = "resnet",
        name       : Optional[str] = "resnet_ibn",
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
        # NOTE: Get Hyperparameters
        self.block      = block
        self.layers     = layers
        self.norm_layer = norm_layer
   
        if self.norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        self._norm_layer = self.norm_layer
        self.inplanes 	 = 64
        
        # NOTE: Features
        self.conv1   = nn.Conv2d(3, self.inplanes, (7, 7), (2, 2), padding=3,
                                 bias=False)
        self.bn1     = self.norm_layer(self.inplanes)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self.make_layer(self.block, 64,  self.layers[0])
        self.layer2  = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3  = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4  = self.make_layer(self.block, 512, self.layers[3], stride=last_stride)
        
        # NOTE: Head (Pool + Classifier layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc		 = self.create_classifier(self.block, self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.init_weights()
            
        # NOTE: Alias
        self.features = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,
            self.layer2, self.layer3, self.layer4,
        )
        self.classifier = self.fc
   
    # MARK: Configure
    
    def make_layer(
        self,
        block : Type[Bottleneck_IBN],
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=(1, 1), stride=(stride, stride), bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn    = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)
    
    @staticmethod
    def create_classifier(block, num_classes: Optional[int]) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Linear(512 * block.expansion, num_classes)
        else:
            classifier = nn.Identity()
        return classifier
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
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
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# MARK: - ResNet50_IBN

@MODELS.register(name="resnet50_ibn_a")
@BACKBONES.register(name="resnet50_ibn_a")
class ResNet50_IBN(ResNet_IBN):
    """ResNet-50 with Detection-Batch Normalization model."""
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnet50-19c8e357.pth",
            file_name="resnet50_ibn_a_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        last_stride: int,
        # BaseModel's args
        name       : Optional[str] = "resnet50_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnet50_ibn_a"] | kwargs
        super().__init__(
            last_stride = last_stride,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNet101_IBN

@MODELS.register(name="resnet101_ibn_a")
@BACKBONES.register(name="resnet101_ibn_a")
class ResNet101_IBN(ResNet_IBN):
    """ResNet-101 with Detection-Batch Normalization model."""
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
            file_name="resnet101_ibn_a_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        last_stride: int,
        # BaseModel's args
        name       : Optional[str] = "resnet101_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnet101_ibn_a"] | kwargs
        super().__init__(
            last_stride = last_stride,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNet152_IBN

@MODELS.register(name="resnet152_ibn_a")
@BACKBONES.register(name="resnet152_ibn_a")
class ResNet152_IBN(ResNet_IBN):
    """ResNet-152 with Detection-Batch Normalization model."""
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnet152-b121ed2d.pth",
            file_name="resnet152_ibn_a_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        last_stride: int,
        # BaseModel's args
        name       : Optional[str] = "resnet152_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnet152_ibn_a"] | kwargs
        super().__init__(
            last_stride = last_stride,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
