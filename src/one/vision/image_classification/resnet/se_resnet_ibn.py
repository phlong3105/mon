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
from one.core import Callable
from one.core import Indexes
from one.core import Int2T
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.nn import SELayer
from one.vision.image_classification.image_classifier import ImageClassifier

__all__ = [
    "SE_ResNet50_IBN",
    "SE_ResNet101_IBN",
    "SE_ResNet152_IBN",
    "SE_ResNet_IBN",
    "SEBasicBlock",
    "SEBottleneck"
]


# MARK: - Modules

def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=(3, 3), stride=(stride, stride),
        padding=1, bias=False
    )


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


class SEBasicBlock(nn.Module):

    expansion = 1
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        inplanes  : int,
        planes    : int,
        stride    : Int2T             = 1,
        downsample: Optional[Callable] = None,
        reduction : int                = 16
    ):
        super().__init__()
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm2d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes, 1)
        self.bn2        = nn.BatchNorm2d(planes)
        self.se         = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride     = stride
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        
        out      = self.conv1(x)
        out      = self.bn1(out)
        out      = self.relu(out)
        
        out      = self.conv2(out)
        out      = self.bn2(out)
        out      = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out  = self.relu(out)
        return out


class SEBottleneck(nn.Module):
    
    expansion = 4

    # MARK: Magic Functions
    
    def __init__(
        self,
        inplanes  : int,
        planes    : int,
        stride    : Int2T             = 1,
        downsample: Optional[Callable] = None,
        ibn       : bool               = False,
        reduction : int                = 16
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=(1, 1), bias=False
        )
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
            
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=(3, 3), stride=(stride, stride),
            padding=1, bias=False
        )
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes * 4, kernel_size=(1, 1),
                                    bias=False)
        self.bn3        = nn.BatchNorm2d(planes * 4)
        self.relu       = nn.ReLU(inplace=True)
        self.se         = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride     = stride

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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out  = self.relu(out)
        return out


# MARK: - SE_ResNet_IBN

cfgs = {
	"se_resnet50_ibn_a" : {"block": SEBottleneck, "layers": [3, 4, 6, 3]},
    "se_resnet101_ibn_a": {"block": SEBottleneck, "layers": [3, 4, 23, 3]},
    "se_resnet152_ibn_a": {"block": SEBottleneck, "layers": [3, 8, 36, 3]},
}


@MODELS.register(name="se_resnet_ibn")
@BACKBONES.register(name="se_resnet_ibn")
class SE_ResNet_IBN(ImageClassifier):
    """

    Args:
        basename (str, optional):
            Model basename. Default: `resnet`.
        name (str, optional):
            Name of the backbone. Default: `se_resnet_ibn`.
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
        block                             = SEBottleneck,
        layers     : ListOrTupleAnyT[int] = (3, 4, 6, 3),
        # BaseModel's args
        basename   : Optional[str] = "resnet",
        name       : Optional[str] = "se_resnet_ibn",
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
        self.block    = block
        self.layers   = layers
        self.inplanes = 64
        
        # NOTE: Features
        self.conv1   = nn.Conv2d(3, 64, (7, 7), (2, 2), padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self.make_layer(self.block, 64,  self.layers[0])
        self.layer2  = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.layer3  = self.make_layer(self.block, 256, self.layers[2], stride=2)
        self.layer4  = self.make_layer(self.block, 512, self.layers[3], stride=1)
        
        # NOTE: Head (Pool + Classifier layer)
        self.avgpool = nn.AvgPool2d(7)
        self.fc      = self.create_classifier(self.block, self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.init_weights()
        
        # NOTE: Alias
        self.features = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4,
        )
        self.classifier = self.fc
     
    # MARK: Configure
    
    def make_layer(
        self,
        block : Type[SEBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
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
        layers.append(block(self.inplanes, planes, stride, downsample, ibn=ibn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, ibn=ibn))
    
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_classifier(block, num_classes: Optional[int]) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Linear(512 * block.expansion, num_classes)
        else:
            classifier = nn.Identity()
        return classifier
    
    def init_weights(self):
        self.conv1.weight.data.normal_(0, math.sqrt(2.0 / (7 * 7 * 64)))
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


# MARK: - SE_ResNet50_IBN

@MODELS.register(name="se_resnet50_ibn")
@BACKBONES.register(name="se_resnet50_ibn")
class SE_ResNet50_IBN(SE_ResNet_IBN):
    """ResNet-50 with Squeeze and Excite and Detection-Batch Normalization model.
    """
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "se_resnet50_ibn",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["se_resnet50_ibn"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)


# MARK: - SE_ResNet101_IBN

@MODELS.register(name="se_resnet101_ibn")
@BACKBONES.register(name="se_resnet101_ibn")
class SE_ResNet101_IBN(SE_ResNet_IBN):
    """ResNet-101 with Squeeze and Excite and Detection-Batch Normalization
    model.
    """
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "se_resnet101_ibn",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["se_resnet101_ibn"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)


# MARK: - SE_ResNet152_IBN

@MODELS.register(name="se_resnet152_ibn")
@BACKBONES.register(name="se_resnet152_ibn")
class SE_ResNet152_IBN(SE_ResNet_IBN):
    """ResNet-152 with Squeeze and Excite and Detection-Batch Normalization
    model.
    """
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "se_resnet152_ibn",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["se_resnet152_ibn"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
