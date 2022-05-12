#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNext with Detection-Batch Normalization models.
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
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "ResNeXt50_IBN",
    "ResNeXt101_IBN",
    "ResNeXt152_IBN",
    "ResNeXt_IBN"
]


# MARK: - Modules

class IBN(nn.Module):
    """Detection-Batch Normalization layer from `Two at Once: Enhancing Learning
    and Generalization Capacities via IBN-Net
    - <https://arxiv.org/pdf/1807.09441.pdf>`
    
    Args:
        planes (int):
            Number of channels for the input image.
        ratio (float):
            Ratio of measurement normalization in the IBN layer.
    """
    
    # MARK: Magic Functions
    
    def __init__(self, planes: int, ratio: float = 0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN   = nn.InstanceNorm2d(self.half, affine=True)
        self.BN   = nn.BatchNorm2d(planes - self.half)

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        split = torch.split(x, self.half, 1)
        out1  = self.IN(split[0].contiguous())
        out2  = self.BN(split[1].contiguous())
        out   = torch.cat((out1, out2), 1)
        return out


class Bottleneck_IBN(nn.Module):
    """RexNeXt bottleneck type C.
    
    Args:
        inplanes (int):
            Input channel dimensionality.
        planes (int):
            Output channel dimensionality.
        base_width (int):
            Base width.
        cardinality (int):
            Num of convolution groups.
        stride (Int2T):
            Conv stride. Replaces pooling layer.
    """
    
    expansion = 4
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        inplanes   : int,
        planes     : int,
        base_width : int,
        cardinality: int,
        stride     : Int2T             = 1,
        downsample : Optional[Callable] = None,
        ibn        : Optional[str]      = None
    ):
        super().__init__()
        
        D          = int(math.floor(planes * (base_width / 64)))
        C          = cardinality
        self.conv1 = nn.Conv2d(inplanes, D * C, kernel_size=(1, 1),
                               stride=(1, 1), padding=0, bias=False)
        if ibn == "a":
            self.bn1 = IBN(D * C)
        else:
            self.bn1 = nn.BatchNorm2d(D * C)
        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=(3, 3), stride=stride,
                               padding=1, groups=C, bias=False)
        self.bn2   = nn.BatchNorm2d(D * C)
        self.conv3 = nn.Conv2d(D * C, planes * 4, kernel_size=(1, 1),
                               stride=(1, 1), padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * 4)
        self.relu  = nn.ReLU(inplace=True)
        
        self.downsample = downsample
    
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


# MARK: - ResNeXt_IBN

cfgs = {
    "resnext50_ibn_a": {
        "block": Bottleneck_IBN, "layers": [3, 4, 6, 3], "base_width": 4,
        "cardinality": 32, "ibn_cfg": ("a", "a", "a", None)
    },
    "resnext101_ibn_a": {
        "block": Bottleneck_IBN, "layers": [3, 4, 23, 3], "base_width": 4,
        "cardinality": 32, "ibn_cfg": ("a", "a", "a", None)
    },
    "resnext152_ibn_a": {
        "block": Bottleneck_IBN, "layers": [3, 8, 36, 3], "base_width": 4,
        "cardinality": 32, "ibn_cfg": ("a", "a", "a", None)
    },
}


@MODELS.register(name="resnext_ibn")
@BACKBONES.register(name="resnext_ibn")
class ResNeXt_IBN(ImageClassifier):
    """ResNeXt with Detection-Batch Normalization model.
    
    Args:
        basename (str, optional):
            Model basename. Default: `resnext`.
        name (str, optional):
            Name of the backbone. Default: `resnext_ibn`.
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
    
    Attributes:
        cfg (Config, optional):
            Config to build the model's layers.
            - If `str`, use the corresponding config from the predefined
              config dict. This is used to build the model dynamically.
            - If a file or filepath, it leads to the external config file that
              is used to build the model dynamically.
            - If `list`, then each element in the list is the corresponding
              config for each layer in the model. This is used to build the
              model dynamically.
            - If `dict`, it usually contains the hyperparameters used to
              build the model manually in the code.
            - If `None`, then you should manually define the model.
            Remark: You have 5 ways to build the model, so choose the style
            that you like.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        block                             = Bottleneck_IBN,
        layers     : ListOrTupleAnyT[int] = (3, 8, 36 , 3),
        base_width : int                  = 4,
        cardinality: int                  = 32,
        ibn_cfg    : tuple                = ("a", "a", "a", None),
        # BaseModel's args
        basename   : Optional[str] = "resnext",
        name       : Optional[str] = "resnext_ibn",
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
        self.block       = block
        self.layers      = layers
        self.base_width  = base_width
        self.cardinality = cardinality
        self.ibn_cfg     = ibn_cfg
        self.inplanes    = 64
        self.output_size = 64
        
        # NOTE: Features
        self.conv1    = nn.Conv2d(3, 64, (7, 7), (2, 2), 3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1   = self.make_layer(
            self.block, 64, self.layers[0], ibn=self.ibn_cfg[0]
        )
        self.layer2   = self.make_layer(
            self.block, 128, self.layers[1], stride=2, ibn=self.ibn_cfg[1]
        )
        self.layer3   = self.make_layer(
            self.block, 256, self.layers[2], stride=2, ibn=self.ibn_cfg[2]
        )
        self.layer4   = self.make_layer(
            self.block, 512, self.layers[3], stride=1, ibn=self.ibn_cfg[3]
        )
        
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
        stride: int           = 1,
        ibn   : Optional[str] = None
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion, kernel_size=(1, 1),
                    stride=(stride, stride), bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes, planes, self.base_width, self.cardinality, stride,
            downsample, ibn
        ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, self.base_width, self.cardinality, 1,
                None, ibn
            ))

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
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
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


# MARK: - ResNeXt50_IBN

@MODELS.register(name="resnext50_ibn_a")
@BACKBONES.register(name="resnext50_ibn_a")
class ResNeXt50_IBN(ResNeXt_IBN):
    """ResNeXt-50 with Detection-Batch Normalization model."""
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnext50_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnext50_ibn_a"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNeXt101_IBN

@MODELS.register(name="resnext101_ibn_a")
@BACKBONES.register(name="resnext101_ibn_a")
class ResNeXt101_IBN(ResNeXt_IBN):
    """ResNeXt-101 with Detection-Batch Normalization model."""
    
    model_zoo = {
        "imagenet": dict(
            path="https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth",
            file_name="resnext101_ibn_a_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnext101_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnext101_ibn_a"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNeXt152_IBN

@MODELS.register(name="resnext152_ibn_a")
@BACKBONES.register(name="resnext152_ibn_a")
class ResNeXt152_IBN(ResNeXt_IBN):
    """ResNeXt-152 with Detection-Batch Normalization model."""
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnext152_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnext152_ibn_a"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
