#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DenseNet with Detection-Batch Normalization models.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from one.core import BACKBONES
from one.core import Indexes
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
	"DenseNet121_IBN_A",
    "DenseNet161_IBN_A",
    "DenseNet169_IBN_A",
    "DenseNet201_IBN_A",
    "DenseNet_IBN",
    "IBN"
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
        self.half = int(planes * (1 - ratio))
        self.BN   = nn.BatchNorm2d(self.half)
        self.IN   = nn.InstanceNorm2d(planes - self.half, affine = True)
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        split = torch.split(x, self.half, 1)
        out1  = self.BN(split[0].contiguous())
        out2  = self.IN(split[1].contiguous())
        out   = torch.cat((out1, out2), 1)
        return out
    

class _DenseLayer(nn.Sequential):
    """
    
    Args:
        growth_rate (int):
            How many filtering to add each layer (`k` in paper).
        bn_size (int):
            Multiplicative factor for number of bottle neck layers (i.e.
            bn_size * k features in the bottleneck layer).
        drop_rate (float):
            Dropout rate after each dense layer.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_input_features: int,
        growth_rate       : int,
        bn_size           : int,
        drop_rate         : float,
        ibn               : bool
    ):
        super().__init__()
        if ibn:
            self.add_module("norm1", IBN(num_input_features, 0.4))
        else:
            self.add_module("norm1", nn.BatchNorm2d(num_input_features))
       
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1", nn.Conv2d(
                num_input_features, bn_size * growth_rate, kernel_size=(1, 1),
                stride=(1, 1), bias=False
            )
        )
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2", nn.Conv2d(
                bn_size * growth_rate, growth_rate, kernel_size=(3, 3),
                stride=(1, 1), padding=1, bias=False
            )
        )
        self.drop_rate = drop_rate

    # MARK: Forward Pass
    
    def forward(self, x: Tensor) -> Tensor:
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    """
    
    Args:
        bn_size (int):
            Multiplicative factor for number of bottle neck layers (i.e.
            bn_size * k features in the bottleneck layer).
        growth_rate (int):
            How many filtering to add each layer (`k` in paper).
        drop_rate (float):
            Dropout rate after each dense layer.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        num_layers        : int,
        num_input_features: int,
        bn_size           : int,
        growth_rate       : int,
        drop_rate         : float,
        ibn               : bool
    ):
        super().__init__()
        for i in range(num_layers):
            if ibn and i % 3 == 0:
                layer = _DenseLayer(
                    num_input_features + i * growth_rate, growth_rate, bn_size,
                    drop_rate, True
                )
            else:
                layer = _DenseLayer(
                    num_input_features + i * growth_rate, growth_rate, bn_size,
                    drop_rate, False
                )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    
    # MARK: Magic Functions
    
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(
            num_input_features, num_output_features, kernel_size=(1, 1),
            stride=(1, 1), bias=False
        ))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))
        

# MARK: - DenseNetIBN

cfgs = {
    "densenet121_ibn_a": {
        "num_init_features": 64, "growth_rate": 42,
        "block_config": (6, 12, 24, 16), "bn_size": 4, "drop_rate": 0
    },
    "densenet161_ibn_a": {
        "num_init_features": 96, "growth_rate": 48,
        "block_config": (6, 12, 36, 24), "bn_size": 4, "drop_rate": 0
    },
    "densenet169_ibn_a": {
        "num_init_features": 64, "growth_rate": 42,
        "block_config": (6, 12, 32, 32), "bn_size": 4, "drop_rate": 0
    },
    "densenet201_ibn_a": {
        "num_init_features": 64, "growth_rate": 42,
        "block_config": (6, 12, 48, 32), "bn_size": 4, "drop_rate": 0
    },
}


@MODELS.register(name="densenet_ibn")
@BACKBONES.register(name="densenet_ibn")
class DenseNet_IBN(ImageClassifier):
    """Densenet-BC model based on `Densely Connected Convolutional Networks -
    <https://arxiv.org/pdf/1608.06993.pdf>`
    
    Args:
        basename (str, optional):
            Model basename. Default: `densenet`.
        name (str, optional):
            Model name. Default: `densenet_ibn`.
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
        growth_rate       : int                  = 42,
        block_config      : ListOrTupleAnyT[int] = (6, 12, 24, 16),
        num_init_features : int                  = 64,
        bn_size           : int                  = 4,
        drop_rate         : float                = 0,
        # BaseModel's args
        basename          : Optional[str]   = "densenet",
        name              : Optional[str]   = "densenet_ibn",
        num_classes       : Optional[int]   = None,
        out_indexes       : Indexes         = -1,
        pretrained        : Pretrained      = False,
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
        self.growth_rate 	   = growth_rate
        # How many filtering to add each layer (`k` in paper).
        self.block_config	   = block_config
        # How many layers in each pooling block.
        self.num_init_features = num_init_features
        # Number of filtering to learn in the first convolution layer.
        self.bn_size    	   = bn_size
        # Multiplicative factor for number of bottle neck layers (i.e.
        # bn_size * k features in the bottleneck layer). Default: `4`.
        self.drop_rate  	   = drop_rate
        # Dropout rate after each dense layer. Default: `0`.

        # NOTE: Features
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, self.num_init_features, kernel_size=(7, 7),
                                stride=(2, 2), padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(self.num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Each denseblock
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            ibn = True
            if i >= 3:
                ibn = False

            block = _DenseBlock(
                num_layers         = num_layers,
                num_input_features = num_features,
                bn_size            = self.bn_size,
                growth_rate        = self.growth_rate,
                drop_rate          = self.drop_rate,
                ibn                = ibn
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * self.growth_rate

            if i != len(self.block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features //= 2
        
        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        
        # NOTE: Classifier
        self.classifier = self.create_classifier(num_features, self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
     
    # MARK: Configure

    @staticmethod
    def create_classifier(
        num_features: int, num_classes: Optional[int]
    ) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Linear(num_features, num_classes)
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
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=(7, 7), stride=(1, 1)).view(x.size()[0], -1)
        x = self.classifier(x)
        return x


# MARK: - DenseNet121_IBN_A

@MODELS.register(name="densenet121_ibn_a")
@BACKBONES.register(name="densenet121_ibn_a")
class DenseNet121_IBN_A(DenseNet_IBN):
    """Densenet-121-IBN-a model from `Densely Connected Convolutional Networks
    - <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://github.com/XingangPan/IBN-Net/releases/download/v1.0/densenet121_ibn_a-e4af5cc1.pth",
            file_name="densenet121_ibn_a_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "densenet121_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["densenet121_ibn_a"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
 

# MARK: - DenseNet161_IBN_A

@MODELS.register(name="densenet161_ibn_a")
@BACKBONES.register(name="densenet161_ibn_a")
class DenseNet161_IBN_A(DenseNet_IBN):
    """Densenet-161-IBN-a model from `Densely Connected Convolutional Networks
    - <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "densenet161_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["densenet161_ibn_a"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        

# MARK: - DenseNet169_IBN_A

@MODELS.register(name="densenet169_ibn_a")
@BACKBONES.register(name="densenet169_ibn_a")
class DenseNet169_IBN_A(DenseNet_IBN):
    """Densenet-169-IBN-a model from `Densely Connected Convolutional Networks
    - <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://github.com/XingangPan/IBN-Net/releases/download/v1.0/densenet169_ibn_a-9f32c161.pth",
            file_name="densenet169_ibn_a_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "densenet169_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["densenet169_ibn_a"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - DenseNet201_IBN_A

@MODELS.register(name="densenet201_ibn_a")
@BACKBONES.register(name="densenet201_ibn_a")
class DenseNet201_IBN_A(DenseNet_IBN):
    """Densenet-201-IBN-a model from `Densely Connected Convolutional Networks
    - <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    
    model_zoo = {}
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "densenet201_ibn_a",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["densenet201_ibn_a"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
