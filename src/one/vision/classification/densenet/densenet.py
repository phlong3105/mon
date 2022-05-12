#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DenseNet models.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.densenet import _DenseBlock
from torchvision.models.densenet import _Transition

from one.core import BACKBONES
from one.core import Indexes
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "DenseNet",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201"
]


# MARK: - DenseNet

cfgs = {
    "densenet121": {
        "growth_rate": 32, "block_config": (6, 12, 24, 16),
        "num_init_features": 64, "bn_size": 4, "drop_rate": 0,
        "memory_efficient": False
    },
    "densenet161": {
        "growth_rate": 48, "block_config": (6, 12, 36, 24),
        "num_init_features": 96, "bn_size": 4, "drop_rate": 0,
        "memory_efficient": False
    },
    "densenet169": {
        "growth_rate": 32, "block_config": (6, 12, 32, 32),
        "num_init_features": 64, "bn_size": 4, "drop_rate": 0,
        "memory_efficient": False
    },
    "densenet201": {
        "growth_rate": 32, "block_config": (6, 12, 48, 32),
        "num_init_features": 64, "bn_size": 4, "drop_rate": 0,
        "memory_efficient": False
    },
}


@MODELS.register(name="densenet")
@BACKBONES.register(name="densenet")
class DenseNet(ImageClassifier):
    """DenseNet backbone.
    
    Args:
        basename (str, optional):
            Model basename. Default: `densenet`.
        name (str, optional):
            Model name. Default: `densenet`.
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
        growth_rate       : int                  = 32,
        block_config      : ListOrTupleAnyT[int] = (6, 12, 24, 16),
        num_init_features : int                  = 64,
        bn_size           : int                  = 4,
        drop_rate         : float                = 0,
        memory_efficient  : bool                 = False,
        # BaseModel's args
        basename          : Optional[str] = "densenet",
        name              : Optional[str] = "densenet",
        num_classes       : Optional[int] = None,
        out_indexes       : Indexes       = -1,
        pretrained        : Pretrained    = False,
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
        self.memory_efficient  = memory_efficient
        # If True, uses checkpointing. Much more memory efficient, but slower.
        # Default: `False`. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`.
        
        # NOTE: Features
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            (
                "conv0", nn.Conv2d(
                    3, self.num_init_features, kernel_size=(7, 7),
                    stride=(2, 2), padding=3, bias=False
                )
             ),
            ("norm0", nn.BatchNorm2d(self.num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = _DenseBlock(
                num_layers         = num_layers,
                num_input_features = num_features,
                bn_size            = self.bn_size,
                growth_rate        = self.growth_rate,
                drop_rate          = self.drop_rate,
                memory_efficient   = self.memory_efficient
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * self.growth_rate
            if i != len(self.block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        
        # NOTE: Classifier
        self.classifier = self.create_classifier(num_features, self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.init_weights()

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
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
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
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# MARK: - DenseNet121

@MODELS.register(name="densenet121")
@BACKBONES.register(name="densenet121")
class DenseNet121(DenseNet):
    """Densenet-121 model from `Densely Connected Convolutional Networks -
    <https://arxiv.org/pdf/1608.06993.pdf>`. Frequired minimum input size of
    the model is 29x29.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/densenet121-a639ec97.pth",
            file_name="densenet121_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "densenet121",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["densenet121"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
    

# MARK: - DenseNet161

@MODELS.register(name="densenet161")
@BACKBONES.register(name="densenet161")
class DenseNet161(DenseNet):
    """Densenet-161 model from `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`. Frequired minimum input size of
    the model is 29x29.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/densenet161-8d451a50.pth",
            file_name="densenet161_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "densenet161",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["densenet161"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        

# MARK: - DenseNet169

@MODELS.register(name="densenet169")
@BACKBONES.register(name="densenet169")
class DenseNet169(DenseNet):
    """Densenet-169 model from `Densely Connected Convolutional Networks -
    <https://arxiv.org/pdf/1608.06993.pdf>`. Frequired minimum input size of
    the model is 29x29.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            file_name="densenet169_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "densenet169",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["densenet169"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - DenseNet201

@MODELS.register(name="densenet201")
@BACKBONES.register(name="densenet201")
class DenseNet201(DenseNet):
    """Densenet-201 model from `Densely Connected Convolutional Networks
    <https://arxiv.org/pdf/1608.06993.pdf>`. Frequired minimum input size of
    the model is 29x29.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/densenet201-c1103571.pth",
            file_name="densenet201_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "densenet201",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["densenet201"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
