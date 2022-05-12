#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG models.
"""

from __future__ import annotations

from typing import cast
from typing import Optional
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor

from one.core import BACKBONES
from one.core import Indexes
from one.core import MODELS
from one.core import Pretrained
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "VGG",
    "VGG11",
    "VGG11Bn",
    "VGG13",
    "VGG13Bn",
    "VGG16",
    "VGG16Bn",
    "VGG19",
    "VGG19Bn"
]


# MARK: - VGG

cfgs: dict[str, list[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512,
          "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M",
          512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512,
          512, "M", 512, 512, 512, 512, "M"],
}


@BACKBONES.register(name="vgg")
@MODELS.register(name="vgg")
class VGG(ImageClassifier):
    """VGG.
    
    Args:
        basename (str, optional):
            Model basename. Default: `vgg`.
        name (str, optional):
            Name of the backbone. Default: `vgg`.
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
        batch_norm (bool):
            Should use batch norm layer? Default: `False`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        cfg		   : list[Union[str, int]],
        batch_norm : bool          = False,
        # BaseModel's args
        basename   : Optional[str] = "vgg",
        name       : Optional[str] = "vgg",
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
        self.features = self.create_features(cfg, batch_norm)
        
        # NOTE: Head (pool + classifier)
        self.avgpool 	= nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self.create_classifier(self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.init_weights()
        
    # MARK: Configure
    
    @staticmethod
    def create_features(cfg, batch_norm) -> nn.Sequential:
        layers: list[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(
                    in_channels, v, kernel_size=(3, 3), padding=1
                )
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_classifier(num_classes: Optional[int]) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        else:
            classifier = nn.Identity()
        return classifier
    
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

# MARK: - VGG11

@MODELS.register(name="vgg11")
@BACKBONES.register(name="vgg11")
class VGG11(VGG):
    """VGG 11-layer model (configuration "A") from `Very Deep Convolutional
    Networks For Large-Scale Image Recognition -
    <https://arxiv.org/pdf/1409.1556.pdf>`. Frequired minimum input size of
    the model is 32x32.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/vgg11-8a719046.pth",
            file_name="vgg11_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "vgg11",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg		    = cfgs["A"],
            batch_norm  = False,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - VGG11Bn

@MODELS.register(name="vgg11_bn")
@BACKBONES.register(name="vgg11_bn")
class VGG11Bn(VGG):
    """VGG 11-layer model (configuration "A") from `Very Deep Convolutional
    Networks For Large-Scale Image Recognition -
    <https://arxiv.org/pdf/1409.1556.pdf>`. Frequired minimum input size of
    the model is 32x32.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            file_name="vgg11_bn_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "vgg11_bn",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg		    = cfgs["A"],
            batch_norm  = True,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        

# MARK: - VGG13

@MODELS.register(name="vgg13")
@BACKBONES.register(name="vgg13")
class VGG13(VGG):
    """VGG 13-layer model (configuration "B") `Very Deep Convolutional
    Networks For Large-Scale Image Recognition
    - <https://arxiv.org/pdf/1409.1556.pdf>`. Frequired minimum input size
    of the model is 32x32.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/vgg13-19584684.pth",
            file_name="vgg13_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "vgg13",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg		    = cfgs["B"],
            batch_norm  = False,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - VGG13Bn

@MODELS.register(name="vgg13_bn")
@BACKBONES.register(name="vgg13_bn")
class VGG13Bn(VGG):
    """VGG 13-layer model (configuration "B") `Very Deep Convolutional
    Networks For Large-Scale Image Recognition
    - <https://arxiv.org/pdf/1409.1556.pdf>`. Frequired minimum input size
    of the model is 32x32.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            file_name="vgg13_bn_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "vgg13_bn",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg		    = cfgs["B"],
            batch_norm  = True,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        

# MARK: - VGG16

@MODELS.register(name="vgg16")
@BACKBONES.register(name="vgg16")
class VGG16(VGG):
    """VGG 16-layer model (configuration "D") `Very Deep Convolutional Networks
    For Large-Scale Image Recognition - <https://arxiv.org/pdf/1409.1556.pdf>`.
    Frequired minimum input size of the model is 32x32.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/vgg16-397923af.pth",
            file_name="vgg16_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "vgg16",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg		    = cfgs["D"],
            batch_norm  = False,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - VGG16Bn

@MODELS.register(name="vgg16_bn")
@BACKBONES.register(name="vgg16_bn")
class VGG16Bn(VGG):
    """VGG 16-layer model (configuration "D") `Very Deep Convolutional Networks
    For Large-Scale Image Recognition - <https://arxiv.org/pdf/1409.1556.pdf>`.
    Frequired minimum input size of the model is 32x32.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            file_name="vgg16_bn_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "vgg16_bn",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg		    = cfgs["D"],
            batch_norm  = True,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        
        
# MARK: - VGG19

@MODELS.register(name="vgg19")
@BACKBONES.register(name="vgg19")
class VGG19(VGG):
    """VGG 19-layer model (configuration "E") `Very Deep Convolutional
    Networks For Large-Scale Image Recognition -
    <https://arxiv.org/pdf/1409.1556.pdf>`. Frequired minimum input size
    of the model is 32x32.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            file_name="vgg19_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "vgg19",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg		    = cfgs["E"],
            batch_norm  = False,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - VGG19Bn

@MODELS.register(name="vgg19_bn")
@BACKBONES.register(name="vgg19_bn")
class VGG19Bn(VGG):
    """VGG 19-layer model (configuration "E") `Very Deep Convolutional
    Networks For Large-Scale Image Recognition -
    <https://arxiv.org/pdf/1409.1556.pdf>`. Frequired minimum input size
    of the model is 32x32.
    """

    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
            file_name="vgg19_bn_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "vgg19_bn",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            cfg		    = cfgs["E"],
            batch_norm  = True,
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
