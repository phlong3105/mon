#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNet models.
"""

from __future__ import annotations

from typing import Optional
from typing import Type
from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import conv1x1

from one.core import BACKBONES
from one.core import Indexes
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.vision.image_classification.image_classifier import ImageClassifier

__all__ = [
    "ResNet",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "ResNeXt50_32X4D",
    "ResNeXt101_32X8D",
    "WideResNet50_2",
    "WideResNet101_2",
]


# MARK: - ResNet

cfgs = {
    "resnet18": {
        "block": BasicBlock, "layers": [2, 2, 2, 2],
        "zero_init_residual": False, "groups": 1, "width_per_group": 64,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
    "resnet34": {
        "block": BasicBlock, "layers": [3, 4, 6, 3],
        "zero_init_residual": False, "groups": 1, "width_per_group": 64,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
    "resnet50": {
        "block": Bottleneck, "layers": [3, 4, 6, 3],
        "zero_init_residual": False, "groups": 1, "width_per_group": 64,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
    "resnet101": {
        "block": Bottleneck, "layers": [3, 4, 23, 3],
        "zero_init_residual": False, "groups": 1, "width_per_group": 64,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
    "resnet152": {
        "block": Bottleneck, "layers": [3, 8, 36, 3],
        "zero_init_residual": False, "groups": 1, "width_per_group": 64,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
    "resnext50_32x4d": {
        "block": Bottleneck, "layers": [3, 4, 6, 3],
        "zero_init_residual": False, "groups": 32, "width_per_group": 4,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
    "resnext101_32x8d": {
        "block": Bottleneck, "layers": [3, 4, 23, 3],
        "zero_init_residual": False, "groups": 32, "width_per_group": 8,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
    "wide_resnet50_2": {
        "block": Bottleneck, "layers": [3, 4, 6, 3],
        "zero_init_residual": False, "groups": 1, "width_per_group": 64 * 2,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
    "wide_resnet101_2": {
        "block": Bottleneck, "layers": [3, 4, 23, 3],
        "zero_init_residual": False, "groups": 1, "width_per_group": 64 * 2,
        "replace_stride_with_dilation": None, "norm_layer": None
    },
}


@MODELS.register(name="resnet")
@BACKBONES.register(name="resnet")
class ResNet(ImageClassifier):
    """ResNet backbone.
    
    Args:
        basename (str, optional):
            Model basename. Default: `resnet`.
        name (str, optional):
            Name of the backbone. Default: `resnet`.
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
        block                                              = Bottleneck,
        layers                      : ListOrTupleAnyT[int] = (3, 4, 23, 3),
        zero_init_residual          : bool                 = False,
        groups                      : int                  = 1,
        width_per_group             : int                  = 64 * 2,
        replace_stride_with_dilation: Optional[bool]       = None,
        norm_layer                  : Optional[nn.Module]  = None,
        # BaseModel's args
        basename   : Optional[str] = "resnet",
        name       : Optional[str] = "resnet",
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
        self.block                        = block
        self.layers                       = layers
        self.zero_init_residual           = zero_init_residual
        self.groups                       = groups
        self.width_per_group              = width_per_group
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.norm_layer                   = norm_layer
        
        if self.norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        self._norm_layer = self.norm_layer
        self.inplanes 	 = 64
        self.dilation 	 = 1
        if self.replace_stride_with_dilation is None:
            # Each element in the tuple indicates if we should replace the 2x2
            # stride with a dilated convolution instead
            self.replace_stride_with_dilation = [False, False, False]
        if len(self.replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element "
                "tuple, got {}".format(self.replace_stride_with_dilation)
            )
        self.groups 	= self.groups
        self.base_width = self.width_per_group
        
        # NOTE: Features
        self.conv1   = nn.Conv2d(3, self.inplanes, (7, 7), (2, 2), padding=3,
                                 bias=False)
        self.bn1     = self.norm_layer(self.inplanes)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1  = self.make_layer(self.block, 64, self.layers[0])
        self.layer2  = self.make_layer(self.block, 128, self.layers[1], stride=2,
                                       dilate=self.replace_stride_with_dilation[0])
        self.layer3  = self.make_layer(self.block, 256, self.layers[2], stride=2,
                                       dilate=self.replace_stride_with_dilation[1])
        self.layer4  = self.make_layer(self.block, 512, self.layers[3], stride=2,
                                       dilate=self.replace_stride_with_dilation[2])
        
        # NOTE: Head (Pool + Classifier layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc		 = self.create_classifier(self.block, self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.init_weights(zero_init_residual=self.zero_init_residual)
            
        # NOTE: Alias
        self.features = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, self.layer1,
            self.layer2, self.layer3, self.layer4,
        )
        self.classifier = self.fc
    
    # MARK: Configure
    
    def make_layer(
        self,
        block : Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int  = 1,
        dilate: bool = False
    ) -> nn.Sequential:
        norm_layer 		  = self._norm_layer
        downsample 		  = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer
            ))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def create_classifier(block, num_classes: Optional[int]) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Linear( 512 * block.expansion, num_classes)
        else:
            classifier = nn.Identity()
        return classifier
    
    def init_weights(self, zero_init_residual: bool = False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch, so that the
        # residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
    
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
        x = self.fc(x)
        return x
    
    
# MARK: - ResNet18

@MODELS.register(name="resnet18")
@BACKBONES.register(name="resnet18")
class ResNet18(ResNet):
    """ResNet-18 model from `Deep Residual Learning for Image Recognition -
    <https://arxiv.org/pdf/1512.03385.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnet18-f37072fd.pth",
            file_name="resnet18_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnet18",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnet18"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNet34

@MODELS.register(name="resnet34")
@BACKBONES.register(name="resnet34")
class ResNet34(ResNet):
    """ResNet-34 model from `Deep Residual Learning for Image Recognition -
    <https://arxiv.org/pdf/1512.03385.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnet34-b627a593.pth",
            file_name="resnet34_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str]    	  	= "resnet34",
        out_indexes: Indexes		  	    = -1,
        num_classes: Optional[int] 	  	    = None,
        pretrained : Union[bool, str, dict] = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnet34"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        

# MARK: - ResNet50

@MODELS.register(name="resnet50")
@BACKBONES.register(name="resnet50")
class ResNet50(ResNet):
    """ResNet-50 model from `Deep Residual Learning for Image Recognition -
    <https://arxiv.org/pdf/1512.03385.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnet50-0676ba61.pth",
            file_name="resnet50_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnet50",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnet50"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNet101

@MODELS.register(name="resnet101")
@BACKBONES.register(name="resnet101")
class ResNet101(ResNet):
    """ResNet-101 model from `Deep Residual Learning for Image Recognition -
    <https://arxiv.org/pdf/1512.03385.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnet101-63fe2227.pth",
            file_name="resnet101_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnet101",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnet101"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNet152

@MODELS.register(name="resnet152")
@BACKBONES.register(name="resnet152")
class ResNet152(ResNet):
    """ResNet-152 model from `Deep Residual Learning for Image Recognition -
    <https://arxiv.org/pdf/1512.03385.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnet152-394f9c45.pth",
            file_name="resnet152_imagenet.pth", num_classes=1000,
        ),
    }

    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnet152",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnet152"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        

# MARK: - ResNeXt50_32X4D

@MODELS.register(name="resnext50_32x4d")
@BACKBONES.register(name="resnext50_32x4d")
class ResNeXt50_32X4D(ResNet):
    """ResNeXt-50 32x4d model from `Aggregated Residual Transformation for
    Deep Neural Networks - <https://arxiv.org/pdf/1611.05431.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
            file_name="resnext50_32x4d_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnext50_32x4d",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnext50_32x4d"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ResNeXt101_32X8D

@MODELS.register(name="resnext101_32x8d")
@BACKBONES.register(name="resnext101_32x8d")
class ResNeXt101_32X8D(ResNet):
    """ResNeXt-101 32x8d model from `Aggregated Residual Transformation for
    Deep Neural Networks" - <https://arxiv.org/pdf/1611.05431.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
            file_name="resnext101_32x8d_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "resnext101_32x8d",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["resnext101_32x8d"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        
        
# MARK: - Wide_ResNet50_2

@MODELS.register(name="wide_resnet50_2")
@BACKBONES.register(name="wide_resnet50_2")
class WideResNet50_2(ResNet):
    """Wide ResNet-50-2 model from `Wide Residual Networks -
    <https://arxiv.org/pdf/1605.07146.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
            file_name="wide_resnet50_2_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "wide_resnet50_2",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["wide_resnet50_2"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - WideResNet101_2

@MODELS.register(name="wide_resnet101_2")
@BACKBONES.register(name="wide_resnet101_2")
class WideResNet101_2(ResNet):
    """Wide ResNet-50-2 model from `Wide Residual Networks -
    <https://arxiv.org/pdf/1605.07146.pdf>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
            file_name="wide_resnet101_2_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "wide_resnet101_2",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["wide_resnet101_2"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
