#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ShuffleNetV2 models.
"""

from __future__ import annotations

from typing import Optional

from torch import nn
from torch import Tensor
from torchvision.models.shufflenetv2 import InvertedResidual

from one.core import BACKBONES
from one.core import Indexes
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "ShuffleNetV2",
    "ShuffleNetV2_x0_5",
    "ShuffleNetV2_x1_0",
    "ShuffleNetV2_x1_5",
    "ShuffleNetV2_x2_0"
]


# MARK: - ShuffleNetV2

cfgs = {
    "shufflenet_v2_x0_5": {
        "stages_repeats": [4, 8, 4],
        "stages_out_channels": [24, 48, 96, 192, 1024],
        "inverted_residual": InvertedResidual
    },
    "shufflenet_v2_x1_0": {
        "stages_repeats": [4, 8, 4],
        "stages_out_channels": [24, 116, 232, 464, 1024],
        "inverted_residual": InvertedResidual
    },
    "shufflenet_v2_x1_5": {
        "stages_repeats": [4, 8, 4],
        "stages_out_channels": [24, 176, 352, 704, 1024],
        "inverted_residual": InvertedResidual
    },
    "shufflenet_v2_x2_0": {
        "stages_repeats": [4, 8, 4],
        "stages_out_channels": [24, 244, 488, 976, 2048],
        "inverted_residual": InvertedResidual
    },
}


@MODELS.register(name="shufflenet_v2")
@BACKBONES.register(name="shufflenet_v2")
class ShuffleNetV2(ImageClassifier):
    """ShuffleNetV2.
    
    Args:
        basename (str, optional):
            Model basename. Default: `shufflenet_v2`.
        name (str, optional):
            Name of the backbone. Default: `shufflenet_v2`.
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
        stages_repeats     : ListOrTupleAnyT[int] = (4, 8, 4),
        stages_out_channels: ListOrTupleAnyT[int] = (24, 244, 488, 976, 2048),
        inverted_residual  : nn.Module            = InvertedResidual,
        # BaseModel's args
        basename   : Optional[str] = "shufflenet_v2",
        name       : Optional[str] = "shufflenet_v2",
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
        if len(stages_repeats) != 3:
            raise ValueError("Expected stages_repeats as list of 3 positive "
                             "ints")
        if len(stages_out_channels) != 5:
            raise ValueError("Expected stages_out_channels as list of 5 "
                             "positive ints")
        self._stage_out_channels = stages_out_channels
        
        input_channels  = 3
        output_channels = self._stage_out_channels[0]
        
        # NOTE: Features
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, (3, 3), (2, 2), 1,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, stages_repeats, self._stage_out_channels[1:]
        ):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(
                    output_channels, output_channels, 1
                ))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        
        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, (1, 1), (1, 1), 0,
                      bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        # NOTE: Head (Pool + Classifier layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = self.create_classifier(output_channels, self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
    
        # NOTE: Alias
        self.features = nn.Sequential(
            self.conv1, self.maxpool, self.stage2, self.stage3, self.stage4,
            self.conv5
        )
        self.classifier = self.fc
    
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
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x


# MARK: - ShuffleNetV2_x0_5

@MODELS.register(name="shufflenet_v2_x0_5")
@BACKBONES.register(name="shufflenet_v2_x0_5")
class ShuffleNetV2_x0_5(ShuffleNetV2):
    """ShuffleNetV2 with 0.5x output channels, as described in `ShuffleNet V2:
    Practical Guidelines for Efficient CNN Architecture Design -
    <https://arxiv.org/abs/1807.11164>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
            file_name="shufflenet_v2_x0_5_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "shufflenet_v2_x0_5",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["shufflenet_v2_x0_5"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ShuffleNetV2_x1_0

@MODELS.register(name="ShuffleNetV2_x1_0")
@BACKBONES.register(name="shufflenet_v2_x1_0")
class ShuffleNetV2_x1_0(ShuffleNetV2):
    """ShuffleNetV2 with 1.0x output channels, as described in `ShuffleNet V2:
    Practical Guidelines for Efficient CNN Architecture Design -
    <https://arxiv.org/abs/1807.11164>`.
    """
    
    model_zoo = {
        "imagenet": dict(
            path="https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
            file_name="shufflenet_v2_x1_0_imagenet.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "shufflenet_v2_x1_0",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["shufflenet_v2_x1_0"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        

# MARK: - ShuffleNetV2_x1_5

@MODELS.register(name="shufflenet_v2_x1_5")
@BACKBONES.register(name="shufflenet_v2_x1_5")
class ShuffleNetV2_x1_5(ShuffleNetV2):
    """ShuffleNetV2 with 1.5x output channels, as described in `ShuffleNet V2:
    Practical Guidelines for Efficient CNN Architecture Design -
    <https://arxiv.org/abs/1807.11164>`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "shufflenet_v2_x1_5",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["shufflenet_v2_x1_5"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ShuffleNetV2_x2_0

@MODELS.register(name="shufflenet_v2_x2_0")
@BACKBONES.register(name="shufflenet_v2_x2_0")
class ShuffleNetV2_x2_0(ShuffleNetV2):
    """ShuffleNetV2 with 2.0x output channels, as described in `ShuffleNet V2:
    Practical Guidelines for Efficient CNN Architecture Design -
    <https://arxiv.org/abs/1807.11164>`.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "shufflenet_v2_x2_0",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["shufflenet_v2_x2_0"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )
