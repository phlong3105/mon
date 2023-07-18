#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the head blocks of different models."""

from __future__ import annotations

__all__ = [
    "AlexNetClassifier", "ConvNeXtClassifier", "GoogleNetClassifier",
    "InceptionClassifier", "LeNetClassifier", "LinearClassifier",
    "MobileOneClassifier", "ShuffleNetV2Classifier", "SqueezeNetClassifier",
    "VGGClassifier",
]

from typing import Callable

import torch
from torch import nn

from mon.coreml.layer.base import (
    activation, base, conv,
    dropout as dropout_layer, linear, mutating, pooling,
)
from mon.globals import LAYERS


# region Classification Head

@LAYERS.register()
class AlexNetClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.drop1   = dropout_layer.Dropout()
        self.linear1 = linear.Linear(
            in_features  = in_channels * 6 * 6,
            out_features = 4096
        )
        self.act1    = activation.ReLU(inplace=True)
        self.drop2   = dropout_layer.Dropout()
        self.linear2 = linear.Linear(in_features=4096, out_features=4096)
        self.act2    = activation.ReLU(inplace=True)
        self.linear3 = linear.Linear(
            in_features  = 4096,
            out_features = out_channels
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = torch.flatten(x, 1)
            y = self.drop1(y)
            y = self.act1(self.linear1(y))
            y = self.drop2(y)
            y = self.act2(self.linear2(y))
            y = self.linear3(y)
            return y
        else:
            return x


@LAYERS.register()
class ConvNeXtClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, norm: Callable = None):
        super().__init__()
        self.out_channels = out_channels
        self.norm    = norm(in_channels)
        self.flatten = mutating.Flatten(start_dim = 1)
        self.linear  = linear.Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = self.norm(x)
            y = self.flatten(y)
            y = self.linear(y)
            return y
        else:
            return x


@LAYERS.register()
class GoogleNetClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.out_channels = out_channels
        self.dropout = dropout
        self.avgpool = pooling.AdaptiveAvgPool2d((1, 1))
        self.dropout = dropout_layer.Dropout(p=dropout)
        self.fc      = linear.Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = self.avgpool(x)
            # N x 1024 x 1 x 1
            y = torch.flatten(y, 1)
            # N x 1024
            y = self.dropout(y)
            y = self.fc(y)
            # N x 1000 (num_classes)
            return y
        else:
            return x


@LAYERS.register()
class InceptionClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.avgpool = pooling.AdaptiveAvgPool2d((1, 1))
        self.dropout = dropout_layer.Dropout(p=0.5)
        self.fc      = linear.Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = x
            # Adaptive average pooling
            y = self.avgpool(y)
            # N x 2048 x 1 x 1
            y = self.dropout(y)
            # N x 2048 x 1 x 1
            y = torch.flatten(y, 1)
            # N x 2048
            y = self.fc(y)
            # N x 1000 (num_classes)
            return y
        else:
            return x


@LAYERS.register()
class LeNetClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.linear1 = linear.Linear(
            in_features  = in_channels,
            out_features = 84,
        )
        self.act1    = activation.Tanh()
        self.linear2 = linear.Linear(
            in_features  = 84,
            out_features = out_channels,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = self.linear1(x)
            y = self.act1(y)
            y = self.linear2(y)
            return y
        else:
            return x


@LAYERS.register()
class LinearClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.linear = linear.Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = torch.flatten(x, 1)
            y = self.linear(y)
            return y
        else:
            return x


@LAYERS.register()
class MobileOneClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.avgpool = pooling.AdaptiveAvgPool2d(1)
        self.fc      = linear.Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = x
            y = self.avgpool(y)
            y = y.view(y.size(0), -1)
            y = self.fc(y)
            return y
        else:
            return x


@LAYERS.register()
class ShuffleNetV2Classifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.linear = linear.Linear(
            in_features  = in_channels,
            out_features = out_channels,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = x.mean([2, 3])  # global_pool
            y = self.linear(y)
            return y
        else:
            return x


@LAYERS.register()
class SqueezeNetClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.out_channels = out_channels
        self.dropout = dropout_layer.Dropout(p=dropout)
        self.conv    = conv.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
        )
        self.act     = activation.ReLU(inplace=True)
        self.avgpool = pooling.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = self.dropout(x)
            y = self.conv(y)
            y = self.act(y)
            y = self.avgpool(y)
            y = torch.flatten(y, dims=1)
            return y
        else:
            return x


@LAYERS.register()
class VGGClassifier(base.HeadLayerParsingMixin, nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.linear1 = linear.Linear(
            in_features  = in_channels * 7 * 7,
            out_features = 4096,
        )
        self.act1    = activation.ReLU(inplace=True)
        self.drop1   = dropout_layer.Dropout()
        self.linear2 = linear.Linear(in_features=4096 , out_features=4096)
        self.act2    = activation.ReLU(inplace=True)
        self.drop2   = dropout_layer.Dropout()
        self.linear3 = linear.Linear(
            in_features  = 4096,
            out_features = out_channels,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.out_channels > 0:
            y = torch.flatten(x, 1)
            y = self.act1(self.linear1(y))
            y = self.drop1(y)
            y = self.act2(self.linear2(y))
            y = self.drop2(y)
            y = self.linear3(y)
            return y
        else:
            return x

# endregion
