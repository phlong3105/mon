#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements DenseNet models."""

from __future__ import annotations

__all__ = [
    "DenseNet",
    "DenseNet121",
    "DenseNet161",
    "DenseNet169",
    "DenseNet201",
]

from abc import ABC
from collections import OrderedDict
from typing import Any

import torch
import torch.utils.checkpoint as cp

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS
from mon.nn import functional as F
from mon.vision.classify import base

console = core.console


# region Module

class DenseLayer(nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        growth_rate     : int,
        bn_size         : int,
        drop_rate       : float,
        memory_efficient: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate        = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, input: list[torch.Tensor]) -> torch.Tensor:
        concatenated_features = torch.cat(input, 1)
        bottleneck_output     = self.conv1(self.relu1(self.norm1(concatenated_features)))  # noqa: T484
        return bottleneck_output
    
    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: list[torch.Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: list[torch.Tensor]) -> torch.Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: list[torch.Tensor]) -> torch.Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: F811
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class DenseBlock(nn.ModuleDict):
    
    version = 2

    def __init__(
        self,
        num_layers      : int,
        in_channels     : int,
        bn_size         : int,
        growth_rate     : int,
        drop_rate       : float,
        memory_efficient: bool = False,
        *args, **kwargs
    ):
        super().__init__()
        for i in range(num_layers):
            layer = DenseLayer(
                in_channels      = in_channels + i * growth_rate,
                growth_rate      = growth_rate,
                bn_size          = bn_size,
                drop_rate        = drop_rate,
                memory_efficient = memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = [input]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class Transition(nn.Sequential):
    
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

# endregion


# region Model

class DenseNet(base.ImageClassificationModel, ABC):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    
    Args:
        growth_rate: How many filters to add each layer (:param:`k` in paper).
        block_config: A list of 4 ints determining how many layers in each
            pooling block.
        num_init_features: The number of filters to learn in the first
            convolution layer.
        bn_size: A multiplicative factor for the number of bottleneck layers.
            (i.e., :math:`bn_size * k features` in the bottleneck layer).
        drop_rate: Dropout rate after each dense layer
        num_classes: Number of classification classes.
        memory_efficient: If ``True``, uses checkpointing. Much more memory
            efficient, but slower. Default: ``False``.
            See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
          
    See Also: :class:`base.ImageClassificationModel`
    """
    
    _zoo: dict = {}
    
    def __init__(
        self,
        growth_rate      : int   = 32,
        block_config     : tuple[int, int, int, int] = (6, 12, 24, 16),
        channels         : int   = 3,
        num_channels     : int   = 64,
        bn_size          : int   = 4,
        drop_rate        : float = 0,
        num_classes      : int   = 1000,
        memory_efficient : bool  = False,
        weights          : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            channels    = channels,
            num_classes = num_classes,
            weights     = weights,
            *args, **kwargs
        )
        self.num_channels     = num_channels
        self.growth_rate      = growth_rate
        self.block_config     = block_config
        self.bn_size          = bn_size
        self.drop_rate        = drop_rate
        self.memory_efficient = memory_efficient
        
        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(self.channels, self.num_channels, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(self.num_channels)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = self.num_channels
        for i, num_layers in enumerate(self.block_config):
            block = DenseBlock(
                num_layers       = num_layers,
                in_channels      = num_features,
                bn_size          = self.bn_size,
                growth_rate      = self.growth_rate,
                drop_rate        = self.drop_rate,
                memory_efficient = self.memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * self.growth_rate
            if i != len(self.block_config) - 1:
                trans = Transition(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, self.num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.constant_(m.bias, 0)
        '''
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                torch.nn.init.kaiming_normal_(m.conv.weight)
            else:
                torch.nn.init.kaiming_normal_(m.weight)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif classname.find("Linear") != -1:
            torch.nn.init.constant_(m.bias, 0)
        '''
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        y = self.classifier(x)
        return y
    

@MODELS.register(name="densenet121")
class DenseNet121(DenseNet):
    """Densenet-121 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.
    
    See Also: :class:`DenseNet`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet121-a639ec97.pth",
            "path"       : "densenet121_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name         = "densenet121",
            growth_rate  = 32,
            block_config = (6, 12, 24, 16),
            num_channels = 64,
            *args, **kwargs
        )


@MODELS.register(name="densenet161")
class DenseNet161(DenseNet):
    """Densenet-161 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.
    
    See Also: :class:`DenseNet`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet161-8d451a50.pth",
            "path"       : "densenet161_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name         = "densenet161",
            growth_rate  = 48,
            block_config = (6, 12, 36, 24),
            num_channels = 96,
            *args, **kwargs
        )


@MODELS.register(name="densenet169")
class DenseNet169(DenseNet):
    """Densenet-169 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.
    
    See Also: :class:`DenseNet`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
            "path"       : "densenet169_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name         = "densenet169",
            growth_rate  = 32,
            block_config = (6, 12, 32, 32),
            num_channels = 64,
            *args, **kwargs
        )


@MODELS.register(name="densenet201")
class DenseNet201(DenseNet):
    """Densenet-201 model from
    `Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_.
    
    See Also: :class:`DenseNet`
    """
    
    _zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/densenet201-c1103571.pth",
            "path"       : "densenet201_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name         = "densenet201",
            growth_rate  = 32,
            block_config = (6, 12, 48, 32),
            num_channels = 64,
            *args, **kwargs
        )
 
# endregion
