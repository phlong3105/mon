#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ConNeXt models."""

from __future__ import annotations

__all__ = [
    "ConvNeXt",
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
]

import functools
from abc import ABC
from typing import Any, Sequence

import torch
from torchvision import ops

from mon import core, nn
from mon.core.typing import _callable
from mon.globals import MODELS
from mon.nn import functional as F
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Module

class LayerNorm2d(nn.LayerNorm):
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        y = x.permute(0, 3, 1, 2)
        return y


class CNBlock(nn.Module):
    
    def __init__(
        self,
        channels             : int,
        layer_scale          : float,
        stochastic_depth_prob: float,
        norm_layer           : _callable = None,
        *args, **kwargs
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = functools.partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=True),
            nn.Permute([0, 2, 3, 1]),
            norm_layer(channels),
            nn.Linear(in_features=channels, out_features=4 * channels, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * channels, out_features=channels, bias=True),
            nn.Permute([0, 3, 1, 2]),
        )
        self.layer_scale      = nn.Parameter(torch.ones(channels, 1, 1) * layer_scale)
        self.stochastic_depth = ops.StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.layer_scale * self.block(x)
        y = self.stochastic_depth(y)
        y += x
        return y

# endregion


# region Model

class CNBlockConfig:
    
    # Stores information listed at Section 3 of the ConvNeXt paper
    def __init__(
        self,
        in_channels : int,
        out_channels: int | None,
        num_layers  : int,
    ):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_layers   = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "in_channels={in_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)
    

class ConvNeXt(base.ImageClassificationModel, ABC):
    """ConNeXt.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {}
    
    def __init__(
        self,
        block_setting        : list[CNBlockConfig],
        stochastic_depth_prob: float     = 0.0,
        layer_scale          : float     = 1e-6,
        channels             : int       = 3,
        num_classes          : int       = 1000,
        block                : Any       = None,
        norm_layer           : _callable = None,
        weights              : Any       = None,
        name                 : str       = "connext",
        *args, **kwargs,
    ):
        super().__init__(
            channels    = channels,
            num_classes = num_classes,
            weights     = weights,
            name        = name,
            *args, **kwargs
        )
        
        if not block_setting:
            raise ValueError("The :param:`block_setting` should not be empty.")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The :param:`block_setting` should be :class:`list[CNBlockConfig]`.")
        if block is None:
            block = CNBlock
        if norm_layer is None:
            norm_layer = functools.partial(LayerNorm2d, eps=1e-6)
        
        self.block_setting         = block_setting
        self.stochastic_depth_prob = stochastic_depth_prob
        self.layer_scale           = layer_scale
        
        layers: list[nn.Module] = []

        # Stem
        firstconv_output_channels = self.block_setting[0].in_channels
        layers.append(
            nn.Conv2dNormAct(
                in_channels      = self.channels,
                out_channels     = firstconv_output_channels,
                kernel_size      = 4,
                stride           = 4,
                padding          = 0,
                norm_layer       = norm_layer,
                activation_layer = None,
                bias             = True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in self.block_setting)
        stage_block_id     = 0
        for cnf in self.block_setting:
            # Bottlenecks
            stage: list[nn.Module] = []
            for _ in range(cnf.num_layers):
                # Adjust stochastic depth probability based on the depth of the stage block
                sd_prob = self.stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.in_channels, self.layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.in_channels),
                        nn.Conv2d(cnf.in_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool  = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (lastblock.out_channels if lastblock.out_channels is not None else lastblock.in_channels)
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channels, self.num_classes)
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.features(x)
        x = self.avgpool(x)
        y = self.classifier(x)
        return y
    

@MODELS.register(name="convnext_base")
class ConvNeXtBase(ConvNeXt):
    """ConvNeXt Base model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
            "path"       : "convnext_base-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "convnext",
        variant: str = "convnext_base",
        *args, **kwargs
    ):
        block_setting = [
            CNBlockConfig(128,  256,  3),
            CNBlockConfig(256,  512,  3),
            CNBlockConfig(512,  1024, 27),
            CNBlockConfig(1024, None, 3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
        super().__init__(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )


@MODELS.register(name="convnext_tiny")
class ConvNeXtTiny(ConvNeXt):
    """ConvNeXt Tiny model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
            "path"       : "convnext_tiny-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }

    def __init__(
        self,
        name   : str = "convnext",
        variant: str = "convnext_tiny",
        *args, **kwargs
    ):
        block_setting = [
            CNBlockConfig(96,  192,  3),
            CNBlockConfig(192, 384,  3),
            CNBlockConfig(384, 768,  9),
            CNBlockConfig(768, None, 3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
        super().__init__(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )


@MODELS.register(name="convnext_small")
class ConvNeXtSmall(ConvNeXt):
    """ConvNeXt Small model architecture from the
    `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_ paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/convnext_small-0c510722.pth",
            "path"       : "convnext_small-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "convnext",
        variant: str = "convnext_small",
        *args, **kwargs
    ):
        block_setting = [
            CNBlockConfig(96,  192,  3),
            CNBlockConfig(192, 384,  3),
            CNBlockConfig(384, 768,  27),
            CNBlockConfig(768, None, 3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
        super().__init__(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )
        
        
@MODELS.register(name="convnext_large")
class ConvNeXtLarge(ConvNeXt):
    """ConNeXt-Large.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
            "path"       : "convnext_large-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }

    def __init__(
        self,
        name   : str = "convnext",
        variant: str = "convnext_large",
        *args, **kwargs
    ):
        block_setting = [
            CNBlockConfig(192,  384,  3),
            CNBlockConfig(384,  768,  3),
            CNBlockConfig(768,  1536, 27),
            CNBlockConfig(1536, None, 3),
        ]
        stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
        super().__init__(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            name                  = name,
            variant               = variant,
            *args, **kwargs
        )
        
# endregion
