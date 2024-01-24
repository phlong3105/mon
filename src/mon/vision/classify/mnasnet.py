#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements MNASNet models."""

from __future__ import annotations

__all__ = [
    "MNASNet",
    "MNASNet0_5",
    "MNASNet0_75",
    "MNASNet1_0",
    "MNASNet1_3",
]

from abc import ABC
from typing import Any

import torch

from mon import core, nn
from mon.globals import MODELS
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Module

class _InvertedResidual(nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        kernel_size     : int,
        stride          : int,
        expansion_factor: int,
        bn_momentum     : float = 0.1,
        *args, **kwargs
    ):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f":param;`stride` should be ``1`` or ``2``, but got {stride}.")
        if kernel_size not in [3, 5]:
            raise ValueError(f":param`kernel_size` should be ``3`` or ``5``, but got {kernel_size}.")
        mid_channels        = in_channels * expansion_factor
        self.apply_residual = in_channels == out_channels and stride == 1
        self.layers         = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_channels, mid_channels, kernel_size, padding=kernel_size // 2, stride=stride, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.apply_residual:
            return self.layers(x) + x
        else:
            return self.layers(x)


def _stack(
    in_channels : int,
    out_channels: int,
    kernel_size : int,
    stride      : int,
    exp_factor  : int,
    repeats     : int,
    bn_momentum : float,
    *args, **kwargs,
) -> nn.Sequential:
    """Creates a stack of inverted residuals."""
    if repeats < 1:
        raise ValueError(f":param:`repeats` should be >= ``1``, but got {repeats}.")
    # First one has no skip, because feature map size changes.
    first     = _InvertedResidual(
        in_channels      = in_channels,
        out_channels     = out_channels,
        kernel_size      = kernel_size,
        stride           = stride,
        expansion_factor = exp_factor,
        bn_momentum      = bn_momentum,
        *args, **kwargs,
    )
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(
                in_channels      = out_channels,
                out_channels     = out_channels,
                kernel_size      = kernel_size,
                stride           = 1,
                expansion_factor = exp_factor,
                bn_momentum      = bn_momentum,
                *args, **kwargs,
            )
        )
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val: float, divisor: int, round_up_bias: float = 0.9) -> int:
    """Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e., (83, 8) → 80, but (84, 8) → 88.
    """
    if not 0.0 < round_up_bias < 1.0:
        raise ValueError(
            f":param:`round_up_bias` should be greater than ``0.0`` and smaller "
            f"than ``1.0``, but got {round_up_bias}"
        )
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha: float) -> list[int]:
    """Scales tensor depths as in reference MobileNet code, prefers rounding up
    rather than down.
    """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]

# endregion


# region Model

class MNASNet(base.ImageClassificationModel, ABC):
    """MNASNet.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2
    
    zoo = {}
    
    def __init__(
        self,
        alpha      : float,
        channels   : int   = 3,
        num_classes: int   = 1000,
        dropout    : float = 0.2,
        weights    : Any   = None,
        name       : str   = "mnasnet",
        *args, **kwargs,
    ):
        super().__init__(
            channels    = channels,
            num_classes = num_classes,
            weights     = weights,
            name        = name,
            *args, **kwargs
        )
        if alpha <= 0.0:
            raise ValueError(f":param:`alpha` should be greater than ``0.0``, but got {alpha}.")
       
        self.alpha   = alpha
        self.dropout = dropout
        
        # Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
        # 1.0 - tensorflow.
        bn_momentum = 1 - 0.9997
        depths      = _get_depths(self.alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(self.channels, depths[0], 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(depths[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False),
            nn.BatchNorm2d(depths[0], momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(depths[1], momentum=bn_momentum),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, bn_momentum),
            _stack(depths[2], depths[3], 5, 2, 3, 3, bn_momentum),
            _stack(depths[3], depths[4], 5, 2, 6, 3, bn_momentum),
            _stack(depths[4], depths[5], 3, 1, 6, 2, bn_momentum),
            _stack(depths[5], depths[6], 5, 2, 6, 4, bn_momentum),
            _stack(depths[6], depths[7], 3, 1, 6, 1, bn_momentum),
            # Final mapping to classifier input.
            nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=bn_momentum),
            nn.ReLU(inplace=True),
        ]
        self.layers     = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(p=self.dropout, inplace=True), nn.Linear(1280, self.num_classes))
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
            torch.nn.init.zeros_(m.bias)
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        x = input
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        y = self.classifier(x)
        return y
    

@MODELS.register(name="mnasnet0_5")
class MNASNet0_5(MNASNet):
    """MNASNet with depth multiplier of 0.5 from `MnasNet: Platform-Aware
    Neural Architecture Search for Mobile <https://arxiv.org/abs/1807.11626>`_
    paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
            "path"       : "mnasnet0_5-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "mnasnet",
        variant: str = "mnasnet0_5",
        *args, **kwargs
    ):
        super().__init__(
            alpha   = 0.5,
            name    = name,
            variant = variant,
            *args, **kwargs
        )


@MODELS.register(name="mnasnet0_75")
class MNASNet0_75(MNASNet):
    """MNASNet with depth multiplier of 0.75 from `MnasNet: Platform-Aware
    Neural Architecture Search for Mobile <https://arxiv.org/abs/1807.11626>`_
    paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet0_75-7090bc5f.pth",
            "path"       : "mnasnet0_75-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "mnasnet",
        variant: str = "mnasnet0_75",
        *args, **kwargs
    ):
        super().__init__(
            alpha   = 0.75,
            name    = name,
            variant = variant,
            *args, **kwargs
        )
        
        
@MODELS.register(name="mnasnet1_0")
class MNASNet1_0(MNASNet):
    """MNASNet with depth multiplier of 1.0 from `MnasNet: Platform-Aware
    Neural Architecture Search for Mobile <https://arxiv.org/abs/1807.11626>`_
    paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
            "path"       : "mnasnet1_0-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "mnasnet",
        variant: str = "mnasnet1_0",
        *args, **kwargs
    ):
        super().__init__(
            alpha   = 1.0,
            name    = name,
            variant = variant,
            *args, **kwargs
        )
        
        
@MODELS.register(name="mnasnet1_3")
class MNASNet1_3(MNASNet):
    """MNASNet with depth multiplier of 1.3 from `MnasNet: Platform-Aware
    Neural Architecture Search for Mobile <https://arxiv.org/abs/1807.11626>`_
    paper.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/mnasnet1_3-a4c69d6f.pth",
            "path"       : "mnasnet1_3-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(
        self,
        name   : str = "mnasnet",
        variant: str = "mnasnet1_3",
        *args, **kwargs
    ):
        super().__init__(
            alpha   = 1.3,
            name    = name,
            variant = variant,
            *args, **kwargs
        )
        
# endregion
