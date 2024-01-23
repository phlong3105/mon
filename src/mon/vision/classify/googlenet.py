#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements GoogleNet models."""

from __future__ import annotations

__all__ = [
    "GoogleNet",
]

from collections import namedtuple
from typing import Any

import torch

from mon import core, nn
from mon.core.typing import _callable
from mon.globals import MODELS
from mon.nn import functional as F
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Module

class BasicConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn   = nn.BatchNorm2d(out_channels, eps=0.001)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.conv(x)
        x = self.bn(x)
        y = F.relu(x, inplace=True)
        return y
    

class Inception(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        ch1x1      : int,
        ch3x3red   : int,
        ch3x3      : int,
        ch5x5red   : int,
        ch5x5      : int,
        pool_proj  : int,
        conv_block : _callable = None,
        *args, **kwargs
    ):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x       = input
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        y       = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return y
    

class InceptionAux(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block : _callable = None,
        dropout    : float     = 0.7,
        *args, **kwargs
    ):
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv    = conv_block(in_channels, 128, kernel_size=1)
        self.fc1     = nn.Linear(2048, 1024)
        self.fc2     = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        y = self.fc2(x)
        # N x 1000 (num_classes)
        return y

# endregion


# region Model

GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": torch.Tensor, "aux_logits2": torch.Tensor | None, "aux_logits1": torch.Tensor | None}


@MODELS.register(name="googlenet")
class GoogleNet(base.ImageClassificationModel):
    """GoogLeNet (Inception v1) model architecture from
    `Going Deeper with Convolutions <http://arxiv.org/abs/1409.4842>`_.
    
    See Also: :class:`mon.vision.classify.base.ImageClassificationModel`
    """
    
    constants = ["aux_logits", "transform_input"]
    
    zoo = {
        "imagenet1k-v1": {
            "url"        : "https://download.pytorch.org/models/googlenet-1378be20.pth",
            "path"       : "googlenet-imagenet1k-v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }

    def __init__(
        self,
        channels       : int                    = 3,
        num_classes    : int                    = 1000,
        aux_logits     : bool                   = True,
        transform_input: bool                   = False,
        init_weights   : bool            | None = None,
        blocks         : list[nn.Module] | None = None,
        dropout        : float                  = 0.2,
        dropout_aux    : float                  = 0.7,
        weights        : Any                    = None,
        name           : str                    = "googlenet",
        *args, **kwargs
    ):
        super().__init__(
            channels    = channels,
            num_classes = num_classes,
            weights     = weights,
            name        = name,
            *args, **kwargs
        )
        
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            console.log(
                f"The default weight initialization of GoogleNet will be "
                f"changed in future releases of ``torchvision``. If you wish "
                f"to keep the old behavior (which leads to long initialization "
                f"times due to scipy/scipy#11299), please set init_weights=True."
            )
            init_weights = True
        if len(blocks) != 3:
            raise ValueError(f":param:`blocks`'s length should be ``3``, but got {len(blocks)}.")
       
        conv_block          = blocks[0]
        inception_block     = blocks[1]
        inception_aux_block = blocks[2]
        
        # Construct model
        self.aux_logits      = aux_logits
        self.transform_input = transform_input
        self.blocks          = blocks
        self.dropout         = dropout
        self.dropout_aux     = dropout_aux
        
        self.conv1       = conv_block(self.channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1    = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2       = conv_block(64, 64, kernel_size=1)
        self.conv3       = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2    = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192,  64,  96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3    = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96,  208, 16,  48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24,  64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24,  64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32,  64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4    = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = inception_aux_block(512, self.num_classes, dropout=self.dropout_aux)
            self.aux2 = inception_aux_block(528, self.num_classes, dropout=self.dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=self.dropout)
        self.fc      = nn.Linear(1024, self.num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        elif init_weights:
            self.apply(self.init_weights)
        
    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    
    def _transform_input(self, x: torch.Tensor) -> torch.sTensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x
    
    @torch.jit.unused
    def eager_outputs(
        self,
        x   : torch.Tensor,
        aux2: torch.Tensor,
        aux1: torch.Tensor | None
    ) -> GoogLeNetOutputs:
        if self.training and self.aux_logits:
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x  # type: ignore[return-value]
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> GoogLeNetOutputs:
        x = input                         # N x 3 x 224 x 224
        x = self.conv1(x)                 # N x 64 x 112 x 112
        x = self.maxpool1(x)              # N x 64 x 56 x 56
        x = self.conv2(x)                 # N x 64 x 56 x 56
        x = self.conv3(x)                 # N x 192 x 56 x 56
        x = self.maxpool2(x)              # N x 192 x 28 x 28
        x = self.inception3a(x)           # N x 256 x 28 x 28
        x = self.inception3b(x)           # N x 480 x 28 x 28
        x = self.maxpool3(x)              # N x 480 x 14 x 14
        x = self.inception4a(x)           # N x 512 x 14 x 14
        aux1: torch.Tensor | None = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)
        x = self.inception4b(x)           # N x 512 x 14 x 14
        x = self.inception4c(x)           # N x 512 x 14 x 14
        x = self.inception4d(x)           # N x 528 x 14 x 14
        aux2: torch.Tensor | None = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)
        x = self.inception4e(x)           # N x 832 x 14 x 14
        x = self.maxpool4(x)              # N x 832 x 7 x 7
        x = self.inception5a(x)           # N x 832 x 7 x 7
        x = self.inception5b(x)           # N x 1024 x 7 x 7
        x = self.avgpool(x)               # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)           # N x 1024
        x = self.dropout(x)
        x = self.fc(x)                    # N x 1000 (num_classes) 
        
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                console.warning("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)
    
# endregion
