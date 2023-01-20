#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements layers and blocks especially used for ShuffleNet
models.
"""

from __future__ import annotations

__all__ = [
    "InvertedResidual",
]

import torch
from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base, common


@constant.LAYER.register()
class InvertedResidual(base.LayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : int,
        *args, **kwargs
    ):
        super().__init__()

        if not (1 <= stride <= 3):
            raise ValueError("Illegal stride value.")
        self.stride = stride

        branch_features = out_channels // 2
        if (self.stride == 1) and (in_channels != branch_features << 1):
            raise ValueError(
                f"Invalid combination of `stride` {stride}, "
                f"`in_channels` {in_channels} and `out_channels` {out_channels} "
                f"values. If stride == 1 then `in_channels` should be equal "
                f"to `out_channels` // 2 << 1."
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                common.Conv2d(
                    in_channels  = in_channels,
                    out_channels = in_channels,
                    kernel_size  = 3,
                    stride       = self.stride,
                    padding      = 1,
                    groups       = in_channels,
                    bias         = False,
                ),
                common.BatchNorm2d(in_channels),
                common.Conv2d(
                    in_channels  = in_channels,
                    out_channels = branch_features,
                    kernel_size  = 1,
                    stride       = 1,
                    padding      = 0,
                    bias         = False
                ),
                common.BatchNorm2d(branch_features),
                common.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            common.Conv2d(
                in_channels  = in_channels if (self.stride > 1) else branch_features,
                out_channels = branch_features,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = False,
            ),
            common.BatchNorm2d(branch_features),
            common.ReLU(inplace=True),
            common.Conv2d(
                in_channels  = branch_features,
                out_channels = branch_features,
                kernel_size  = 3,
                stride       = self.stride,
                padding      = 1,
                groups       = branch_features,
                bias         = False,
            ),
            common.BatchNorm2d(branch_features),
            common.Conv2d(
                in_channels  = branch_features,
                out_channels = branch_features,
                kernel_size  = 1,
                stride       = 1,
                padding      = 0,
                bias         = False,
            ),
            common.BatchNorm2d(branch_features),
            common.ReLU(inplace=True),
        )
    
    @staticmethod
    def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
        b, c, h, w         = x.size()
        channels_per_group = c // groups
        # reshape
        x = x.view(b, groups, channels_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(b, -1, h, w)
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            y      = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            y = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        y = self.channel_shuffle(y, 2)
        return y

    @classmethod
    def parse_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        if isinstance(f, list | tuple):
            c1, c2 = ch[f[0]], args[0]
        else:
            c1, c2 = ch[f],    args[0]
        args = [c1, *args[0:]]
        ch.append(c2)
        return args, ch
