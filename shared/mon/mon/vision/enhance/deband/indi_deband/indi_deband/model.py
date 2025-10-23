#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements InDi-Deband model for image/video debanding

References:
    - Paper:
    - Code: https://github.com/ksasso1028/indi-debanding
"""

__all__ = [
    "InDiDeband",
]

from typing import Any

import box
import torch
import torch.nn as nn

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .module import (
    AttnBlock,
    EncoderFFTime2d,
    PaddedConv2d,
    Snake,
    UpscaleFFTime2d,
)

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


@MODELS.register(name="indi_deband", arch="indi_deband")
class InDiDeband(nn.Module, ModelMixin):
    """Implements InDi-Deband model for image/video debanding.
    
    A 2D Unet that takes in a timestep T in encoder and decoder blocks.
    Uses sin activations, denoising target is done in the fourier domain instead of raw signal.

    References:
        - Paper:
        - Code: https://github.com/ksasso1028/indi-debanding
    """
    
    arch     : str          = "indi_deband"
    name     : str          = "indi_deband"
    tasks    : list[Task]   = [Task.DEBAND]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        blocks          : int  = 4,
        in_channels     : int  = 24,
        kernel_size     : int  = 3,
        layout          : int  = 3,     
        channel_factor  : int  = 48,
        scale_factor    : int  = 2,
        encoder_dilation: int  = 4,
        decoder_dilation: int  = 1,
        neck            : bool = False,
        weights         : Any  = None
    ):
        super().__init__()
        
        self.neck             = neck
        self.layout           = layout
        self.channel_factor   = channel_factor
        self.scale_factor     = scale_factor
        self.encoder_dilation = encoder_dilation
        self.decoder_dilation = decoder_dilation
        
        self.linear  = nn.Linear(1, in_channels)
        self.conv    = PaddedConv2d(self.layout, in_channels, kernel_size=3, stride=1)
        self.act     = nn.SiLU()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        start = in_channels
        for block in range(blocks):
            stride = self.scale_factor
            self.encoder.append(
                EncoderFFTime2d(
                    channels    = start,
                    kernel_size = kernel_size,
                    stride      = stride,
                    dilation    = 1,
                    mul         = self.channel_factor,
                    weight      = True,
                    linear      = in_channels,
                )
            )
            # Set channel size for next block
            start = start + self.channel_factor
        
        # Condition to add bottleneck
        self.linearAct = Snake(start)
        if self.neck:
            self.bottleneck = AttnBlock(start)

        for block in range(blocks):
            stride = 1
            self.decoder.append(
                UpscaleFFTime2d(
                    channels     = start,
                    kernel_size  = kernel_size,
                    stride       = stride,
                    dilation     = 1,
                    scale_factor = self.scale_factor,
                    mul          = self.channel_factor,
                    weight       = True,
                    linear       = in_channels,
                )
            )
            # Set channel size for next block
            start = start - self.channel_factor
        
        self.process = nn.Conv2d(start, self.layout, kernel_size=1, stride=1, padding=0)
        
        # Load weights
        self.load_weights(weights)
    
    def forward(self, mix: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t  = (self.linear(t.unsqueeze(-1).type(torch.float).to(mix.device)))
        x  = mix.clone()
        original = x[:, :self.layout, :].clone()
        x  = self.act(self.conv(x))
        og = x.clone()
        features = [x]
        for module in self.encoder:
            x = module(x, t)
            features.append(x)
        clone = x.clone()
        if self.neck:
            x = self.bottleneck(x.clone())
            x = clone + x
        for i, module in enumerate(self.decoder):
            index = i + 1
            x     = x[:, :, :features[-abs(index)].size(-2), :features[-abs(index)].size(-1)] + features[-abs(index)]
            x      = module(x, t)
        x = x[:, :, :original.size(-2), :original.size(-1)]
        x = (og[:, :, :original.size(-2), :original.size(-1)] + x)
        # split X
        # layer specific resnet
        cut   = self.process(x)
        synth = (cut[:, :self.layout, :original.size(-2), :original.size(-1)])
        out   = synth
        return out
