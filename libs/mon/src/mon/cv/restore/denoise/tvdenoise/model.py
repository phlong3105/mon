#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Total Variation De-noising."""

__all__ = [
    "TVDenoise",
]

import box
import kornia
import torch

from mon import nn
from mon.core import MLType, MODELS, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(variant="tvdenoise", name="tvdenoise")
class TVDenoise(nn.Module, nn.ModelMetadataMixin):
    
    _arch     : str          = "tvdenoise"
    _name     : str          = "tvdenoise"
    _tasks    : list[Task]   = [Task.DENOISE]
    _mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    _model_dir: Path         = root_dir
    _zoo      : dict         = box.Box()
    
    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss(reduction="mean")
        self.tv = kornia.losses.TotalVariation()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device  = x.device
        self.l2 = self.l2.to(device)
        self.tv = self.tv.to(device)
        y       = nn.Parameter(data=x.clone(), requires_grad=True)
        y       = y.to(device)
        return self.l2_term(y, x) + 0.0001 * self.tv(y)
