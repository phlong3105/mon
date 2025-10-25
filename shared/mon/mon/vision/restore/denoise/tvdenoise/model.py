#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Total Variation De-noising."""

__all__ = [
    "TVDenoise",
]

import box
import kornia
import torch

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


@MODELS.register(name="tvdenoise", arch="tvdenoise")
class TVDenoise(nn.Module, ModelMixin):
    
    arch     : str          = "tvdenoise"
    name     : str          = "tvdenoise"
    tasks    : list[Task]   = [Task.DENOISE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
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
