#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements data augmentation functions by extending
:mod:`torchvision.transforms` package. These functions are mainly applied to
:class:`torch.Tensor` images.
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
# noinspection PyUnresolvedReferences
import torchvision.transforms
# noinspection PyUnresolvedReferences
from torchvision.transforms import *

from mon import nn


# region Adjust

class RandomGammaCorrection(nn.Module):
    
    def __init__(self, gamma: float | Sequence[float] | None = None):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        from mon.vision import adjust_gamma
        if self.gamma is None:
            # More chances of selecting 0 (original image)
            self.gamma = random.choice([0.5, 1, 2])
            return adjust_gamma(input, self.gamma, gain=1)
        elif isinstance(self.gamma, tuple):
            gamma = random.uniform(*self.gamma)
            return adjust_gamma(input, gamma, gain=1)
        elif self.gamma == 0:
            return input
        else:
            return adjust_gamma(input, self.gamma, gain=1)
    
# endregion
