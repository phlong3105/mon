#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Total Variation De-noising.

This module implements Total Variation de-noising model.
"""

from __future__ import annotations

__all__ = [
    "TVDenoise",
]

import kornia
import torch

from mon import core, nn

console = core.console


# region Model

class TVDenoise(nn.Module):
    
    def __init__(self, noisy_image: torch.Tensor):
        super().__init__()
        self.l2_term        = nn.MSELoss(reduction="mean")
        self.regularization = kornia.losses.TotalVariation()
        # Create the variable, which will be optimized to produce the noise-free image.
        self.clean_image    = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image    = noisy_image
    
    @property
    def clean_image(self) -> torch.Tensor:
        return self._clean_image
    
    @clean_image.setter
    def clean_image(self, clean_image: torch.Tensor):
        self._clean_image = clean_image
      
    def forward(self):
        return (
            self.l2_term(self.clean_image, self.noisy_image)
            + 0.0001 * self.regularization(self.clean_image)
        )
    
# endregion
