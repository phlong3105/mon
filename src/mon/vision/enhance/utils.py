#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utils

This module implements utility functions for the enhancement models.
"""

from __future__ import annotations

__all__ = [
	"PseudoGTGenerator",
]

import torch

from mon import core, nn

console = core.console


# region Pseudo-GT Image Generator

class PseudoGTGenerator:
    """To create the pseudo GT image, we compare and combine the 2N generated
    reference images, the original image, and the output of the enhancement
    network of the same image in the previous epoch.
    
    References:
        https://github.com/VinAIResearch/PSENet-Image-Enhancement/blob/main/source/framework.py
    """
    
    def __init__(
        self,
        number_refs  : int   = 1,
        gamma_upper  : float = 3.0,
        gamma_lower  : float = -2.0,
        exposed_level: float = 0.5,
        pool_size    : int   = 25,
    ):
        self.number_refs = number_refs
        self.gamma_upper = gamma_upper
        self.gamma_lower = gamma_lower
        self.iqa         = nn.GoodLookingImageMetric(exposed_level=exposed_level, pool_size=pool_size)
    
    def __call__(
        self,
        image      : torch.Tensor,
        prev_output: torch.Tensor = None,
    ) -> torch.Tensor:
        b, c, h, w           = image.shape
        underexposed_ranges  = torch.linspace(0, self.gamma_upper, steps=self.number_refs + 1).to(image.device)[:-1]
        step_size            = self.gamma_upper / self.number_refs
        underexposed_gamma   = torch.exp(torch.rand([b, self.number_refs], device=image.device) * step_size + underexposed_ranges[None, :])
        overrexposed_ranges  = torch.linspace(self.gamma_lower, 0, steps=self.number_refs + 1).to(image.device)[:-1]
        step_size            = - self.gamma_lower / self.number_refs
        overrexposed_gamma   = torch.exp(torch.rand([b, self.number_refs], device=image.device) * overrexposed_ranges[None, :])
        gammas               = torch.cat([underexposed_gamma, overrexposed_gamma], dim=1)
        # gammas: [b, nref], im: [b, c, h, w] -> synthetic_references: [b, nref, c, h, w]
        synthetic_references = 1 - (1 - image[:, None]) ** gammas[:, :, None, None, None]
        
        if prev_output is not None:
            # previous_iter_output = self.model(image)[0].clone().detach()
            prev_output = prev_output.clone().detach()
            references  = torch.cat([image[:, None], prev_output[:, None], synthetic_references], dim=1)
        else:
            references  = torch.cat([image[:, None], synthetic_references], dim=1)
	       
        nref       = references.shape[1]
        scores     = self.iqa(references.view(b * nref, c, h, w))
        scores     = scores.view(b, nref, 1, h, w)
        max_idx    = torch.argmax(scores, dim=1)
        max_idx    = max_idx.repeat(1, c, 1, 1)[:, None]
        pseudo_gt  = torch.gather(references, 1, max_idx)
        return pseudo_gt.squeeze(1)

# endregion
