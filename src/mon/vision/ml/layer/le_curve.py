#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements light enhancement curve.
"""

from __future__ import annotations

__all__ = [
     "PixelwiseHigherOrderLECurve",
]

import torch
from torch import nn

from mon.coreml.layer import base
from mon.globals import LAYERS


@LAYERS.register()
class PixelwiseHigherOrderLECurve(base.MergingLayerParsingMixin, nn.Module):
    """Pixelwise Light-Enhancement Curve is a higher-order curves that can be
    applied iteratively to enable more versatile adjustment to cope with
    challenging low-light conditions:
        LE_{n}(x) = LE_{n−1}(x) + A_{n}(x) * LE_{n−1}(x)(1 − LE_{n−1}(x)),
        
        where `A` is a parameter map with the same size as the given image, and
        `n` is the number of iterations, which controls the curvature.
    
    This module is designed to go with:
        - ZeroDCE   (estimate 3 * n curve parameter maps)
        - ZeroDCE++ (estimate 3   curve parameter maps)
    
    Args:
        n: Number of iterations.
    """
    
    def __init__(self, n: int):
        super().__init__()
        self.n = n
    
    def forward(self, input: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Split
        y = input[0]  # Trainable curve parameters learned from the previous layer
        x = input[1]  # Original input image
        
        # Prepare curve parameter
        _, c1, _, _ = x.shape  # Should be 3
        _, c2, _, _ = y.shape  # Should be 3 * n
        single_map = True
        
        if c2 == c1 * self.n:
            single_map = False
            y = torch.split(y, c1, dim=1)
        elif c2 == 3:
            pass
        else:
            raise ValueError(
                f"Curve parameter maps 'c2' must be '3' or '3 * {self.n}'. "
                f"But got: {c2}."
            )
        
        # Estimate curve parameter
        for i in range(self.n):
            y_i = y if single_map else y[i]
            x   = x + y_i * (torch.pow(x, 2) - x)
        
        y = list(y) if isinstance(y, tuple) else y
        y = torch.cat(y, dim=1) if isinstance(y, list) else y
        return y, x
