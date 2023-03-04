#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements HINet (Half-Instance Normalization Network) models."""

from __future__ import annotations

__all__ = [
    "HINet",
]

from torch import nn

from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.enhance import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="hinet")
class HINet(base.ImageEnhancementModel):
    """Half-Instance Normalization Network.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs = {
        # "hinet": _current_dir/"config/hinet.yaml"
        
    }
    zoo     = {
        "hinet-deblur-gopro"     : dict(
            name        = "gopro",
            path        = "",
            filename    = "hinet-deblur-gopro.pth",
            num_classes = None,
        ),
        "hinet-deblur-reds"      : dict(
            name        = "reds",
            path        = "",
            filename    = "hinet-deblur-reds.pth",
            num_classes = None,
        ),
        "hinet-denoise-sidd-x1.0": dict(
            name        = "sidd",
            path        = "",
            filename    = "hinet-denoise-sidd-x1.0.pth",
            num_classes = None,
        ),
        "hinet-denoise-sidd-x0.5": dict(
            name        = "sidd",
            path        = "",
            filename    = "hinet-denoise-sidd-x0.5.pth",
            num_classes = None,
        ),
        "hinet-derain-rain13k"   : dict(
            name        = "rain13k",
            path        = "",
            filename    = "hinet-derain-rain13k.pth",
            num_classes = None,
        ),
    }
    
    def init_weight(self, m: nn.Module):
        """Initialize model's weights."""
        pass

# endregion
