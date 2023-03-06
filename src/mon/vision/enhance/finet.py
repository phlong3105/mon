#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements HINet (Half-Instance Normalization Network) models."""

from __future__ import annotations

__all__ = [
    "FINet",
]

from typing import Any

from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.enhance import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="finet")
class FINet(base.ImageEnhancementModel):
    """Fractional-Instance Normalization Network.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}

    def __init__(self, config: Any = "finet-a.yaml", *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
    
# endregion
