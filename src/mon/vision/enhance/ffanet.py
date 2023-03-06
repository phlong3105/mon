#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements FFANet models."""

from __future__ import annotations

__all__ = [
    "FFANet",
]

from typing import Any

from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.enhance import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="ffanet")
class FFANet(base.ImageEnhancementModel):
    """Half-Instance Normalization Network.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(self, config: Any = "ffanet.yaml", *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)
    
# endregion
