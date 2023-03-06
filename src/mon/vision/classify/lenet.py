#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements LeNet models."""

from __future__ import annotations

__all__ = [
    "LeNet",
]

from typing import Any

from mon.coreml import model as mmodel
from mon.foundation import pathlib
from mon.globals import MODELS
from mon.vision.classify import base

_current_dir = pathlib.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="lenet")
class LeNet(base.ImageClassificationModel):
    """LeNet.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        if "config" in kwargs:
            _ = kwargs.pop("config")
        super().__init__(config="lenet.yaml", *args, **kwargs)
    
# endregion
