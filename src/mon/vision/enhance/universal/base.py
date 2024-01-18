#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for universal image enhancement models.
"""

from __future__ import annotations

__all__ = [
    "UniversalImageEnhancementModel",
]

from abc import ABC

from mon.globals import ZOO_DIR
from mon.vision import core
from mon.vision.enhance import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Model

class UniversalImageEnhancementModel(base.ImageEnhancementModel, ABC):
    """The base class for all universal image enhancement models.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`.
    """
    
    @property
    def config_dir(self) -> core.Path:
        return core.Path(__file__).absolute().parent / "config"
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / "vision" / "enhance" / "universal" / self.name
    
# endregion
