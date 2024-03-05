#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for universal image enhancement models.
"""

from __future__ import annotations

__all__ = [
    "UniversalImageEnhancementModel",
]

from abc import ABC

from mon import core
from mon.globals import ZOO_DIR
from mon.vision.enhance import base

console = core.console


# region Model

class UniversalImageEnhancementModel(base.ImageEnhancementModel, ABC):
    """The base class for all universal image enhancement models.
    
    See Also: :class:`base.ImageEnhancementModel`.
    """
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / "vision" / "enhance" / "universal" / self.name
    
# endregion
