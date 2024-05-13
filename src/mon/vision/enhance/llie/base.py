#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for low light image enhancement models.
"""

from __future__ import annotations

__all__ = [
    "LowLightImageEnhancementModel",
]

from abc import ABC

from mon import core
from mon.globals import Task, ZOO_DIR
from mon.vision.enhance import base

console = core.console


# region Model

class LowLightImageEnhancementModel(base.ImageEnhancementModel, ABC):
    """The base class for all low light image enhancement models.
    
    See Also: :class:`base.ImageEnhancementModel`.
    """
    
    _tasks: list[Task] = [Task.LLIE]
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / "mon" / "vision" / "enhance" / "llie"
    
# endregion
