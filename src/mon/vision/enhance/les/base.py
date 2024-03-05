#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for light effect suppression models.
"""

from __future__ import annotations

__all__ = [
    "LightEffectSuppressionModel",
]

from abc import ABC

from mon import core
from mon.globals import ZOO_DIR, Task
from mon.vision.enhance import base

console = core.console


# region Model

class LightEffectSuppressionModel(base.ImageEnhancementModel, ABC):
    """The base class for all light effect suppression models.
    
    See Also: :class:`base.ImageEnhancementModel`.
    """
    
    _tasks: list[Task] = [Task.LES]
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / "vision" / "enhance" / "les" / self.name
    
# endregion
