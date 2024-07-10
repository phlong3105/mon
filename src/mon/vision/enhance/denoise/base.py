#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for de-nosing models.
"""

from __future__ import annotations

__all__ = [
    "DenoisingModel",
]

from abc import ABC

from mon import core
from mon.globals import Task, ZOO_DIR
from mon.vision.enhance import base

console = core.console


# region Model

class DenoisingModel(base.ImageEnhancementModel, ABC):
    """The base class for all de-noising models.
    
    See Also: :class:`base.ImageEnhancementModel`.
    """
    
    _tasks: list[Task] = [Task.DENOISE]
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / "vision" / "enhance" / "denoise"
    
# endregion
