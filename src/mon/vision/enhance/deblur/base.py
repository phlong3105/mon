#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for de-blurring models."""

from __future__ import annotations

__all__ = [
    "DeblurringModel",
]

from abc import ABC

from mon import core
from mon.globals import ZOO_DIR, Task
from mon.vision.enhance import base

console = core.console


# region Model

class DeblurringModel(base.ImageEnhancementModel, ABC):
    """The base class for all de-blurring models.
    
    See Also: :class:`base.ImageEnhancementModel`.
    """
    
    _tasks: list[Task] = [Task.DEBLUR]
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / "vision" / "enhance" / "deblur" / self.name
    
# endregion
