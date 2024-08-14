#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for de-hazing models."""

from __future__ import annotations

__all__ = [
    "DehazingModel",
]

from abc import ABC

from mon import core
from mon.globals import Task, ZOO_DIR
from mon.vision.enhance import base

console = core.console


# region Model

class DehazingModel(base.ImageEnhancementModel, ABC):
    """The base class for all de-hazing models.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`.
    """
    
    tasks  : list[Task] = [Task.DEHAZE]
    zoo_dir: core.Path  = ZOO_DIR / "vision" / "enhance" / "dehaze"
    
# endregion
