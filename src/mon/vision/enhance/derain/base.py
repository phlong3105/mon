#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for single-image deraining models.
"""

from __future__ import annotations

__all__ = [
    "DerainingModel",
]

from abc import ABC

from mon import core
from mon.globals import Task, ZOO_DIR
from mon.vision.enhance import base

console = core.console


# region Model

class DerainingModel(base.ImageEnhancementModel, ABC):
    """The base class for all single-image deraining models.
    
    See Also: :class:`base.ImageEnhancementModel`.
    """
    
    tasks  : list[Task] = [Task.DERAIN]
    zoo_dir: core.Path  = ZOO_DIR / "vision" / "enhance" / "derain"
    
# endregion
