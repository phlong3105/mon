#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for de-blurring models."""

from __future__ import annotations

__all__ = [
    "DeblurringModel",
]

from abc import ABC

from mon import core
from mon.globals import Task, ZOO_DIR
from mon.vision.enhance import base

console = core.console


# region Model

class DeblurringModel(base.ImageEnhancementModel, ABC):
    """The base class for all de-blurring models."""
    
    tasks  : list[Task] = [Task.DEBLUR]
    zoo_dir: core.Path  = ZOO_DIR / "vision" / "enhance" / "deblur"
    
# endregion
