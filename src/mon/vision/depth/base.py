#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for depth estimation models."""

from __future__ import annotations

__all__ = [
    "DepthEstimationModel",
]

from abc import ABC

from mon import core, nn
from mon.globals import Task, ZOO_DIR

console = core.console


# region Model

class DepthEstimationModel(nn.Model, ABC):
    """The base class for all depth estimation models.
    
    See Also: :class:`nn.Model`.
    """
    
    tasks  : list[Task] = [Task.DEPTH]
    zoo_dir: core.Path  = ZOO_DIR / "vision" / "depth"
    
# endregion
