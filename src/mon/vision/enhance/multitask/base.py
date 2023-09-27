#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for multitask image enhancement models.
"""

from __future__ import annotations

__all__ = [
    "MultitaskImageEnhancementModel",
]

from abc import ABC

from mon.core import pathlib
from mon.globals import ZOO_DIR
from mon.vision.enhance import base


# region Model

class MultitaskImageEnhancementModel(base.ImageEnhancementModel, ABC):
    """The base class for all multitask image enhancement models.
    
    See Also: :class:`mon.nn.model.Model`.
    """
    
    @property
    def config_dir(self) -> pathlib.Path:
        return pathlib.Path(__file__).absolute().parent / "config"
    
    @property
    def zoo_dir(self) -> pathlib.Path:
        return ZOO_DIR / "vision" / "enhance" / "llie" / self.name
    
# endregion
