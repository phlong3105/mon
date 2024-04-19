#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for enhancement models."""

from __future__ import annotations

__all__ = [
    "ImageEnhancementModel",
]

from abc import ABC
from typing import Sequence

import torch

from mon import core, nn
from mon.globals import ZOO_DIR

console = core.console


# region Model

class ImageEnhancementModel(nn.Model, ABC):
    """The base class for all image enhancement models.
    
    See Also: :class:`nn.Model`.
    """
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / "mon" / "vision" / "enhance" / self.name
    
# endregion
