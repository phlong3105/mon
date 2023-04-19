#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module extends the :mod:`enum` module."""

from __future__ import annotations

__all__ = [
    "auto", "Enum", "EnumMeta", "Flag", "IntEnum", "IntFlag", "unique",
]

import enum
import random
from enum import *


# region Enum

class Enum(enum.Enum):
    """An extension of Python :class:`enum.Enum` class."""
    
    @classmethod
    def random(cls):
        """Return a random enum."""
        return random.choice(seq=list(cls))
    
    @classmethod
    def random_value(cls):
        """Return a random enum value."""
        return cls.random().value
    
    @classmethod
    def keys(cls) -> list:
        """Return a list of all enums."""
        return [e for e in cls]
    
    @classmethod
    def values(cls) -> list:
        """A list of all enums' values."""
        return [e.value for e in cls]
    
# endregion
