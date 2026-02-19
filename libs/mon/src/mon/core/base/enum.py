#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic Enum Structures.

This module provides enhanced Enum classes with additional functionality.
"""

from __future__ import annotations

__all__ = [
    "DefaultEnumMeta",
    "Enum",
    "EnumType",
    "Flag",
    "IntEnum",
    "IntFlag",
    "ReprEnum",
    "StrEnum",
]

from enum import (
    Enum,
    EnumMeta,
    EnumType,
    Flag,
    IntEnum,
    IntFlag,
    ReprEnum,
    StrEnum as StrEnum_,
)
from typing import Any, override


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class StrEnum(StrEnum_):

    # --- Retrieval ---
    @classmethod
    def names(cls):
        """Returns a list of all enum member names."""
        return [member.name for member in cls]

    @classmethod
    def values(cls):
        """Returns a list of all enum member values."""
        return [member.value for member in cls]

# endregion


# ==============================================================================
# region MIXINS
# ==============================================================================

class DefaultEnumMeta(EnumMeta):
    """Metaclass for Enums with default member support."""

    # --- Callable & Context Manager ---
    @override
    def __call__(cls, value: Any, *args, **kwargs):
        # If no value is passed, return the first member
        if value is None:
            return list(cls)[0]

        # If the value is "default", return the DEFAULT member
        if value == "default":
            return getattr(cls, "DEFAULT", list(cls)[0])

        # Otherwise, behave like a normal Enum
        return super().__call__(value, *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
