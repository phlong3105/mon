#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for enhancement models.
"""

from __future__ import annotations

__all__ = [
    "EnhancementModel",
]

from abc import ABC

from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class EnhancementModel(Model, ABC):
    """A base class for all enhancement models."""

    requires: dict = {"image"}
    provides: dict = {"enhanced"}

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
