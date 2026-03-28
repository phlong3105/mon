#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for background subtraction models.
"""

from __future__ import annotations

__all__ = [
    "BackgroundSubtractionModel",
]

from abc import ABC

from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class BackgroundSubtractionModel(Model, ABC):
    """A base class for all background subtraction models."""

    requires: dict = {"image"}
    provides: dict = {"background", "foreground"}

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
