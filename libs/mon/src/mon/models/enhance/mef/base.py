#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for multiple-exposure fusion models.
"""

from __future__ import annotations

__all__ = [
    "MEFModel",
]

from abc import ABC

from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class MEFModel(Model, ABC):
    """A base class for all multiple-exposure fusion models."""

    requires: dict = {"images"}
    provides: dict = {"enhanced"}

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
