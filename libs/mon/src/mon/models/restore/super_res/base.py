#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for super-resolution models.
"""

from __future__ import annotations

__all__ = [
    "SuperResolutionModel",
]

from abc import ABC

from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class SuperResolutionModel(Model, ABC):
    """A base class for all super-resolution models."""

    requires: set = {"x_lr", "y_hr"}
    provides: set = {"x_hr"}

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
