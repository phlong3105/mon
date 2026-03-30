#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Models.

This module defines the base class for restoration models.
"""

from __future__ import annotations

__all__ = [
    "RestorationModel",
]

from abc import ABC

from mon.nn import Model


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class RestorationModel(Model, ABC):
    """A base class for all restoration models."""

    requires: set = {"image"}
    provides: set = {"restored"}

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
