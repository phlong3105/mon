#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fold and unfold layers.

This module provides various fold and unfold layers used for reducing the
spatial dimensionality of feature maps.
"""

from __future__ import annotations

__all__ = [
    "Fold",
    "Unfold",
]

from torch.nn.modules.fold import Fold, Unfold


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
