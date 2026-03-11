#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""IZ-DCE utilities.

This module provides various utilities for IZ-DCE.
"""

from __future__ import annotations

__all__ = [
    "weights_init",
]


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Weights Initialization ---

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
