#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""External APIs.

This module collects all external functionalities that are commonly used in
this package. It is intended to be imported by other modules for convenience.
"""

from __future__ import annotations

__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from torch.nn.modules.linear import Bilinear, Identity, LazyLinear, Linear


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
