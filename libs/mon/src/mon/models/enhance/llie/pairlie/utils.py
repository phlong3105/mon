#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities.

This module provides various utilities for PairLIE.
"""

from __future__ import annotations

__all__ = [
    "joint_L_horizontal",
    "joint_RGB_horizontal",
]

from PIL import Image


# ==============================================================================
# region UTILITIES
# ==============================================================================

def joint_RGB_horizontal(im1, im2) -> Image:
    assert im1.size == im2.size
    w, h = im1.size
    result = Image.new("RGB", (w * 2, h))
    result.paste(im1, box=(0, 0))
    result.paste(im2, box=(w, 0))
    return result


def joint_L_horizontal(im1, im2) -> Image:
    assert im1.size == im2.size
    w, h = im1.size
    result = Image.new("L", (w * 2, h))
    result.paste(im1, box=(0, 0))
    result.paste(im2, box=(w, 0))
    return result

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
