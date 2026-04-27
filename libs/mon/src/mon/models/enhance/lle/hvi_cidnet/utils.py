#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities.

This module provides various utilities for HVI-CIDNet.
"""

from __future__ import annotations

from PIL import Image

__all__ = [
    "is_image_file",
    "load_image",
]


# ==============================================================================
# region UTILITIES
# ==============================================================================

def is_image_file(filename: str) -> bool:
    return any(
        filename.endswith(extension)
        for extension in [".png", ".jpg", ".bmp", ".JPG", ".jpeg"]
    )


def load_image(filepath: str) -> Image.Image:
    image = Image.open(filepath).convert('RGB')
    return image


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
