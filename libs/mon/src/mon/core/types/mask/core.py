#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mask data structures.

This module provides base classes and mixins for masks.
"""

from __future__ import annotations

__all__ = [
    "SemanticMask",
]

import cv2

from ..image import Image


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class SemanticMask(Image):
    """Semantic segmentation mask management class.

    Extend ``Image`` to handle semantic mask data and provide properties and
    methods related to mask data.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, flags: int = cv2.IMREAD_GRAYSCALE, *args, **kwargs):
        """Initialize a new instance.

        Args:
            flags: OpenCV flag to read the mask. Defaults to cv2.IMREAD_GRAYSCALE.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        # Continue the initialization chain
        super().__init__(flags=flags, *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
