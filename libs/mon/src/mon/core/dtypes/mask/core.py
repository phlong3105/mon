#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mask data structures.

This module provides the base classes and mixins for mask.
"""

from __future__ import annotations

__all__ = [
    "SemanticMask",
]

import cv2

from mon.core.dtypes.image import Image


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
    """A basic class for managing a semantic segmentation mask.
    
    Extend Image to handle semantic mask data and provide properties and methods
    related to mask data.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, flags: int = cv2.IMREAD_GRAYSCALE, *args, **kwargs):
        """Initialize a new instance.

        Args:
            flags: OpenCV flag to read the mask. Defaults to cv2.IMREAD_GRAYSCALE.
        """
        # Continue the initialization chain
        super().__init__(flags=flags, *args, **kwargs)

# endregion
