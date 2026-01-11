#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth data structures.

This module provides base classes and mixins for depth data.
"""

from __future__ import annotations

__all__ = [
    "DepthMap",
]

import cv2

from mon.core.constants import SOURCE
from mon.core.enum import DepthSource
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

class DepthMap(Image):
    """Depth map management class.

    Extend Image to handle depth map data and provide properties and methods
    related to depth data.

    Attributes:
        _source (DepthSource): Source of the depth data.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        source: DepthSource = SOURCE.DEPTH,
        flags : int         = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            source: Source of the depth data. Defaults to SOURCE.DEPTH.
            flags: OpenCV flag to read the depth map. Defaults to
                cv2.IMREAD_GRAYSCALE.
            *args: Additional positional arguments forwarded to Image.
            **kwargs: Additional keyword arguments forwarded to Image.
        """
        # Validate and set the depth source
        self._source = DepthSource(source)
        
        # Continue the initialization chain
        super().__init__(flags=flags, *args, **kwargs)
        
    # ---- Properties ---
    @property
    def source(self) -> DepthSource:
        """Return the depth data source."""
        return self._source

# endregion
