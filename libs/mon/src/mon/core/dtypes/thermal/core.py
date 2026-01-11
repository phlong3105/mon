#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Thermal data structures.

This module provides base classes and mixins for thermal data.
"""

from __future__ import annotations

__all__ = [
    "InfraredMap",
]

import cv2

from mon.core.constants import SOURCE
from mon.core.enum import InfraredSource
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

class InfraredMap(Image):
    """Infrared map management class.

    Extend ``Image`` to handle infrared map data and provide properties and
    methods related to infrared data.

    Attributes:
        _source (InfraredSource): Source of the infrared data.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        source: InfraredSource = SOURCE.INFRARED,
        flags : int            = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            source: Source of the infrared data. Defaults to SOURCE.INFRARED.
            flags: OpenCV flag used to read the infrared map.
                Defaults to cv2.IMREAD_GRAYSCALE.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        # Validate and set the depth source
        self._source = InfraredSource(source)
        
        # Continue the initialization chain
        super().__init__(flags=flags, *args, **kwargs)
        
    # ---- Properties ---
    @property
    def source(self) -> InfraredSource:
        """Return the infrared data source."""
        return self._source

# endregion
