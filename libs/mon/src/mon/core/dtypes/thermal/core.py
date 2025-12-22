#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Thermal classes and mixins.

This module provides the base classes and mixins for thermal data.
"""

__all__ = [
    "InfraredMap",
]

import cv2

from mon.core.enum import InfraredSource
from ..image import Image


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---


# --- Lifecycle Mixins ---


# --- Compute Mixins ---


# ==============================================================================
# CONCRETE IMPLEMENTATIONS (The Concrete Classes)
# ==============================================================================

# --- Primary Data Types ---
class InfraredMap(Image):
    """A basic class for managing an infrared map.

    This class extends Image to handle infrared-specific attributes and provide
    functionality related to infrared data.

    Attributes:
        _source (InfraredSource): The configured infrared data source.
    """
    
    def __init__(
        self,
        source: InfraredSource = InfraredSource.INFRARED,
        flags : int            = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        """Initialize the InfraredMap instance.

        Args:
            source: Source of the infrared data. Defaults to InfraredSource.INFRARED.
            flags: OpenCV flag used to read the infrared map. Defaults to cv2.IMREAD_GRAYSCALE.

        Raises:
            ValueError: If ``source`` is not a valid InfraredSource.
        """
        # Validate inputs
        source = InfraredSource(source)
        if source not in InfraredSource:
            raise ValueError(f"``source`` must be one of {InfraredSource}, got {source}.")
        
        # Assign attributes
        self._source = source
        
        super().__init__(flags=flags, *args, **kwargs)  # This will call the data setter
     
    # ---- Properties ---
    @property
    def source(self) -> InfraredSource:
        """Return the configured infrared data source."""
        return self._source
