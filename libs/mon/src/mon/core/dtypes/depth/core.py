#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depth data classes and mixins.

This module provides the base classes and mixins for depth data.
"""

__all__ = [
    "DepthMap",
]

import cv2

from mon.core.enum import DepthSource
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
class DepthMap(Image):
    """A basic class for managing a depth map.

    This class extends Image to handle depth-specific attributes and provide
    properties related to depth data.

    Attributes:
        _source (DepthSource): The configured depth data source.
    """
    
    def __init__(
        self,
        source: DepthSource = DepthSource.DAv2_ViTB,
        flags : int         = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        """Initialize a DepthMap instance.

        Args:
            source: Source of the depth data. Defaults to DepthSource.DAv2_ViTB.
            flags: OpenCV flag used to read the depth map. Defaults to cv2.IMREAD_GRAYSCALE.
            *args: Additional positional arguments forwarded to Image.
            **kwargs: Additional keyword arguments forwarded to Image.

        Raises:
            ValueError: If ``source`` is not a valid DepthSource.
        """
        # Validate inputs
        source = DepthSource(source)
        if source not in DepthSource:
            raise ValueError(f"``source`` must be one of {DepthSource}, got {source}.")
        
        # Assign attributes
        self._source = source
        
        super().__init__(flags=flags, *args, **kwargs)  # This will call the data setter
        
    # ---- Properties ---
    @property
    def source(self) -> DepthSource:
        """Return the configured depth data source."""
        return self._source
