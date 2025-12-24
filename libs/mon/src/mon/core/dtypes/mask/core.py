#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mask base classes and mixins.

This module provides the base classes and mixins for masks.
"""

__all__ = [
    "SemanticMask",
]

import cv2

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
class SemanticMask(Image):
    """A basic class for managing a semantic segmentation mask.
    
    Extend Image to handle semantic mask data and provide properties and methods
    related to mask data.
    """
    
    def __init__(self, flags: int = cv2.IMREAD_GRAYSCALE, *args, **kwargs):
        """Initialize a new instance.

        Args:
            flags: OpenCV flag to read the mask. Defaults to cv2.IMREAD_GRAYSCALE.
        """
        # Initialize parent classes and assign attributes
        super().__init__(flags=flags, *args, **kwargs)  # This will call the data setter
