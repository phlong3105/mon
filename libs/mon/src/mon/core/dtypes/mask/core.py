#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Mask data classes and mixins.

This module provides the base classes and mixins for mask data.
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
    
    This class extends Image to handle semantic segmentation mask-specific
    operations.
    """
    
    def __init__(self, flags: int = cv2.IMREAD_GRAYSCALE, *args, **kwargs):
        """Initialize the semantic segmentation mask.

        Args:
            flags: OpenCV flag used to read the segmentation mask. Defaults to cv2.IMREAD_GRAYSCALE.
        """
        super().__init__(flags=flags, *args, **kwargs)  # This will call the data setter
