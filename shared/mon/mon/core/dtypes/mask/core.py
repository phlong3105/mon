#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for segmentation mask data type.

This module provides a base class for handling semantic segmentation masks,
extending the image data type with specific attributes and methods for mask
information.
"""

__all__ = [
    "SemanticMask",
]

import cv2

from ..image import Image


class SemanticMask(Image):
    """A base class for a single semantic segmentation mask (i.e., must have a
    valid file path).
    
    This class extends Image to handle a single semantic segmentation mask, which
    can be provided either as an in-memory array/tensor or as a file path. It
    includes methods specific to segmentation masks.
    """
    
    def __init__(self, flags: int = cv2.IMREAD_GRAYSCALE, *args, **kwargs):
        """Initializes the SemanticMask instance.
        
        Args:
            flags (int): OpenCV flag to read segmentation mask. Defaults to
                cv2.IMREAD_GRAYSCALE.
        """
        super().__init__(flags=flags, *args, **kwargs)
