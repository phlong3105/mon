#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for depth map data type.

This module provides a base class for handling depth map data.
"""

__all__ = [
    "DepthMap",
]

import cv2

from mon.core.enum import DepthSource
from ..image import Image


class DepthMap(Image):
    """A base class for depth map data type.
    
    This class extends Image to handle depth map data. It includes properties
    to access the source of depth data.
    """
    
    def __init__(
        self,
        source: DepthSource = DepthSource.DAv2_ViTB,
        flags : int         = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        """Initializes the DepthMap instance.
        
        Args:
            source (DepthSource): The source of depth data. Defaults to
                DepthSource.DAv2_ViTB.
            flags (int): Flags for image loading. Defaults to cv2.IMREAD_GRAYSCALE.
        
        Raises:
            ValueError: If ``source`` is not a valid DepthSource.
        """
        super().__init__(flags=flags, *args, **kwargs)
        
        # Validate inputs
        source = DepthSource(source)
        if source not in DepthSource:
            raise ValueError(f"``source`` must be one of {DepthSource}, got {source}.")
        
        # Assign attributes
        self._source = source
        
    # ---- Properties -----
    @property
    def source(self) -> DepthSource:
        """Getter for the source of depth data.
        
        Returns:
            DepthSource: The source of depth data.
        """
        return self._source
