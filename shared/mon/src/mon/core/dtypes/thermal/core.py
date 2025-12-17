#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for thermal map data type.

This module provides a base class for handling infrared map data, extending the
image data type with specific attributes and methods for infrared information.
"""

__all__ = [
    "InfraredMap",
]

import cv2

from mon.core.enum import InfraredSource
from ..image import Image


class InfraredMap(Image):
    """A base class for a single infrared map (i.e., must have a valid file path).
    
    This class extends Image to handle a single infrared map, which can be
    provided either as an in-memory array/tensor or as a file path. It includes
    an attribute to specify the source of the infrared data.
    """
    
    def __init__(
        self,
        source: InfraredSource = InfraredSource.INFRARED,
        flags : int            = cv2.IMREAD_GRAYSCALE,
        *args, **kwargs
    ):
        """Initializes the InfraredMap instance.
        
        Args:
            source (InfraredSource): The source of the infrared data. Defaults
                to InfraredSource.INFRARED.
            flags (int): OpenCV flag to read infrared map. Defaults to
                cv2.IMREAD_GRAYSCALE.
        """
        super().__init__(flags=flags, *args, **kwargs)
        
        # Validate inputs
        source = InfraredSource(source)
        if source not in InfraredSource:
            raise ValueError(f"``source`` must be one of {InfraredSource}, got {source}.")
        
        # Assign attributes
        self._source = source
     
    # ---- Properties -----
    @property
    def source(self) -> InfraredSource:
        """Getter for the source of the infrared data.
        
        Returns:
            InfraredSource: The source of the infrared data.
        """
        return self._source
