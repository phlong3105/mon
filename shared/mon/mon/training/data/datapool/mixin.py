#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for mixins.

This module provides mixin classes to extend the functionality of data pools.
"""

__all__ = [
    "DataPoolMixin",
    "SAMMixin",
]

import abc

import numpy as np
from ultralytics import SAM

from mon.core import BBoxFormat, Path
from mon.core.dtypes import bbox as B, Image, Instance


# ----- Abstract Mixins -----
class DataPoolMixin(abc.ABC):
    """An abstract mixin class for data pools.

    This class serves as a base for mixin classes that extend the functionality
    of data pools. It does not implement any specific functionality itself,
    but provides a common interface for mixins to build upon.
    """
    
    @abc.abstractmethod
    def on_load_end(self):
        """A hook method called at the end of the data pool loading process.

        This method can be overridden by subclasses to perform additional
        operations after the data pool has been loaded.
        """
        pass


# noinspection PyUnresolvedReferences
class SAMMixin(DataPoolMixin):
    """A mixin class uses Segment Anything Model (SAM) to extract object masks
    from bounding box annotations. This implementation use Ultralytics' SAM
    implementation.
    
    This is primarily used to extend ImageDataPool to support mask extraction.
    We assume access to ``datapoints`` attribute from the base class.
    
    Attributes:
        _sam (SAM): An instance of Ultralytics' SAM model.
        _fg_color (tuple): The RGB color to use for the foreground mask.
    """
    
    def __init__(
        self,
        sam     : str = "sam2.1_l.pt",
        fg_color: tuple[int, int, int] = (255, 255, 255),
        *args, **kwargs
    ):
        """Initializes the SAMMixin.
        
        Args:
            sam (str or SAM): Either the name/path of the SAM model weights, or
                an instance of Ultralytics' SAM model. Defaults to "sam2.1_l.pt".
            fg_color (tuple): The RGB color to use for the foreground mask.
                Defaults to (255, 255, 255).
        """
        super().__init__(*args, **kwargs)
    
        # Initialize SAM model
        if isinstance(sam, str | Path):
            sam = SAM(sam)
        elif isinstance(sam, SAM):
            pass
        else:
            raise TypeError(f"``sam`` must be a string or an instance of ultralytics.SAM, got {type(sam)}.")
        self._sam: SAM = sam
        
        self._fg_color = fg_color
        
    def on_load_end(self):
        """Called at the end of the data pool loading process to extract masks
        using SAM.
        
        This method iterates through all images and their corresponding bounding
        box annotations, uses SAM to extract object masks, and assigns the masks
        to the respective labels.
        
        Raises:
            AttributeError: If the parent class does not have a ``datapoints``
                attribute.
        """
        if not hasattr(self, "datapoints"):
            raise AttributeError("``SAMMixin`` requires ``datapoints`` attribute from parent class.")
        
        # Iterate through all labels
        images : list[Image]          = self.datapoints["image"]
        llabels: list[list[Instance]] = self.datapoints["label"]
        for image, labels in zip(images, llabels):
            # Prepare bounding boxes
            bbox    = np.array([l.data for l in labels])
            bbox    = B.convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=image.imgsz)
            
            # Extract masks using SAM
            results = self._sam(image.data, bboxes=bbox[:, 0:4], device=torch.device("cuda"), verbose=False)
            results = results[0]
            
            # Create semantic mask
            semantic = np.zeros(image.shape, dtype=np.uint8)
            if results.masks is not None:
                for m in results.masks.xy:
                    semantic = cv2.fillPoly(semantic, [np.array(m, dtype=np.int32)], self._fg_color)
           
            # Assign masks to labels
            for l in labels:
                x1, y1, x2, y2 = l.xyxy.astype(int)
                area  = (x2 - x1) * (y2 - y1)
                m     = semantic_mask[y1:y2, x1:x2]
                count = np.count_nonzero(m) / 3.0  # Count non-zero pixels in a single channel
                if count >= float(area * 0.6):     # Skip if the mask is too small
                    # Dilate the mask to ensure it covers the background below the object (e.g., wheels)
                    kernel = np.array([
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]
                    ], dtype=np.uint8)
                    m = cv2.dilate(m, kernel, iterations=1)
                    l.mask = m
