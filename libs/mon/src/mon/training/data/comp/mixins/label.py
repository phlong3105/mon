#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for mixins that perform label processing.

This module defines mixin classes that extend the functionality of data handling
classes in the MON framework by adding label processing capabilities. These
mixins can be combined with other classes to provide features such as mask
extraction using the Segment Anything Model (SAM).
"""

__all__ = [
    "SAMInstanceMixin",
]

import cv2
import numpy as np
import torch
from ultralytics import SAM

from mon.core import BBoxFormat, Path, create_device
from mon.core.dtypes import bbox as B, Image, Instance
from .base import DatasetMixin


class SAMInstanceMixin(DatasetMixin):
    """A mixin that uses Ultralytics' SAM to extract instance masks from bounding
    box annotations.
    
    We assume access to ``datapoints`` attribute from the base class.
    
    Attributes:
        _sam (SAM): An instance of Ultralytics' SAM model.
        _fg_color (tuple): The RGB color to use for the foreground mask.
        device (torch.device): The device to run the SAM model on.
    """
    
    def __init__(
        self,
        sam     : str = "sam2.1_l.pt",
        fg_color: tuple[int, int, int] = (255, 255, 255),
        device  : torch.device = torch.device("cuda"),
        *args, **kwargs
    ):
        """Initializes the SAMMixin.
        
        Args:
            sam (str or SAM): Either the name/path of the SAM model weights, or
                an instance of Ultralytics' SAM model. Defaults to "sam2.1_l.pt".
            fg_color (tuple): The RGB color to use for the foreground mask.
                Defaults to (255, 255, 255).
            device (torch.device): The device to run the SAM model on.
                Defaults to torch.device("cuda").
        """
        super().__init__(*args, **kwargs)
        # Validate and set device
        self.device = create_device(device)
        
        # Initialize SAM model
        if isinstance(sam, str | Path):
            sam = SAM(sam)
        elif isinstance(sam, SAM):
            pass
        else:
            raise TypeError(f"``sam`` must be a string or an instance of ultralytics.SAM, got {type(sam)}.")
        self._sam: SAM = sam
        
        self._fg_color = fg_color
    
    # --- Hooks ---
    def on_load_start(self):
        """Called before load() to prepare the SAM model.
        
        This method ensures that the SAM model is moved to the correct device
        before any data loading occurs.
        """
        pass
    
    # noinspection PyUnresolvedReferences
    def on_load_end(self):
        """Called after load() to extract instance masks using SAM.
        
        This method iterates through all images and their corresponding bounding
        box annotations, uses SAM to extract object masks, and assigns the masks
        to the respective labels.
        
        Raises:
            AttributeError: If the parent class does not have the required
                ``datapoints`` attribute or if it does not contain both
                ``image`` and ``label`` keys.
        """
        if not hasattr(self, "datapoints"):
            raise AttributeError("``SAMMixin`` requires ``datapoints`` attribute from parent class.")
        if "image" not in self.datapoints or "label" not in self.datapoints:
            raise AttributeError("``datapoints`` must contain both ``image`` and ``label`` keys.")
        
        # Iterate through all labels
        images     : list[Image]          = self.datapoints["image"]
        labels_list: list[list[Instance]] = self.datapoints["label"]
        for image, labels in zip(images, labels_list):
            # Prepare bounding boxes
            bbox    = np.array([l.data for l in labels])
            bbox    = B.convert(bbox, fmt=BBoxFormat.CXCYWHN2XYXY, imgsz=image.imgsz)
            
            # Extract masks using SAM
            results = self._sam(image.data, bboxes=bbox[:, 0:4], device=self.device, verbose=False)
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
                m     = semantic[y1:y2, x1:x2]
                count = np.count_nonzero(m) / 3.0  # Count non-zero pixels in a single channel
                if count >= float(area * 0.6):     # Skip if the mask is too small
                    # Dilate the mask to ensure it covers the background below the object (e.g., wheels)
                    kernel = np.array([
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]
                    ], dtype=np.uint8)
                    m      = cv2.dilate(m, kernel, iterations=1)
                    l.mask = m
