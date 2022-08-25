#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vision.

Taxonomy:
	Vision
	  |__ Low-Level
	  |     |__ Acquisition
	  |     |__ (Image/Video) Enhancement
	  |     |__ Filtering
	  |     |__ (Basic) Transformation
	  |
	  |__ Middle-Level
	  |     |__ Motion
	  |     |__ Segmentation
	  |     |__ Shape
	  |
	  |__ High-Level
	        |__ Action
	        |__ (Image/Video) Classification
	        |__ Detection
	        |__ Reconstruction
	        |__ Reidentification
	        |__ Tracking
"""

from __future__ import annotations

from .classification import *
from .enhancement import *
