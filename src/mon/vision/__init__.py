#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements data, transforms, and models specific to computer
vision.

We seamlessly integrate commonly used vision libraries such as
:mod:`torchvision`, :mod:`kornia`, :mod:`cv2`. The goal is to provide a unified
interface to all vision functions.

A note for other developers:
- I extend :mod:`mon.core` and :mod:`mon.nn` inside this package so that all
importing calls can be simplified.
"""

from __future__ import annotations

import mon.vision.classify  # Image Classification
import mon.vision.core      # Basic Operations (extending :mod:`mon.core`)
import mon.vision.data      # Data
import mon.vision.detect    # Object Detection
import mon.vision.drawing   # Drawing Functions
import mon.vision.enhance   # Image Enhancement
import mon.vision.feature   # Feature Extraction
import mon.vision.filter    # Image Filtering
import mon.vision.geometry  # Geometry
import mon.vision.io        # Image/Video IO
import mon.vision.nn        # Neural Network Components (extending :mod:`mon.nn`)
import mon.vision.prior     # Vision Prior
import mon.vision.tracking  # Object Tracking
import mon.vision.view      # Display Functions
from mon.vision.classify import *
from mon.vision.core import *
from mon.vision.data import *
from mon.vision.detect import *
from mon.vision.drawing import *
from mon.vision.enhance import *
from mon.vision.feature import *
from mon.vision.filter import *
from mon.vision.geometry import *
from mon.vision.io import *
from mon.vision.nn import *
from mon.vision.prior import *
from mon.vision.tracking import *
from mon.vision.view import *
