#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements data, transforms, and models specific to computer
vision.

We seamlessly integrate commonly used vision libraries such as
:mod:`torchvision`, :mod:`kornia`, :mod:`cv2`. The goal is to provide a unified
interface to all vision functions.
"""

from __future__ import annotations

import mon.vision.classify  # Image Classification
import mon.vision.detect  # Object Detection
import mon.vision.draw  # Drawing Functions
import mon.vision.enhance  # Image Enhancement
import mon.vision.feature  # Feature Extraction
import mon.vision.filter  # Image Filtering
import mon.vision.geometry  # Geometry
import mon.vision.prior  # Vision Prior
import mon.vision.track  # Object Tracking
import mon.vision.view  # Display Functions
from mon.vision.classify import *
from mon.vision.detect import *
from mon.vision.draw import *
from mon.vision.enhance import *
from mon.vision.feature import *
from mon.vision.filter import *
from mon.vision.geometry import *
from mon.vision.prior import *
from mon.vision.track import *
from mon.vision.view import *
