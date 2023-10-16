#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements datasets, transforms, and models specific to computer
vision.

We seamlessly integrate commonly used vision libraries such as
:mod:`torchvision`, :mod:`kornia`, :mod:`cv2`. The goal is to provide an unified
interface to all vision functions.
"""

from __future__ import annotations

import mon.vision.classify
import mon.vision.core
import mon.vision.dataset
import mon.vision.detect
import mon.vision.drawing
import mon.vision.enhance
import mon.vision.feature
import mon.vision.filter
import mon.vision.geometry
import mon.vision.io
import mon.vision.prior
import mon.vision.tracking
import mon.vision.view
from mon.vision.classify import *
from mon.vision.core import *
from mon.vision.dataset import *
from mon.vision.detect import *
from mon.vision.drawing import *
from mon.vision.enhance import *
from mon.vision.feature import *
from mon.vision.filter import *
from mon.vision.geometry import *
from mon.vision.io import *
from mon.vision.prior import *
from mon.vision.tracking import *
from mon.vision.view import *
