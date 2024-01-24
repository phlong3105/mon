#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements multiple label types. We try to support all possible
data types: :class:`torch.Tensor`, :class:`numpy.ndarray`, or :class:`Sequence`,
but we prioritize :class:`torch.Tensor`.
"""

from __future__ import annotations

import mon.data.base.label.base
import mon.data.base.label.classlabel
import mon.data.base.label.detection
import mon.data.base.label.heatmap
import mon.data.base.label.image
import mon.data.base.label.keypoint
import mon.data.base.label.polyline
import mon.data.base.label.regression
import mon.data.base.label.segmentation
from mon.data.base.label.base import *
from mon.data.base.label.classlabel import *
from mon.data.base.label.detection import *
from mon.data.base.label.heatmap import *
from mon.data.base.label.image import *
from mon.data.base.label.keypoint import *
from mon.data.base.label.polyline import *
from mon.data.base.label.regression import *
from mon.data.base.label.segmentation import *
