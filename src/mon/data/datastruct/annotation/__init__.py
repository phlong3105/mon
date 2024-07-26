#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements multiple annotation types. We try to support all
possible data types: :class:`torch.Tensor`, :class:`numpy.ndarray`, or
:class:`Sequence`, but we prioritize :class:`torch.Tensor`.

The term "annotation" is commonly used in machine learning and deep learning to
describe both ground truth label and model prediction. Basically, both of them
share similar data structure.
"""

from __future__ import annotations

import mon.data.datastruct.annotation.base
import mon.data.datastruct.annotation.bbox
import mon.data.datastruct.annotation.category
import mon.data.datastruct.annotation.classlabel
import mon.data.datastruct.annotation.image
import mon.data.datastruct.annotation.keypoint
import mon.data.datastruct.annotation.polyline
import mon.data.datastruct.annotation.value
from mon.data.datastruct.annotation.base import *
from mon.data.datastruct.annotation.bbox import *
from mon.data.datastruct.annotation.category import *
from mon.data.datastruct.annotation.classlabel import *
from mon.data.datastruct.annotation.image import *
from mon.data.datastruct.annotation.keypoint import *
from mon.data.datastruct.annotation.polyline import *
from mon.data.datastruct.annotation.value import *
