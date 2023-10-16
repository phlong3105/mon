#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package extends :mod:`mon.nn` with components for deep learning-based
vision models.
"""

from __future__ import annotations

import mon.vision.nn.loss
import mon.vision.nn.metric
# noinspection PyUnresolvedReferences
from mon.nn import *
from mon.vision.nn.loss import *
from mon.vision.nn.metric import *
