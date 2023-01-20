#!/usr/bin/env python
# -*- coding: utf-8 -*-

""":mod:`mon.vision.dataset.base` package implements label and dataset types
used in vision tasks and datasets. We try to support all possible data types:
:class:`torch.Tensor`, :class:`np.ndarray`, or :class:`Sequence`, but we
prioritize :class:`torch.Tensor`.
"""

from __future__ import annotations

import mon.vision.dataset.base.label
from mon.vision.dataset.base.dataset import *
from mon.vision.dataset.base.label import *
