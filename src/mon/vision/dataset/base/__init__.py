#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements label and dataset types used in vision tasks and
datasets. We try to support all possible data types: :class:`torch.Tensor`,
:class:`numpy.ndarray`, or :class:`Sequence`, but we prioritize
:class:`torch.Tensor`.
"""

from __future__ import annotations

import mon.vision.dataset.base.label
from mon.vision.dataset.base.dataset import *
from mon.vision.dataset.base.label import *
