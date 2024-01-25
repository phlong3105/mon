#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements label and dataset types. We try to support all
possible data types: :class:`torch.Tensor`, :class:`numpy.ndarray`, or
:class:`Sequence`, but we prioritize :class:`torch.Tensor`.
"""

from __future__ import annotations

import mon.data.base.datamodule
import mon.data.base.dataset
import mon.data.base.label
from mon.data.base.datamodule import *
from mon.data.base.dataset import *
from mon.data.base.label import *
