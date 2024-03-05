#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements the base classes for different labels, datasets,
datamodules, and result writers.

The base classes are designed to be implemented by the user to create their
own custom labels, datasets, datamodules, and result writers.

We try to support all possible data types: :class:`torch.Tensor`,
:class:`numpy.ndarray`, or :class:`Sequence`, but we prioritize
:class:`torch.Tensor`.
"""

from __future__ import annotations

import mon.data.base.datamodule
import mon.data.base.dataset
import mon.data.base.label
import mon.data.base.writer
from mon.data.base.datamodule import *
from mon.data.base.dataset import *
from mon.data.base.label import *
from mon.data.base.writer import *
