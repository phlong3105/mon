#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements the data structures for annotations, datasets, and
datamodules.

The base classes are designed to be implemented by the user to create their
own custom labels, datasets, datamodules, and result writers.

We try to support all possible data types: :class:`torch.Tensor`,
:class:`numpy.ndarray`, or :class:`Sequence`, but we prioritize
:class:`torch.Tensor`.
"""

from __future__ import annotations

import mon.data.datastruct.annotation
import mon.data.datastruct.datamodule
import mon.data.datastruct.dataset
from mon.data.datastruct.annotation import *
from mon.data.datastruct.datamodule import *
from mon.data.datastruct.dataset import *
