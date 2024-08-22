#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data Structures Package.

This package implements the data structures for annotations, datasets, and
datamodules. The base classes are designed to be implemented by the user to
create their own custom labels, datasets, datamodules, and result writers.

We try to support all possible data types: :obj:`torch.Tensor`,
:obj:`numpy.ndarray`, or :obj:`Sequence`, but we prioritize :obj:`torch.Tensor`.
"""

from __future__ import annotations

import mon.core.data.annotation
import mon.core.data.datamodule
import mon.core.data.dataset
from mon.core.data.annotation import *
from mon.core.data.datamodule import *
from mon.core.data.dataset import *
