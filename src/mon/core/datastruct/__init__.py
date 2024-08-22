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

import mon.core.datastruct.annotation
import mon.core.datastruct.datamodule
import mon.core.datastruct.dataset
from mon.core.datastruct.annotation import *
from mon.core.datastruct.datamodule import *
from mon.core.datastruct.dataset import *
