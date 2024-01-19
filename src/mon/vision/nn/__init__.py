#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package extends :mod:`mon.nn` with components for deep learning-based
vision models.
"""

from __future__ import annotations

import mon.vision.nn.layer
import mon.vision.nn.loss
import mon.vision.nn.metric
# noinspection PyUnresolvedReferences
from mon.nn import *
# noinspection PyUnresolvedReferences
from mon.nn.typing import (
	_callable, _ratio_2_t, _ratio_3_t, _ratio_any_t, _scalar_or_tuple_1_t,
	_scalar_or_tuple_2_t, _scalar_or_tuple_3_t, _scalar_or_tuple_4_t,
	_scalar_or_tuple_5_t, _scalar_or_tuple_6_t, _scalar_or_tuple_any_t,
	_size_1_t, _size_2_opt_t, _size_2_t, _size_3_opt_t, _size_3_t, _size_4_t,
	_size_5_t, _size_6_t, _size_any_opt_t, _size_any_t,
)
from mon.vision.nn.layer import *
from mon.vision.nn.loss import *
from mon.vision.nn.metric import *
