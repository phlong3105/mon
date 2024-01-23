#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements evaluation metrics by extending the
:mod:`torchmetrics` package.
"""

from __future__ import annotations

import mon.nn.metric.base
import mon.nn.metric.classification
import mon.nn.metric.efficiency
import mon.nn.metric.image
import mon.nn.metric.nominal
import mon.nn.metric.regression
from mon.nn.metric.base import *
from mon.nn.metric.classification import *
from mon.nn.metric.efficiency import *
from mon.nn.metric.image import *
from mon.nn.metric.nominal import *
from mon.nn.metric.regression import *
