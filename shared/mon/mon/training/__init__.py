#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package provides training pipelines and utilities for machine learning models.

References:
    - Definition: https://www.ibm.com/think/topics/model-training#1580786329
"""

__all__ = []  # Populate top modules to ``mon`` namespace.

from mon.training import (
    albumentations as A,
    data as data,
    foundation as foundation,
    losses as losses,
    metrics as metrics,
    optims as optims,
    runtime as runtime,
)

from mon.training.data import *
from mon.training.foundation import *
from mon.training.losses import *
from mon.training.metrics import *
from mon.training.optims import *
from mon.training.runtime import *
