#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""``mon`` framework.

This package provides a unified library for research and development. It mainly
covers computer vision and artificial intelligence.
"""

__author__ = "Long H. Pham"
__version__ = "2.10.0"

import time

_start_time = time.time()

from .core import *
from .dataset import transform
from . import cv, dataset, metrics, nn  # This will populate all factories

_end_time = time.time()
log(f"`mon` loaded in: {_end_time - _start_time:.4f} seconds.")
