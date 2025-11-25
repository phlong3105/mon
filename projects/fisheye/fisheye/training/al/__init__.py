#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements active learning strategies for training.

Contents:
    - base.py: Base class for ActiveLearner.
    - query  : Contains various query strategies for picking data to label.
    - models : Learner models, including: wrappers for different frameworks.
    - pool.py: Data pool management (e.g., data, remove samples).
    - utils  : Helper functions, including: data splitters, metrics, random samplers.

References:
    - Website: https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/physicsnemo.active_learning.html#
    - Code: https://github.com/NVIDIA/physicsnemo/tree/main/physicsnemo/active_learning
"""

__all__ = [

]

from .models import *
from .pool import *
from .query import *
from .utils import *
