#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metrics.

This package contains various metrics used for assessing the performance of
machine learning models.

Notes:
    - Design Pattern: Template Method.
    - Goal: Provide a structured way to define a family of methods or classes
      that share a common interface/inheritance but aren't tied to the specific
      "interchanged" requirement of the "Strategy Pattern".
    - Structure:
        ::
        
            template/
            ├── __init__.py    # Registry and factory logic
            ├── base.py        # Base classes and mixins
            ├── basic.py       # Basic functionalities
            ├── ...
            ├── utils.py       # Utility functions and helpers
            └── external/      # Expose external libraries
                └── ...
"""

__all__ = [
    "ImageQualityAssessment",
    "benchmark",
    "compute_model_stats",
    "scale_gt_mean",
]

from .base import *
from .basic import *
from .complexity import benchmark, compute_model_stats
from .external import *
from .image import ImageQualityAssessment, scale_gt_mean
from .utils import *


# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---


# --- Resolve (Retrieving spokes by name/key) ---
