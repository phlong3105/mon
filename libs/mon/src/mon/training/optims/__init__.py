#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Optimizers and learning rate schedulers.

This package contains various optimization algorithms and learning rate
schedulers commonly used in training machine learning models.

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
    "ASGD",
    "Adadelta",
    "Adafactor",
    "Adagrad",
    "Adam",
    "AdamW",
    "Adamax",
    "ChainedScheduler",
    "ConstantLR",
    "CosineAnnealingLR",
    "CosineAnnealingWarmRestarts",
    "CyclicLR",
    "ExponentialLR",
    "LBFGS",
    "LRScheduler",
    "LambdaLR",
    "LinearLR",
    "MultiStepLR",
    "MultiplicativeLR",
    "NAdam",
    "OneCycleLR",
    "Optimizer",
    "PolynomialLR",
    "RAdam",
    "RMSprop",
    "ReduceLROnPlateau",
    "Rprop",
    "SGD",
    "SequentialLR",
    "SparseAdam",
    "StepLR",
]

from .base import *
from .basic import (
    Adadelta,
    Adafactor,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LBFGS,
    LinearLR,
    LRScheduler,
    MultiplicativeLR,
    MultiStepLR,
    NAdam,
    OneCycleLR,
    Optimizer,
    PolynomialLR,
    RAdam,
    ReduceLROnPlateau,
    RMSprop,
    Rprop,
    SequentialLR,
    SGD,
    SparseAdam,
    StepLR,
)
from .external import *
from .utils import *

# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---


# --- Resolve (Retrieving spokes by name/key) ---
