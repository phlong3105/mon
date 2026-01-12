#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic optimizers and learning rate schedulers from PyTorch.

This module provides various optimizers and learning rate schedulers commonly
used in training machine learning models.
"""

from __future__ import annotations

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
    # "lr_scheduler",
    # "swa_utils",
]

from torch.optim import (
    Adafactor,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    LBFGS,
    lr_scheduler,
    Muon,
    NAdam,
    Optimizer,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam,
    swa_utils,
)
from torch.optim.lr_scheduler import (
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    SequentialLR,
    CosineAnnealingLR,
    ChainedScheduler,
    ReduceLROnPlateau,
    CyclicLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialLR,
    LRScheduler,
)
