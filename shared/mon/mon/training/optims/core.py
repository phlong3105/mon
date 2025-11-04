#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registers ``torch``'s optimizers and learning rate schedulers."""

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

from torch.optim import *               # Expose all optimizers from ``torch.optim``
from torch.optim.lr_scheduler import *  # Expose all schedulers from ``torch.optim.lr_scheduler``
