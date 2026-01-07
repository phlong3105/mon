#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model training.

This package contains components and functionalities to facilitate the training
of machine learning models. It includes data augmentation techniques, dataset
and dataloader builders, loss functions, metrics, and optimization algorithms.

References:
    - Definition: https://www.ibm.com/think/topics/model-training#1580786329
"""

__all__ = []  # Prevent accidental imports of submodules.

from . import augment, data, loss, metric, optim
from .augment import albumentations
from .data import build_dataloader, build_dataset
