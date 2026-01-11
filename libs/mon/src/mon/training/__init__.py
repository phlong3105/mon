#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model training.

This package contains components and functionalities to facilitate the training
of machine learning models.

References:
    - Definition: https://www.ibm.com/think/topics/model-training#1580786329
"""

from __future__ import annotations

__all__ = []

from . import augment, data, loss, metric, optim
from .augment import albumentations
from .data import build_dataloader, build_dataset
