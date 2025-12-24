#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model training pipelines.

This package provides modules and functions to facilitate the training of machine
learning models. It includes data augmentation techniques, dataset and dataloader
builders, loss functions, metrics, and optimization algorithms.

References:
    - Definition: https://www.ibm.com/think/topics/model-training#1580786329
"""

from . import augment, data, losses, metrics, optims
from .augment import albumentations
from .data import build_dataloader, build_dataset
