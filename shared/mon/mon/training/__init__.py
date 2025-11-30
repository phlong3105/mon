#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for model training pipelines.

This package provides modules and functions to facilitate the training of machine
learning models. It includes data augmentation techniques, dataset and dataloader
builders, loss functions, metrics, and optimization algorithms.

References:
    - Definition: https://www.ibm.com/think/topics/model-training#1580786329
"""

__all__ = [
    # Flat exposed APIs
    "build_dataloader",
    "build_dataset",
    # Hierarchical exposed APIs
    "albumentations",
    "augment",
    "data",
    "losses",
    "metrics",
    "optims",
]  # Public APIs

from .augment import albumentations
from .data import build_dataloader, build_dataset
