#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package provides training pipelines and utilities for machine learning models.

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
