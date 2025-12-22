#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Constants and type definitions for training data.

Provide constants and type definitions for training data.
"""

__all__ = [
    "Modalities",
    "Modality",
]

from collections import namedtuple
from typing import Dict, TypeAlias

Modality  = namedtuple("Modality", [
    "name",     # The name of the directory that contains the modality data.
    "type",     # Albumentations target type (e.g. "image", "mask", ...) for augmentations.
    "module",   # The tensor class that performs I/O operations.
    "train",    # If ``True``, this modality is included in train/val set.
    "test",     # If ``True``, this modality is included in test set.
    "primary"   # If ``True``, this is the primary modality.
], defaults=[None, None, True, False, False])
Modalities: TypeAlias = Dict[str, Modality]
