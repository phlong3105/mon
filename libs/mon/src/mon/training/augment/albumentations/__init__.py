#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Albumentations-based data augmentation and transformation.

This package provides various data augmentations and transformations using the
albumentations library.

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

# __all__ = []  

from typing import Any

from .base import *
from .basic import *
from .ftt import *
from .ifish import *
from .pixel import *
from .resize import *
from .utils import *


# ==============================================================================
# REGISTRY & FACTORY (Type Resolution)
# ==============================================================================

# --- Register (Adding new spokes to the hub) ---


# --- Resolve (Retrieving spokes by name/key) ---
class Compose(A.Compose):
    """An extended version of ``albumentations.Compose`` that builds
    transformations from configuration dictionaries.
    """
    
    def __init__(self, transforms: list[Any], **kwargs):
        """Initialize a new instance.
        
        Args:
            transforms: List of transformations. If any element in ``transforms``
                is a dict, it will be used to build the corresponding
                transformation operation.
            **kwargs: Additional keyword arguments passed to the base
                ``albumentations.Compose``.
        """
        transforms = build_transforms(transforms)
        super().__init__(transforms, **kwargs)


def build_transforms(transforms: list[Any]) -> list[A.BasicTransform]:
    """Build a list of albumentations transformation operations.
    
    Args:
        transforms: A list of transformation operations. If any element in
            ``transforms`` is a dict, it will be used to build the corresponding
            transformation operation.
            
    Returns:
       A list of albumentations transformation operations.
       
    Raises:
        ValueError: If no valid transformation operations are found in ``transforms``.
    """
    transform_ops = []
    for i, t in enumerate(transforms):
        if isinstance(t, dict):
            t = ALBUMENTATIONS.build(**t)
        if t and isinstance(t, A.BasicTransform):
            transform_ops.append(t)
    
    if len(transform_ops) == 0:
        raise ValueError(f"``transforms`` must contain at least one valid transformation.")

    return transform_ops
