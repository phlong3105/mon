#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for albumentations-based data augmentation and transformation.

This package provides various data augmentation and transformation techniques
using the albumentations library. It includes custom transformations and
utilities to build and compose complex augmentation pipelines for image data.
"""

__all__ = []

from typing import Any

from .core import *
from .ftt import FisheyeTomographyTransform
from .ifish import iFishTransform
from .pixel import NormalizeWithMask
from .resize import ResizeDivisibleBy


# ----- Extended Compose -----
class Compose(A.Compose):
    """An extended version of ``albumentations.Compose`` that builds transformations
    from configuration dictionaries.
    """
    
    def __init__(self, transforms: list[Any], **kwargs):
        """Initializes the Compose instance.
        
        Args:
            transforms (list[Any]): List of transformations. If any element in
                ``transforms`` is a dict, it will be used to build the corresponding
                transformation operation.
            **kwargs: Additional keyword arguments passed to the base
                ``albumentations.Compose``.
        """
        transforms = build_transforms(transforms)
        super().__init__(transforms, **kwargs)


# ----- Builder -----
def build_transforms(transforms: list[Any]) -> list[A.BasicTransform]:
    """Builds a list of albumentations transformation operations.
    
    Args:
        transforms (list[Any]): A list of transformation operations. If any
            element in ``transforms`` is a dict, it will be used to build the
            corresponding transformation operation.
            
    Returns:
        list[A.BasicTransform]: A list of albumentations transformation operations.
        
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


def build_compose(transforms: list[Any], **kwargs) -> A.Compose:
    """Builds an ``albumentations.Compose`` instance from a list of transformations.
    
    Args:
        transforms (list[Any]): A list of transformations. If any element in
            ``transforms`` is a dict, it will be used to build the corresponding
            transformation operation.
        **kwargs: Additional keyword arguments passed to the ``albumentations.Compose``.
    
    Returns:
        A.Compose: An ``albumentations.Compose`` instance.
    """
    transform_ops = build_transforms(transforms)
    transform     = A.Compose(transforms=transform_ops, **kwargs)
    return transform
