#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Wraps and extends ``albumentations`` package for image augmentations and
transformations on ``numpy.ndarray``.
"""

__all__ = []

from typing import Any

from .core import *
from .fisheye import *
from .pixel import *
from .resize import *


# ----- Extended Compose -----
'''
class Compose(A.Compose):
    """An extension of ``albumentations.Compose`` with convenience methods for
    building transformations.
    
    Args:
        transforms: List of transformations to compose. If any element in
            ``transforms`` is a ``dict``, it will be used to build the
            corresponding transformation operation.
        kwargs: Additional arguments to pass to the ``albumentations.Compose``
            constructor.
    """
    
    def __init__(self, transforms: list[Any], **kwargs):
        transforms = build_transforms(transforms)
        super().__init__(transforms, **kwargs)
'''


# ----- Builder -----
def build_transforms(transforms: list[Any]) -> list[A.BasicTransform]:
    """Builds a ``list`` of transformations.
    
    Args:
        transforms: List of transformations to compose. If any element in
            ``transforms`` is a ``dict``, it will be used to build the
            corresponding transformation operation.
            
    Returns:
        A ``list`` of ``albumentations.BasicTransform`` instances.
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
    """Builds an instance of ``albumentations.Compose``.
    
    Args:
        transforms: List of transformations to compose. If any element in
            ``transforms`` is a ``dict``, it will be used to build the
            corresponding transformation operation.
        kwargs: Additional arguments to pass to the ``albumentations.Compose``
            constructor.
    
    Returns:
        An instance of ``albumentations.Compose`` containing the specified transformations.
    """
    transform_ops = build_transforms(transforms)
    transform     = A.Compose(transforms=transform_ops, **kwargs)
    return transform
