#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transformation.

This package contains data transformation functions.
"""

from __future__ import annotations

from typing import Any, TypeAlias, Union

from .base import *
from .nlp import *
from .vision import *

# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

TransformLike: TypeAlias = Union[BasicTransform, dict[str, Any]]
ComposeLike: TypeAlias = Union[
    Compose,
    list[TransformLike],
    dict[str, TransformLike],
]

# endregion


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def build_compose(transforms: ComposeLike | None, *args, **kwargs) -> Compose:
    """Build a ``Compose`` instance from the given input.

    Args:
        transforms (ComposeLike): Either a ``Compose`` instance, a list of
            transformation operations, or a dictionary specifying the
            transformation operations.
        *args: Additional positional arguments for ``Compose`` constructor.
        **kwargs: Additional keyword arguments for ``Compose`` constructor.

    Returns:
        Compose: A ``Compose`` instance containing the specified transformation
            operations.

    Raises:
        ValueError: If no valid transformation operations in ``transforms``.
    """
    if transforms is None:
        return Compose(transforms=[], *args, **kwargs)
    elif isinstance(transforms, Compose):
        return transforms
    elif isinstance(transforms, (list, dict)):
        if isinstance(transforms, dict):
            transforms = list(transforms.values())
        for i, t in enumerate(transforms):
            if isinstance(t, dict):
                transforms[i] = ALBUMENTATIONS.build(**t)
        return Compose(transforms=transforms, *args, **kwargs)
    else:
        raise TypeError(f"Unsupported type: {type(transforms).__name__}")

# endregion
