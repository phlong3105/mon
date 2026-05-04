#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Transformation.

This package contains data transformation functions.

File Structure:
::

    transform/
    ├── __init__.py
    ├── base.py         # Expose Albumentations transformations
    ├── nlp.py          # NLP-specific transformations
    └── vision.py       # Vision-specific transformations
"""

from __future__ import annotations

from typing import Any

# noinspection PyUnusedImports
from albumentations.core.composition import (
    BaseCompose,
    BboxParams,
    Compose as Compose_,
    KeypointParams,
    OneOf,
    OneOrOther,
    RandomOrder,
    ReplayCompose,
    SelectiveChannelTransform,
    Sequential,
    SomeOf,
)
# noinspection PyUnusedImports
from albumentations.core.transforms_interface import (
    BasicTransform as BaseTransform_,
    CustomTransformsApplyMixin,
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    Transform3D,
)

from .base import *
from .nlp import *
from .vision import *


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class BasicTransform(BaseTransform_):
    """Extend ``albumentations.BaseTransform`` with convenience methods for
     building transformation pipelines.
     """

    # --- Creation ---
    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs) -> "BasicTransform":
        """Create a new instance from a configuration dictionary."""
        # Extract relevant keys
        name = config.pop("name", config.pop("type", None))

        if name is None:
            raise KeyError(f"Missing 'name' or 'type' key in 'config': {config}.")

        # Build the object
        config |= kwargs
        return ALBUMENTATIONS.build(name=name, **config)


class Compose(Compose_):
    """Extend ``albumentations.Compose`` with convenience methods for building
     transformation pipelines.
     """

    # --- Creation ---
    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs) -> "Compose":
        """Create a new instance from a configuration dictionary."""
        # Extract relevant keys
        transforms_: list[Any] = config.pop("transforms", config.pop("ops", []))

        # Validate inputs
        if not isinstance(transforms_, list):
            raise TypeError(
                f"Expected a list of transformations, "
                f"but got: {type(transforms_).__name__}."
            )

        # Build the objects
        for i, t in enumerate(transforms_):
            if isinstance(t, dict):
                transforms_[i] = BasicTransform.from_config(t)

        # Return the new instance
        config |= kwargs
        return cls(transforms=transforms_, **config)

    @classmethod
    def from_transforms(cls, transforms: list[BasicTransform | dict[str, Any]], **kwargs) -> "Compose":
        """Create a new instance from a list of transformations."""
        # Validate inputs
        if not isinstance(transforms, list):
            raise TypeError(
                f"Expected a list of transformations, "
                f"but got: {type(transforms).__name__}."
            )

        # Build the objects
        for i, t in enumerate(transforms):
            if isinstance(t, dict):
                transforms[i] = BasicTransform.from_config(t)

        # Return the new instance
        return cls(transforms=transforms, **kwargs)

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def build_transform(value: Any, **kwargs) -> BasicTransform | None:
    """Build a transformation instance from a given value.

    Args:
        value (Any): Either a transformation instance or a configuration
            dictionary to build the transformation from.
        **kwargs: Additional keyword arguments to pass to the transformation
            constructor.

    Returns:
        BasicTransform | None: A transformation instance, or None if the input
            value is None.

    Raises:
        TypeError: If the input value is not a valid type for building a
            transformation.
    """
    if value is None:
        return None
    elif isinstance(value, BasicTransform):
        return value
    elif isinstance(value, dict):
        return BasicTransform.from_config(value, **kwargs)
    else:
        raise TypeError(f"Unsupported transformation type: {type(value).__name__}.")


def build_compose(value: Any, **kwargs) -> Compose | None:
    """Build a ``Compose`` instance from a given value.

    Args:
        value (Any): Either a ``Compose`` instance, a list of transformations,
            or a configuration dictionary to build the ``Compose`` instance from.
        **kwargs: Additional keyword arguments to pass to the ``Compose``
            constructor.

    Returns:
        Compose | None: A ``Compose`` instance, or None if the input value is None.

    Raises:
        TypeError: If the input value is not a valid type for building a
            ``Compose`` instance.
    """
    if value is None:
        return None
    elif isinstance(value, Compose):
        return value
    elif isinstance(value, list):
        return Compose.from_transforms(value, **kwargs)
    elif isinstance(value, dict):
        return Compose.from_config(value, **kwargs)
    else:
        raise TypeError(f"Unsupported compose type: {type(value).__name__}.")

# endregion
