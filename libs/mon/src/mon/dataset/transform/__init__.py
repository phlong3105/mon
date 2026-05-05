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

from mon.core import is_list_of
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
        """Create a new instance from a configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary containing the
                necessary information to build the transformation.
            **kwargs: Additional keyword arguments to pass to the constructor.
        """
        # Extract relevant keys
        name = config.pop("name", config.pop("type", None))
        if name is None:
            raise KeyError(f"missing 'name' or 'type' key.")

        # Build the object
        config |= kwargs
        return ALBUMENTATIONS.build(name=name, **config)

    @classmethod
    def from_any(cls, value: Any, **kwargs) -> "BasicTransform":
        """Create a new instance from arbitrary input.

        Args:
            value (Any): The input value to create a transformation from.
                This can be a transformation instance, a configuration dictionary,
                or any other type that can be converted to a configuration dictionary.
            **kwargs: Additional keyword arguments to pass to the constructor.
        """
        if value is None:
            return NoOp()
        elif isinstance(value, cls):
            return value
        elif isinstance(value, dict):
            return cls.from_config(value, **kwargs)
        else:
            raise TypeError(f"unsupported transformation type {type(value).__name__}.")


class Compose(Compose_):
    """Extend ``albumentations.Compose`` with convenience methods for building
     transformation pipelines.
     """

    # --- Creation ---
    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs) -> "Compose":
        """Create a new instance from a configuration dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary containing the
                necessary information to build the transformation pipeline.
            **kwargs: Additional keyword arguments to pass to the constructor.
        """
        # Extract relevant keys
        transforms_: list[Any] = config.pop("transforms", config.pop("ops", []))

        # Validate inputs
        if not isinstance(transforms_, list):
            raise TypeError(
                f"expected a list of transformations, got {type(transforms_).__name__}."
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
        """Create a new instance from a list of transformations.

        Args:
            transforms (list[BasicTransform | dict[str, Any]]):
                A list of transformations, where each transformation is either
                a ``BasicTransform`` instance or a configuration dictionary to
                build a ``BasicTransform`` instance from.
            **kwargs: Additional keyword arguments to pass to the constructor.
        """
        # Normalize inputs
        transforms = [BasicTransform.from_any(t) for t in transforms]

        # Validate inputs
        if not is_list_of(transforms, BasicTransform):
            raise TypeError(f"expected a list of BasicTransform.")

        # Return the new instance
        return cls(transforms=transforms, **kwargs)

    @classmethod
    def from_any(cls, value: Any, **kwargs) -> "Compose":
        """Create a new instance from arbitrary input.

        Args:
            value (Any): The input value to create a transformation pipeline from.
                This can be a ``Compose`` instance, a list of transformations,
                a configuration dictionary, or any other type that can be converted
                to a list of transformations or a configuration dictionary.
            **kwargs: Additional keyword arguments to pass to the constructor.
        """
        if value is None:
            return cls(transforms=[])
        elif isinstance(value, cls):
            return value
        elif isinstance(value, list):
            return cls.from_transforms(value, **kwargs)
        elif isinstance(value, dict):
            return cls.from_config(value, **kwargs)
        else:
            raise TypeError(f"unsupported Compose type {type(value).__name__}.")

# endregion
