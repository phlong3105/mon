#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dataset Mixin.

This module defines mixins for datasets to extend their functionality.
"""

from __future__ import annotations

__all__ = [
    "DatasetCollationMixin",
    "DatasetRegisterMixin",
]

from abc import ABC
from typing import Any

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from mon.core import is_list_of, Task


# ==============================================================================
# region CREATION
# ==============================================================================

class DatasetRegisterMixin(ABC):
    """A mixin class for datasets that can be registered in a factory.

    Attributes:
        name (str, optional): Name of the dataset. `Must be defined in subclasses
            or set during initialization.`
        tasks (list[Task]): List of supported tasks. `Must be defined in
            subclasses or set during initialization.`
    """

    name: str = ""
    tasks: list[Task] = []

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str = "",
        tasks: list[Task] | None = None,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str, optional): Name of the data container. If provided, it
                overrides the class-level default. Defaults to None.
            tasks (list[Task], optional): List of supported tasks. If provided,
                it overrides the class-level default. Defaults to None.
            *args: Positional arguments to forward to the superclass constructor.
            **kwargs: Keyword arguments to forward to the superclass constructor.
        """
        # Assign attributes
        if isinstance(name, str):
            self.name = name
        if is_list_of(tasks, Task):
            # We use list() to create a copy, preventing shared state bugs
            self.tasks = list(tasks)

        # Continue the initialization chain
        super().__init__(*args, **kwargs)


# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---


# --- Standardization ---


# --- Structural ---

class DatasetCollationMixin:
    """Batch collation mixin for data containers.

    Define the collate function for batching datapoints when using with a
    DataLoader.
    """

    # noinspection PyTypeChecker
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        """Collate a batch of input items.

        Args:
            batch (list[dict]): List of datapoints to collate. Each datapoint
                is expected to be a dictionary with consistent keys.

        Returns:
            dict[str, Any]: Collated batch where each key maps to a batched
                value. Tensors and NumPy arrays are stacked, while other types
                are collected into lists. The "meta" key is converted to a
                tuple to ensure immutability.
        """
        if not isinstance(batch, list):
            raise TypeError(
                f"Expected 'batch' to be a list, but got {type(batch).__name__}."
            )
        if not batch:
            return {}

        # Faster Transposition: Group items by key
        # This replaces the complex zip(*[b.values()]) logic
        keys = batch[0].keys()
        # Validates batch item types and key sets
        for i, d in enumerate(batch):
            if not isinstance(d, dict):
                raise TypeError(
                    f"Expected 'batch' item at index {i} to be a dict, "
                    f"but got {type(d).__name__}."
                )
            if set(d.keys()) != set(keys):
                raise ValueError(
                    f"Expected 'batch' item at index {i} to have keys "
                    f"{set(keys)}, but got {set(d.keys())}."
                )

        collated = {k: [d[k] for d in batch] for k in keys}

        for k, v in collated.items():
            # Immutability for metadata
            if k == "meta":
                collated[k] = tuple(v)  # Freeze metadata to prevent runtime modification
                continue

            # Performance: O(1) Type Checking
            # We check only the first element, assuming batch homogeneity
            first_item = v[0]

            if first_item is None:
                collated[k] = None
            elif isinstance(first_item, Tensor):
                collated[k] = torch.stack(v, dim=0)
            elif isinstance(first_item, ndarray):
                collated[k] = np.stack(v, axis=0)
            # v remains a list for other types (like strings or custom objects)

        return collated


# --- Statistical ---


# --- Geometric ---

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
