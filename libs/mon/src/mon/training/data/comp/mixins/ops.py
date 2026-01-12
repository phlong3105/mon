#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Atomic operation mixins for data containers.

This module provides mixins for atomic operations.
"""

from __future__ import annotations

__all__ = [
    "BatchCollateMixin",
    "RegistrableMixin",
]

import abc
from typing import Any

import numpy as np
import torch

from mon.core import Task


# ==============================================================================
# region CREATION
# ==============================================================================

class RegistrableMixin(abc.ABC):
    """Metadata attribute mixin for data containers.

    Define common dataset attributes for categorization and factory registration.

    Attributes:
        _name (str | None): Name of the data container. Defaults to None.
        _tasks (list[mon.core.Task]): List of supported tasks. Defaults to [].
    """
    
    _name : str | None = None
    _tasks: list[Task] = []
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name : str | None        = None,
        tasks: list[Task] | None = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Name of the data container. If provided, it overrides the
                class-level default. Defaults to None.
            tasks: List of supported tasks. If provided, it overrides the
                class-level default. Defaults to None.
        """
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Expected 'name' to be a str, but got {type(name).__name__}.")
        if tasks is not None and not isinstance(tasks, list):
            raise TypeError(f"Expected 'tasks' to be a list, but got {type(tasks).__name__}.")
            
        # If provided, these instance variables will override the class-level defaults
        if name is not None:
            self._name = name
        if tasks is not None:
            # We use list() to create a copy, preventing shared state bugs
            self._tasks = list(tasks)
        
        # Continue the initialization chain
        super().__init__(*args, **kwargs)
        
    def __init_subclass__(cls, *args, **kwargs):
        """Validate subclass attributes on inheritance."""
        super().__init_subclass__(*args, **kwargs)
        
        # Check for EXPLICIT definition in the subclass (not inherited)
        for attr in ["_name", "_tasks"]:
            if attr not in cls.__dict__:
                raise TypeError(f"Class {cls.__name__} must define '{attr}' attribute.")
        
        # Check for VALID values
        if not isinstance(cls._name, str):
            raise TypeError(f"Expected '_name' to be a str, but got {type(cls._name).__name__}.")
        if not cls._name:
            raise ValueError(f"Expected '_name' to be a non-empty str, but got '{cls._name}'.")
            
        if not isinstance(cls._tasks, list):
            raise TypeError(f"Expected '_tasks' to be a list, but got {type(cls._tasks).__name__}.")
        if not cls._tasks:
            raise ValueError(f"Expected '_tasks' to be a non-empty list, but got {cls._tasks}.")
        
    # --- Properties ---
    @property
    def name(self) -> str:
        """Return the name of the data container."""
        return self._name
    
    @property
    def tasks(self) -> list[Task]:
        """Return the list of supported tasks."""
        return self._tasks

# endregion


# ==============================================================================
# region VALIDATION
# ==============================================================================


# endregion


# ==============================================================================
# region RETRIEVAL
# ==============================================================================

# --- Accessing ---


# --- Selection ---


# --- Aggregation ---


# endregion


# ==============================================================================
# region MUTATION
# ==============================================================================

# --- Alternation ---


# --- Rearrangement ---


# --- Addition ---


# --- Removal ---


# endregion


# ==============================================================================
# region COMPUTATION
# ==============================================================================

# --- Arithmetic ---


# --- Comparison ---


# --- Logical ---


# --- Geometric ---


# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---


# --- Standardization ---


# --- Structural ---

class BatchCollateMixin:
    """Batch collation mixin for data containers.

    Define the collate function for batching datapoints when using with a
    DataLoader.
    """
    
    # noinspection PyTypeChecker
    @staticmethod
    def collate_fn(batch: list[dict]) -> dict[str, Any]:
        """Collate a batch of input items.

        Args:
            batch: List of dictionaries, where each dictionary is a datapoint.

        Returns:
            Collated dictionary for torch.utils.data.dataset.DataLoader.
        """
        if not isinstance(batch, list):
            raise TypeError(f"Expected 'batch' to be a list, but got {type(batch).__name__}.")
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
                    f"Expected 'batch' item at index {i} to have keys {set(keys)}, "
                    f"but got {set(d.keys())}."
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
            elif isinstance(first_item, torch.Tensor):
                collated[k] = torch.stack(v, dim=0)
            elif isinstance(first_item, np.ndarray):
                collated[k] = np.stack(v, axis=0)
            # v remains a list for other types (like strings or custom objects)
        
        return collated
    

# --- Statistical ---


# --- Geometric ---


# endregion


# ==============================================================================
# region DESTRUCTION
# ==============================================================================


# endregion
