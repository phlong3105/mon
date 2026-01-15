#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Classes data structures.

This module provides base classes and mixins for classes.
"""

from __future__ import annotations

__all__ = [
    "Class",
    "ClassList",
    "Probabilities",
]

from typing import Any

import box
import numpy as np

from mon.core.console import log, rprint_list_dicts
from mon.core.fileio import load_yaml
from mon.core.pathlib import Path
from ..array import TensorOrArray


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---

Class = dict[str, Any]  # An alias for a dictionary of arbitrary key-value pairs.


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class ClassList(list[Class]):
    """Basic class for managing a list of classes.

    Extend the built-in ``list`` to handle a list of class dictionaries and
    provide properties and methods related to class management.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, data: list[dict] | Path | str = ()):
        """Initialize a new instance.

        Args:
            data: Either a list of class dictionaries or a Path to a YAML file
                defining the classes. Defaults to ().
        """
        # Validate and load classes
        if isinstance(data, (Path, str)):
            classes = load_yaml(path=data)
            classes = classes.get("classes", [])
        elif data in [None, ()]:
            classes = []
        else:
            classes = data

        # Continue the initialization chain
        super().__init__(classes)

    # --- Properties ---
    @property
    def trainable_classes(self) -> ClassList:
        """Return a ClassList of trainable class IDs (IDs in [0, 254])."""
        return ClassList([item for item in self if 0 <= item["id"] < 255])

    @property
    def keys(self) -> list[str]:
        """Return the keys in each class dictionary."""
        return list(self[0].keys()) if self else []

    @property
    def names(self) -> list[str]:
        """Return a list of class names."""
        return [item["name"] for item in self]

    @property
    def ids(self) -> list[int]:
        """Return a list of class IDs."""
        return [item["id"] for item in self]

    @property
    def id_to_class(self) -> dict[int, dict]:
        """Return a mapping from class IDs to class dictionaries."""
        return {item["id"]: item for item in self}

    @property
    def id_to_name(self) -> dict[int, str]:
        """Return a mapping from class IDs to class names."""
        return {item["id"]: item["name"] for item in self}

    @property
    def id_to_train_id(self) -> dict[int, int]:
        """Return a mapping from class IDs to train IDs."""
        return {
            item["id"]: item["train_id"]
            for item in self
            if "train_id" in item and 0 <= item["id"] < 255 and 0 <= item["train_id"] < 255
        }

    @property
    def id_color(self) -> dict[int, list[int] | tuple[int, int, int]]:
        """Return a mapping from class IDs to RGB colors."""
        return {item["id"]: item["color"] for item in self}

    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return len(self)

    @property
    def num_trainable_classes(self) -> int:
        """Return the number of trainable classes."""
        return len(self.trainable_classes)

    @property
    def palette(self) -> np.ndarray:
        """Return a palette for segmentation masks or drawing."""
        # Generates a (256, 3) array where index = ID
        palette = np.zeros((256, 3), dtype=np.uint8)
        for item in self:
            if 0 <= item["id"] < 256:
                palette[item["id"]] = item.get("color", [0, 0, 0])
        return palette

    # --- Access ---
    def get_by_id(self, class_id: int) -> dict | None:
        """Retrieve a class dictionary by ID safely.

        Args:
            class_id: Class ID.

        Returns:
            Class dictionary or None.
        """
        for item in self:
            if item.get("id") == class_id:
                return item
        return None

    def get_color(self, class_id: int, default: tuple = (255, 255, 255)) -> tuple:
        """Retrieve color for drawing.

        Args:
            class_id: Class ID.
            default: Default color. Defaults to (255, 255, 255).

        Returns:
            Color for drawing.
        """
        item = self.get_by_id(class_id)
        if item and "color" in item:
            return tuple(item["color"])
        return default

    # --- Utils ---
    def print(self):
        """Print class labels in a formatted table."""
        if not self:
            log("[yellow]No class is available.")
        else:
            log("Classes:")
            rprint_list_dicts(self)


class Probabilities(TensorOrArray):
    """Basic class for managing classification probabilities.

    Extend ``TensorOrArray`` to handle classification probabilities. Provide
    properties to access top-1 and top-5 class indices, and their confidence
    scores.

    Attributes:
        _data (numpy.ndarray): Probability vector of shape (``_num_classes``).
        _num_classes (int): Total number of classes.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, data: np.ndarray | int, num_classes: int = None):
        """Initialize a new instance.

        Args:
            data: Probability vector as a numpy.ndarray of shape
                (``num_classes``), or an integer representing the class ID.
            num_classes: Total number of classes. Defaults to None.

        Raises:
            ValueError: If ``num_classes`` is provided and is not a positive
                integer.
        """
        # Validate and set num_classes if provided
        if num_classes and num_classes <= 0:
            raise ValueError(f"Expected 'num_classes' to be a positive integer, but got {num_classes}.")

        self._num_classes = num_classes
        # Call the setter to ensure type validation on init
        self.data         = data

        # Continue the initialization chain
        super().__init__(data=self.data)

    # ---- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return the probability vector."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray | int):
        """Setter for the probability vector.

        Args:
            value: Probability vector as a numpy.ndarray of shape
                (``num_classes``), or an integer representing the class ID.

        Raises:
            ValueError: If ``value`` is an integer and ``num_classes`` is not
                provided or is invalid.
            TypeError: If ``value`` is not a numpy.ndarray or int.
        """
        if isinstance(value, int):
            if self.num_classes is None:
                raise ValueError(
                    f"Expected 'num_classes' to be provided when 'data' is an integer, but got None."
                )
            from .ops import class_id_to_one_hot
            value = class_id_to_one_hot(class_id=value, num_classes=self.num_classes)
        elif isinstance(value, np.ndarray):
            # Set num_classes if not already set
            if self.num_classes is None:
                self._num_classes = value.shape[0]
            # Validate data shape
            elif value.ndim != 1 or value.shape[0] != self.num_classes:
                raise ValueError(
                    f"Expected 'data' to be a numpy.ndarray of shape ({self.num_classes},), "
                    f"but got {value.shape}."
                )
        else:
            raise TypeError(
                f"Expected 'data' to be a numpy.ndarray or int, but got {type(value).__name__}."
            )

        self._data = value

    @property
    def num_classes(self) -> int:
        """Return the total number of classes."""
        return self._num_classes

    @property
    def top1_idx(self) -> int:
        """Return the index of the top-1 class."""
        # argmax is fast, but it's still an O(N) operation
        return int(np.argmax(self.data))

    @property
    def top5_idxes(self) -> list[int]:
        """Return the indices of the top-5 classes."""
        return list(np.argsort(self.data)[-5:][::-1])

    @property
    def top1(self) -> float:
        """Return the confidence score of the top-1 class."""
        # Instead of calling top1_idx (which calls argmax again),
        # use np.max for the value directly.
        return float(np.max(self.data))

    @property
    def top5(self) -> np.ndarray:
        """Return the confidence scores of the top-5 classes."""
        return self.data[self.top5_idxes]

# endregion
