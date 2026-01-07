#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class base classes and mixins.

This module provides the base classes and mixins for classes.
"""

__all__ = [
    "Class",
    "ClassList",
    "Probabilities",
]

from typing import Any

import numpy as np

from mon.core.console import log, rprint_list_dicts
from mon.core.pathlib import Path
from mon.core.runtime import load_config
from ..array import TensorOrArray


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---
Class = dict[str, Any]  # An alias for a dictionary of arbitrary key-value pairs.


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---


# --- Lifecycle Mixins ---


# --- Compute Mixins ---


# ==============================================================================
# CONCRETE IMPLEMENTATIONS (The Concrete Classes)
# ==============================================================================

# --- Primary Data Types ---
class ClassList(list[Class]):
    """A basic class for managing a list of classes.
    
    Extend the built-in list to handle a list of class dictionaries and provide
    properties and methods related to class management.
    
    Notes:
        - I choose the "List" suffix to indicate that this class will behave
          like a Python list.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, data: list[dict] | Path = ()):
        """Initialize a new instance.
        
        Args:
            data: Either a list of class dictionaries or a Path to a YAML file
                defining the classes. Defaults to ().
        """
        if isinstance(data, Path | str):
            classes = load_config(config=data, verbose=False)
            classes = classes.get("classes", [])
        elif data in [None, ()]:
            classes = []
        else:
            classes = data
        
        # Initialize parent classes and assign attributes
        super().__init__(classes)
    
    # --- Properties ---
    @property
    def trainable_classes(self) -> "ClassList":
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
    
    # --- Utils ---
    def print(self):
        """Print class labels in a formatted table."""
        if not self:
            log("[yellow]No class is available.")
        else:
            log("Classes:")
            rprint_list_dicts(self)


class Probabilities(TensorOrArray):
    """A basic class for managing classification probabilities.

    This class extends BaseTensorOrArray to handle classification probabilities.
    It provides properties to access top-1 and top-5 class indices, and their
    confidence scores.
    
    Attributes:
        _data (np.ndarray): Probability vector of shape (``_num_classes``).
        _num_classes (int): Total number of classes.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, data: np.ndarray | int, num_classes: int = None):
        """Initialize a new instance.
        
        Args:
            data: Probability vector as a numpy.ndarray of shape (``num_classes``),
                or an integer representing the class ID. If an integer is
                provided, it will be converted to a one-hot encoded vector.
            num_classes: Total number of classes. Required if ``data`` is
                provided as an integer. Defaults to None.
        
        Raises:
            ValueError: If ``num_classes`` is provided and is not a positive
                integer.
        """
        # Validate and set num_classes if provided
        if num_classes is not None and num_classes <= 0:
            raise ValueError(f"``num_classes`` must be a positive integer, got {num_classes}.")
        self._num_classes = num_classes
        
        super().__init__(data=data)  # This will call the data setter
        
    # ---- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return the probability vector."""
        return self._data
    
    @data.setter
    def data(self, data: np.ndarray | int):
        """Setter for the probability vector.
        
        Args:
            data: Probability vector as a numpy.ndarray of shape (``num_classes``),
                or an integer representing the class ID. If an integer is
                provided, it will be converted to a one-hot encoded vector.
        
        Raises:
            ValueError: If ``data`` is an integer and ``num_classes`` is not
                provided or is invalid.
            TypeError: If ``data`` is not a numpy.ndarray or int.
        """
        if isinstance(data, int):
            if self.num_classes is None:
                raise ValueError("``num_classes`` must be provided when ``data`` is an integer representing class ID.")
            from .ops import class_id_to_one_hot
            data = class_id_to_one_hot(class_id=data, num_classes=self.num_classes)
        elif isinstance(data, np.ndarray):
            # Set num_classes if not already set
            if self.num_classes is None:
                self._num_classes = data.shape[0]
            # Validate data shape
            elif data.ndim != 1 or data.shape[0] != self.num_classes:
                raise ValueError(f"``data`` must be a 1D array of shape ({self.num_classes}), got {data.shape}.")
        else:
            raise TypeError(f"``data`` must be a numpy.ndarray or int, got {type(data)}.")
        
        self._data = data
    
    @property
    def num_classes(self) -> int:
        """Return the total number of classes."""
        return self._num_classes
    
    @property
    def top1_idx(self) -> int:
        """Return the index of the class with the highest probability."""
        return int(np.argmax(self.data))
    
    @property
    def top5_idxes(self) -> list[int]:
        """Return the indices of the top-5 classes."""
        return list(np.argsort(self.data)[-5:][::-1])
    
    @property
    def top1(self) -> float:
        """Return the confidence score of the top-1 class."""
        return self.data[self.top1_idx]
    
    @property
    def top5(self) -> np.ndarray:
        """Return the confidence scores of the top-5 classes."""
        return self.data[self.top5_idxes]
