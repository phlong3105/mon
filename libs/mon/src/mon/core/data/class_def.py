#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class Definition Data Structures.

This module provides data structures and utilities for handling class
definitions in datasets.
"""

from __future__ import annotations

__all__ = [
    "Class",
    "ClassList",
    "build_classlist",
]

from dataclasses import dataclass
from typing import Any, Iterable

from mon.core.base import IndexList
from mon.core.fileio import load_yaml, save_yaml
from mon.core.path import Path
from mon.core.typing import Int3


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class Class:
    """Data structure for handling a single class definition.

    Attributes:
        name (str): Name of the class.
        id (int): Unique identifier for the class.
        train_id (int, optional): Training identifier for the class. Defaults to -1.
        category (str, optional): Category of the class. Defaults to "unknown".
        category_id (int, optional): Identifier for the category. Defaults to -1.
        color (Int3, optional): RGB color representation for visualization.
            Defaults to (255, 255, 255).
        ignore_in_eval (bool, optional): Flag indicating if the class should be
            ignored during evaluation. Defaults to False.
    """

    name: str
    id: int
    train_id: int = -1
    category: str = "unknown"
    category_id: int = -1
    color: Int3 = (255, 255, 255)
    ignore_in_eval: bool = False

    # --- Properties ---
    @property
    def id_color(self) -> Int3:
        """Return a tuple of the class ID repeated three times for color image."""
        return (self.id,) * 3

    @property
    def is_trainable(self) -> bool:
        """Return whether the class is trainable (ID in [0, 254])."""
        return 0 <= self.train_id < 255


class ClassList(IndexList[Class]):
    """A list of ``Class`` instances, accessible by index or name.

    Extend ``IndexList`` to provide dictionary-like access to ``Class``
    instances by their ``name`` attribute.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, data: Iterable | None = None, *args, **kwargs):
        """Initialize a new instance.

        Args:
            data (Iterable | None, optional): Initial data to populate the list.
                Defaults to None.
        """
        # We hardcode the item_type, key, and id here
        # Users just call ClassList() without arguments
        super().__init__(item_type=Class, data=data, key="name", id="id")

    # --- Properties ---
    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self.__len__()

    @property
    def num_trainable_classes(self) -> int:
        """Return the number of trainable classes (IDs in [0, 254])."""
        return sum(1 for cls in self.data if cls.is_trainable)

    @property
    def names(self) -> list[str]:
        """Return a list of class names."""
        return [item.name for item in self.data]

    @property
    def ids(self) -> list[int]:
        """Return a list of class IDs."""
        return list(self._id_map.keys())

    @property
    def palette(self) -> list[Int3]:
        """Return the color palette as a list of RGB tuples."""
        return [item.color for item in self.data]

    # --- Input ---
    @classmethod
    def load(cls, path: Path, *args, **kwargs) -> "ClassList":
        """Load class definitions from a YAML file.

        Args:
            path (Path): Path to the input YAML file.

        Returns:
            ClassList: A new instance of ``ClassList`` populated with the
                loaded class definitions.
        """
        path = Path(path).normalize()
        if not path.has_ext(".yaml", ".yml", exists=True):
            raise ValueError(f"YAML file not found at {path.as_posix()}")

        classes: dict = load_yaml(path=path)
        classes = classes.get("classes", [])
        return cls([Class(**c) for c in classes], *args, **kwargs)

    # --- Output ---
    def save(self, path: Path):
        """Save class definitions to a YAML file.

        Args:
            path (Path): Path to the output YAML file.

        Raises:
            ValueError: If ``path`` is not a YAML file.
        """
        path: Path = Path(path).normalize()
        if not path.has_ext(".yaml", ".yml", exists=False):
            raise ValueError(f"expected a valid YAML file, got {path.as_posix()}")

        classes_dict = {"classes": [item.__dict__ for item in self.data]}
        save_yaml(data=classes_dict, path=path)

    # --- Creation ---
    @classmethod
    def from_file(cls, path: Path) -> "ClassList":
        """Create a new instance from a YAML file."""
        return cls.load(path)

# endregion


# ==============================================================================
# region CREATION
# ==============================================================================

def build_classlist(value: Any) -> ClassList:
    """Build a ``ClassList`` instance from a given value.

    Args:
        value (Any): Either a ``ClassList`` instance, a list of class definitions,
            or a path to a YAML file containing class definitions.

    Returns:
        ClassList: A ``ClassList`` instance.

    Raises:
        TypeError: If the input value is not a valid type for building a ``ClassList``.
    """
    if value is None:
        return ClassList()
    elif isinstance(value, ClassList):
        return value
    elif isinstance(value, list):
        return ClassList(value)
    elif isinstance(value, (Path, str)):
        return ClassList.from_file(value)
    else:
        raise TypeError(f"unsupported ClassList type {type(value).__name__}.")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
