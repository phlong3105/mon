#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Class Definition Data Structure.

This module provides data structures for handling class definitions in datasets.
"""

from __future__ import annotations

__all__ = [
    "Class",
    "ClassList",
    "ClassListLike",
    "build_classlist",
]

from dataclasses import dataclass
from typing import Any, Iterable, TypeAlias, Union

from mon.core.data.structs import IndexList
from mon.core.fileio import load_yaml, save_yaml
from mon.core.path import Path
from mon.core.typing import PathLike
from mon.core.utils import is_valid_str


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
        color (tuple[int, int, int], optional): RGB color representation for
            visualization. Defaults to (255, 255, 255).
        ignore_in_eval (bool, optional): Flag indicating if the class should be
            ignored during evaluation. Defaults to False.
    """

    name: str
    id: int
    train_id: int = -1
    category: str = "unknown"
    category_id: int = -1
    color: tuple[int, int, int] = (255, 255, 255)
    ignore_in_eval: bool = False

    # --- Properties ---
    @property
    def id_color(self) -> tuple[int, int, int]:
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
            data (Iterable, optional): Initial data to populate the list.
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
        return self.keys()

    @property
    def ids(self) -> list[int]:
        """Return a list of class IDs."""
        return list(self._id_map.keys())

    @property
    def palette(self) -> list[tuple[int, int, int]]:
        """Return the color palette as a list of RGB tuples."""
        return [item.color for item in self.data]

    # --- Input ---
    @classmethod
    def load(cls, path: PathLike, *args, **kwargs) -> "ClassList":
        """Load class definitions from a YAML file.

        Args:
            path (PathLike): Path to the input YAML file.

        Returns:
            ClassList: A new instance of ``ClassList`` populated with the
                loaded class definitions.
        """
        if is_valid_str(path):
            path = Path(path).normalize()
        if not path.has_ext(".yaml", ".yml", exist=True):
            raise ValueError(f"YAML file not found at '{path}'.")

        classes = load_yaml(path=path)
        classes = classes.get("classes", [])
        return cls([Class(**c) for c in classes], *args, **kwargs)

    # --- Output ---
    def save(self, path: PathLike):
        """Save class definitions to a YAML file.

        Args:
            path (PathLike): Path to the output YAML file.

        Raises:
            ValueError: If ``path`` is not a YAML file.
        """
        if is_valid_str(path):
            path = Path(path).normalize()
        if not path.has_ext(".yaml", ".yml", exist=False):
            raise ValueError(f"Expected a valid YAML file, but got '{path}'.")

        classes_dict = {
            "classes": [item.__dict__ for item in self.data]
        }
        save_yaml(data=classes_dict, path=path)

    # --- Creation ---
    @classmethod
    def from_file(cls, path: PathLike) -> "ClassList":
        """Create a new instance from a YAML file."""
        return cls.load(path)

# endregion


# ==============================================================================
# region TYPE DEFINITIONS
# ==============================================================================

ClassListLike: TypeAlias = Union[
    ClassList,
    list[Union[Class, dict[str, Any]]],
    dict[str, Union[Class, dict[str, Any]]],
    PathLike,
]

# endregion


# ==============================================================================
# region REGISTRY & FACTORY
# ==============================================================================

def build_classlist(classlist: ClassListLike | None, *args, **kwargs) -> ClassList:
    """Build a ``Classes`` instance from the given input.

    Args:
        classlist (ClassListLike): Either a ``ClassList`` instance, a list of
            ``Class`` instances or dictionaries, a dictionary mapping class names
            to ``Class`` instances or dictionaries, or a path to a YAML file
            containing class definitions.
        *args: Positional arguments for ``ClassList`` constructor.
        **kwargs: Keyword arguments for ``ClassList`` constructor.

    Returns:
        ClassList: A ``ClassList`` instance.

    Raises:
        TypeError: If the input type is unsupported.
    """
    if classlist is None:
        return ClassList(*args, **kwargs)
    elif isinstance(classlist, ClassList):
        return classlist
    elif isinstance(classlist, (list, dict)):
        # Convert dict to list
        if isinstance(classlist, dict):
            classlist = list(classlist.values())
        for i, c in enumerate(classlist):
            # Convert dict to Class instance
            classlist[i] = Class(**c) if isinstance(c, dict) else c
        return ClassList(classlist, *args, **kwargs)
    elif isinstance(classlist, (Path, str)):
        return ClassList.load(path=classlist, *args, **kwargs)
    else:
        raise TypeError(f"Unsupported type: {type(classlist).__name__}")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
