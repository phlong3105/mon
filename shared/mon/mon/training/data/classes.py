#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module defines the ``Classes`` class, which represents a list of class-labels
defined in a dataset.

It provides properties and methods to access various aspects of the class labels,
such as trainable classes, names, IDs, and color mappings.
"""

__all__ = [
    "Classes",
]

from mon.core import log, rprint_list_dicts


class Classes(list[dict]):
    """List of class labels defined in a dataset.
    
    Notes:
        Inherits from Python ``list`` for built-in functionality.
    """
    
    @property
    def trainable_classes(self) -> "Classes":
        """Returns all trainable classes.
        
        Returns:
            New ``Classes`` with classes where ``id`` is in [0, 254].
        """
        return Classes([item for item in self if 0 <= item["id"] < 255])
    
    @property
    def keys(self) -> list[str]:
        """Returns all keys in the class labels.

        Returns:
            List of keys from the first class label.
        """
        return list(self[0].keys()) if self else []
    
    @property
    def names(self) -> list[str]:
        """Returns all names in the class labels.

        Returns:
            List of ``name`` values from class labels.
        """
        return [item["name"] for item in self]
    
    @property
    def ids(self) -> list[int]:
        """Returns all IDs in the class labels.

        Returns:
            List of ``id`` values from class labels.
        """
        return [item["id"] for item in self]
    
    @property
    def id_to_class(self) -> dict[int, dict]:
        """Maps IDs to class label dictionaries.

        Returns:
            Dict mapping ``id`` to class label items.
        """
        return {item["id"]: item for item in self}
    
    @property
    def id_to_name(self) -> dict[int, str]:
        """Maps IDs to class names.

        Returns:
            Dict mapping ``id`` to ``name``.
        """
        return {item["id"]: item["name"] for item in self}
    
    @property
    def id_to_train_id(self) -> dict[int, int]:
        """Maps IDs to trainable IDs.

        Returns:
            Dict mapping ``id`` to ``train_id`` for IDs in [0, 254].
        """
        return {
            item["id"]: item["train_id"]
            for item in self
            if "train_id" in item and 0 <= item["id"] < 255 and 0 <= item["train_id"] < 255
        }
    
    @property
    def id_color(self) -> dict[int, list[int] | tuple[int, int, int]]:
        """Maps IDs to RGB colors.

        Returns:
            Dict mapping ``id`` to ``color`` values.
        """
        return {item["id"]: item["color"] for item in self}
    
    @property
    def num_classes(self) -> int:
        """Returns the total number of classes.

        Returns:
            Integer count of class labels.
        """
        return len(self)
    
    @property
    def num_trainable_classes(self) -> int:
        """Returns the number of trainable classes.

        Returns:
            Integer count of trainable class labels.
        """
        return len(self.trainable_classes)
    
    def print(self):
        """Prints class labels in a formatted table."""
        if not self:
            log("[yellow]No class is available.")
        else:
            log("Classes:")
            rprint_list_dicts(self)
