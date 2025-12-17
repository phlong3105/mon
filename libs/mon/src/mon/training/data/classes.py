#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for class labels.

This module provides the Classes class, which manages a list of class labels
defined in a dataset. It supports loading from a YAML file, accessing class
properties, and filtering trainable classes.
"""

__all__ = [
    "Classes",
]

from mon.core import log, Path, rprint_list_dicts, load_config


class Classes(list[dict]):
    """A class for managing class labels.
    
    This class extends the built-in list to handle a list of class label
    dictionaries. It supports loading class definitions from a YAML file,
    accessing various properties, and filtering trainable classes.
    """
    
    def __init__(self, seq: list[dict] | Path = ()):
        """Initializes the Classes instance.
        
        Args:
            seq (list[dict] or Path, optional): Either a list of class label
                dictionaries, or a Path to a YAML file defining the classes.
                Defaults to an empty tuple. Defaults to ().
        """
        if isinstance(seq, Path | str):
            classes = load_config(config=seq, verbose=False)
            classes = classes.get("classes", [])
        elif seq in [None, ()]:
            classes = []
        else:
            classes = seq
            
        super().__init__(classes)
    
    # ----- Properties -----
    @property
    def trainable_classes(self) -> "Classes":
        """Getter for trainable classes (IDs in [0, 254]).
        
        Returns:
            A Classes instance containing only trainable class labels.
        """
        return Classes([item for item in self if 0 <= item["id"] < 255])
    
    @property
    def keys(self) -> list[str]:
        """Getter for keys in the class labels.
        
        Returns:
            list[str]: List of keys from class label dictionaries.
        """
        return list(self[0].keys()) if self else []
    
    @property
    def names(self) -> list[str]:
        """Getter for class names.
        
        Returns:
            list[str]: List of ``name`` values from class labels.
        """
        return [item["name"] for item in self]
    
    @property
    def ids(self) -> list[int]:
        """Getter for class IDs.
        
        Returns:
            list[int]: List of ``id`` values from class labels.
        """
        return [item["id"] for item in self]
    
    @property
    def id_to_class(self) -> dict[int, dict]:
        """Getter for mapping IDs to class label dictionaries.
        
        Returns:
            dict[int, dict]: Dict mapping ``id`` to the entire class label
                dictionary.
        """
        return {item["id"]: item for item in self}
    
    @property
    def id_to_name(self) -> dict[int, str]:
        """Getter for mapping IDs to class names.
        
        Returns:
            dict[int, str]: Dict mapping ``id`` to ``name``.
        """
        return {item["id"]: item["name"] for item in self}
    
    @property
    def id_to_train_id(self) -> dict[int, int]:
        """Getter for mapping IDs to train IDs.
        
        Returns:
            dict[int, int]: Dict mapping ``id`` to ``train_id``.
        """
        return {
            item["id"]: item["train_id"]
            for item in self
            if "train_id" in item and 0 <= item["id"] < 255 and 0 <= item["train_id"] < 255
        }
    
    @property
    def id_color(self) -> dict[int, list[int] | tuple[int, int, int]]:
        """Getter for mapping IDs to colors.
        
        Returns:
            dict[int, list[int] | tuple[int, int, int]]: Dict mapping ``id`` to
                ``color``.
        """
        return {item["id"]: item["color"] for item in self}
    
    @property
    def num_classes(self) -> int:
        """Getter for the number of classes.
        
        Returns:
            int: Number of classes.
        """
        return len(self)
    
    @property
    def num_trainable_classes(self) -> int:
        """Getter for the number of trainable classes.
        
        Returns:
            int: Number of trainable classes.
        """
        return len(self.trainable_classes)
    
    # ----- Utils -----
    def print(self):
        """Prints class labels in a formatted table."""
        if not self:
            log("[yellow]No class is available.")
        else:
            log("Classes:")
            rprint_list_dicts(self)
