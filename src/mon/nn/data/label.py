#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all labels in datasets."""

from __future__ import annotations

__all__ = [
    "ClassLabel", "ClassLabels", "Label", "majority_voting",
]

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
import torch

from mon.core import console, file, pathlib, rich


# region Label

class Label(ABC):
    """The base class for all label classes. A label instance represents a
    logical collection of data associated with a particular task.
    """
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)
    """
    
    @property
    @abstractmethod
    def data(self) -> list | None:
        """The label's data."""
        pass

    @property
    def nparray(self) -> np.ndarray | None:
        """The label's data as a :class:`numpy.ndarray`."""
        data = self.data
        if isinstance(data, list):
            data = np.array([i for i in data if isinstance(i, int | float)])
        return data
    
    @property
    def tensor(self) -> torch.Tensor | None:
        """The label's data as a :class:`torch.Tensor`."""
        data = self.data
        if isinstance(data, list):
            data = torch.Tensor([i for i in data if isinstance(i, int | float)])
        return data


# endregion


# region Class-Label

class ClassLabel(dict, Label):
    """A class-label represents a class pre-defined in a dataset. It consists of
    basic attributes such as ID, name, and color.
    """
    
    @classmethod
    def from_value(cls, value: ClassLabel | dict) -> ClassLabel:
        """Create a :class:`ClassLabels` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, dict):
            return ClassLabel(value)
        elif isinstance(value, ClassLabel):
            return value
        else:
            raise ValueError(
                f"value must be a ClassLabel class or a dict, but got "
                f"{type(value)}."
            )
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return None


class ClassLabels(list[ClassLabel]):
    """A list of all the class-labels defined in a dataset.
    
    Notes:
        We inherit the standard Python :class:`list` to take advantage of the
        built-in functions.
    """
    
    def __init__(self, seq: list[ClassLabel | dict]):
        super().__init__(ClassLabel.from_value(value=i) for i in seq)
    
    def __setitem__(self, index: int, item: ClassLabel | dict):
        super().__setitem__(index, ClassLabel.from_value(item))
    
    def insert(self, index: int, item: ClassLabel | dict):
        super().insert(index, ClassLabel.from_value(item))
    
    def append(self, item: ClassLabel | dict):
        super().append(ClassLabel.from_value(item))
    
    def extend(self, other: list[ClassLabel | dict]):
        super().extend([ClassLabel.from_value(item) for item in other])
    
    @classmethod
    def from_dict(cls, value: dict) -> ClassLabels:
        """Create a :class:`ClassLabels` object from a dictionary :param:`d`.
        The dictionary must contain the key 'classlabels', and it's
        corresponding value is a list of dictionary. Each item in the list
        :param:`d["classlabels"]` is a dictionary describing a
        :class:`ClassLabel` object.
        """
        if "classlabels" not in value:
            raise ValueError("value must contains a 'classlabels' key.")
        classlabels = value["classlabels"]
        if not isinstance(classlabels, list | tuple):
            raise TypeError(
                f"classlabels must be a list or tuple, but got "
                f"{type(classlabels)}."
            )
        return cls(seq=classlabels)
    
    @classmethod
    def from_file(cls, path: pathlib.Path) -> ClassLabels:
        """Create a :class:`ClassLabels` object from the content of a '.json'
        file specified by the :param:`path`.
        """
        path = pathlib.Path(path)
        if not path.is_json_file():
            raise ValueError(f"path must be a .json file, but got {path}.")
        return cls.from_dict(file.read_from_file(path=path))
    
    @classmethod
    def from_value(cls, value: Any) -> ClassLabels | None:
        """Create a :class:`ClassLabels` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, ClassLabels):
            return value
        if isinstance(value, dict):
            return cls.from_dict(value)
        if isinstance(value, list | tuple):
            return cls(value)
        if isinstance(value, str | pathlib.Path):
            return cls.from_file(value)
        return None
    
    @property
    def classes(self) -> list[ClassLabel]:
        """An alias."""
        return self
    
    def color_legend(self, height: int | None = None) -> np.array:
        """Create a legend figure of all the classlabels.
        
        Args:
            height: The height of the legend. If None, it will be
                25px * :meth:`__len__`.
        
        Return:
            An RGB color legend figure.
        """
        num_classes = len(self)
        row_height = 25 if (height is None) else int(height / num_classes)
        legend = np.zeros(
            ((num_classes * row_height) + 25, 300, 3),
            dtype=np.uint8
        )
        
        # Loop over the class names + colors
        for i, label in enumerate(self):
            color = label.color  # Draw the class name + color on the legend
            color = color[
                    ::-1]  # Convert to BGR format since OpenCV operates on
            # BGR format.
            cv2.putText(
                img       = legend,
                text      = label.name,
                org       = (5, (i * row_height) + 17),
                fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.5,
                color     = (0, 0, 255),
                thickness = 2
            )
            cv2.rectangle(
                img       = legend,
                pt1       = (150, (i * 25)),
                pt2       = (300, (i * row_height) + 25),
                color     = color,
                thickness = -1
            )
        return legend
    
    def colors(
        self,
        key: str = "id",
        exclude_negative_key: bool = True,
    ) -> list:
        """Return a list of colors corresponding to the items in :attr:`self`.
        
        Args:
            key: The key to search for. Default: 'id'.
            exclude_negative_key: If `True`, excludes the key with negative
                value. Default: `True`.
            
        Return:
            A list of colors.
        """
        colors = []
        for c in self:
            key_value = c.get(key, None)
            if (key_value is None) or (exclude_negative_key and key_value < 0):
                continue
            color = c.get("color", [255, 255, 255])
            colors.append(color)
        return colors
    
    @property
    def id2label(self) -> dict[int, dict]:
        """A dictionary mapping items' IDs (keys) to items (values)."""
        return {label["id"]: label for label in self}
    
    def ids(
        self,
        key: str = "id",
        exclude_negative_key: bool = True,
    ) -> list:
        """Return a list of IDs corresponding to the items in :attr:`self`.
        
        Args:
            key: The key to search for. Default: 'id'.
            exclude_negative_key: If True, excludes the key with negative value.
                Default: True.
            
        Return:
            A list of IDs.
        """
        ids = []
        for c in self:
            key_value = c.get(key, None)
            if (id is None) or (exclude_negative_key and key_value < 0):
                continue
            ids.append(key_value)
        return ids
    
    @property
    def name2label(self) -> dict[str, dict]:
        """A dictionary mapping items' names (keys) to items (values)."""
        return {c["name"]: c for c in self.classes}
    
    def names(self, exclude_negative_key: bool = True) -> list:
        """Return a list of names corresponding to the items in :attr:`self`.
        
        Args:
            exclude_negative_key: If True, excludes the key with negative value.
                Default: True.
            
        Return:
            A list of IDs.
        """
        names = []
        for c in self:
            key_value = c.get("id", None)
            if (key_value is None) or (exclude_negative_key and key_value < 0):
                continue
            name = c.get("name", "")
            names.append(name)
        return names
    
    def num_classes(
        self,
        key: str = "id",
        exclude_negative_key: bool = True,
    ) -> int:
        """Counts the number of items.
        
        Args:
            key: The key to search for. Default: 'id'.
            exclude_negative_key: If True, excludes the key with negative value.
                Default: True.
            
        Return:
            The number of items (classes) in the dataset.
        """
        count = 0
        for c in self:
            key_value = c.get(key, None)
            if (key_value is None) or (exclude_negative_key and key_value < 0):
                continue
            count += 1
        return count
    
    def get_class(self, key: str = "id", value: Any = None) -> dict | None:
        """Return the item (class-label) matching the given :param:`key` and
        :param:`value`.
        """
        for c in self:
            key_value = c.get(key, None)
            if (key_value is not None) and (value == key_value):
                return c
        return None
    
    def get_class_by_name(self, name: str) -> dict | None:
        """Return the item (class-label) with the :param:`key` is 'name' and
        value matching the given :param:`name`.
        """
        return self.get_class(key="name", value=name)
    
    def get_id(self, key: str = "id", value: Any = None) -> int | None:
        """Return the ID of the item (class-label) matching the given
        :param:`key` and :param:`value`.
        """
        classlabel: dict = self.get_class(key=key, value=value)
        return classlabel["id"] if classlabel is not None else None
    
    def get_id_by_name(self, name: str) -> int | None:
        """Return the name of the item (class-label) with the :param:`key` is
        'name' and value matching the given :param:`name`.
        """
        classlabel = self.get_class_by_name(name=name)
        return classlabel["id"] if classlabel is not None else None
    
    def get_name(self, key: str = "id", value: Any = None) -> str | None:
        """Return the name of the item (class-label) with the :param:`key` is
        'name' and value matching the given :param:`name`.
        """
        c = self.get_class(key=key, value=value)
        return c["name"] if c is not None else None
    
    @property
    def tensor(self) -> torch.Tensor | None:
        return None
    
    def print(self):
        """Print all items (class-labels) in rich format."""
        if len(self) <= 0:
            console.log("[yellow]No class is available.")
            return
        console.log("Classlabels:")
        rich.print_table(self.classes)


def majority_voting(labels: list[ClassLabel]) -> ClassLabel:
    """Counts the number of appearances of each class-label, and returns the
    label with the highest count.
    
    Args:
        labels: A list of :class:`ClassLabel`s.
    
    Return:
        The :class:`ClassLabel` object that has the most votes.
    """
    # Count number of appearances of each label.
    unique_labels = {}
    label_voting  = {}
    for label in labels:
        k = label.get("id")
        v = label_voting.get(k)
        if v:
            label_voting[k]  = v + 1
        else:
            unique_labels[k] = label
            label_voting[k]  = 1
    
    # Get k (label's id) with max v
    max_id = max(label_voting, key=label_voting.get)
    return unique_labels[max_id]

# endregion
