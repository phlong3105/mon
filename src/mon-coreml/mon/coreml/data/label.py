#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all labels in datasets."""

from __future__ import annotations

__all__ = [
    "ClassLabel", "ClassLabels", "Label", "majority_voting",
]

from abc import ABC, abstractmethod
from typing import Any, Sequence, TYPE_CHECKING

import cv2
import munch
import numpy as np
import torch

from mon import foundation
from mon.foundation import (
    console, error_console, file_handler, pathlib, rich,
)

if TYPE_CHECKING:
    from mon.coreml.typing import ClassLabelsType, PathType


# region Label

class Label(ABC):
    """The base class for all label classes. A label instance represent a
    logical collection of data associated with a particular task.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self.__setattr__(k, v)
    
    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor | None:
        """The label's data as a :class:`torch.Tensor`."""
        pass

# endregion


# region Class-Label

class ClassLabel(munch.Munch, Label):
    """A class-label represents a class pre-defined in a dataset. It consists of
    basic attributes such as: id, name, and color.
    """

    @property
    def tensor(self) -> torch.Tensor | None:
        """The class-label's data as a :class:`torch.Tensor`."""
        return None


class ClassLabels(list):
    """A list of all the class-labels defined in a dataset.
    
    Notes:
        We inherit the standard Python :class:`list` to take advantage of the
        built-in functionality.
    """
    
    def __init__(self, iterable: Sequence[ClassLabel]):
        """Creates a :class:`ClassLabels` object from a sequence. Each item in
        the list is a dictionary describing a :class:`ClassLabel` object.
        """
        assert isinstance(iterable, list | tuple) \
               and all(isinstance(i, ClassLabel) for i in iterable)
        super().__init__(i for i in iterable)

    def __setitem__(self, index, item):
        assert isinstance(item, ClassLabel)
        super().__setitem__(index, item)
    
    def insert(self, index, item):
        assert isinstance(item, ClassLabel)
        super().insert(index, item)

    def append(self, item):
        assert isinstance(item, ClassLabel)
        super().append(item)

    def extend(self, other):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(item for item in other)
    
    @classmethod
    def from_dict(cls, d: dict) -> ClassLabels:
        """Creates a :class:`ClassLabels` object from a dictionary :param:`d`.
        The dictionary must contain the key "classlabels" and it's corresponding
        value is a list of dictionary. Each item in the list
        :param:`d["classlabels"]` is a dictionary describing a
        :class:`ClassLabel` object.
        """
        assert isinstance(d, dict) and hasattr(d, "classlabels")
        l = d["classlabels"]
        assert isinstance(l, list | tuple)
        return cls(iterable=l)
        
    @classmethod
    def from_file(cls, path: PathType) -> ClassLabels:
        """Creates a :class:`ClassLabels` object from the content of a ".json"
        file specified by the :param:`path`.
        """
        path = pathlib.Path(path)
        assert path.is_json_file()
        return cls.from_dict(file_handler.load_from_file(path=path))
    
    @classmethod
    def from_value(cls, value: ClassLabelsType) -> ClassLabels | None:
        """Creates a :class:`ClassLabels` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, ClassLabels):
            return value
        if isinstance(value, dict | munch.Munch):
            return cls.from_dict(value)
        if isinstance(value, list | tuple):
            return cls(value)
        if isinstance(value, str | foundation.Path):
            return cls.from_file(value)
        error_console.log(
            f":param:`value` must be a :class:`ClassLabels`, :class:`dict`, "
            f":class:`munch.Munch`, :class:`str`, or "
            f":class:`mon.foundation.Path`. "
            f"But got: {type(value)}."
        )
        return None
    
    @property
    def classes(self) -> list[ClassLabel]:
        """An alias."""
        return self
    
    def color_legend(self, height: int | None = None) -> np.array:
        """Creates a legend figure of all the classlabels.
        
        Args:
            height: The height of the legend. If None, it will be
                25px * :meth:`__len__`.
        
        Returns:
            An RGB color legend figure.
        """
        num_classes = len(self)
        row_height  = 25 if (height is None) else int(height / num_classes)
        legend      = np.zeros(((num_classes * row_height) + 25, 300, 3), dtype=np.uint8)

        # Loop over the class names + colors
        for i, label in enumerate(self):
            color = label.color  # Draw the class name + color on the legend
            color = color[::-1]  # Convert to BGR format since OpenCV operates on BGR format.
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
        key                 : str  = "id",
        exclude_negative_key: bool = True,
    ) -> list:
        """Returns a list of colors corresponding to the items in :attr:`self`.
        
        Args:
            key: The key to search for. Defaults to "id".
            exclude_negative_key: If True, excludes the key with negative value.
                Defaults to True.
            
        Returns:
            A list of colors.
        """
        labels_colors = []
        for label in self:
            if hasattr(label, key) and hasattr(label, "color"):
                if exclude_negative_key and label[key] <  0:
                    continue
                labels_colors.append(label.color)
        return labels_colors

    @property
    def id2label(self) -> dict[int, dict]:
        """A dictionary mapping items' IDs (keys) to items (values)."""
        return {label["id"]: label for label in self}

    def ids(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
    ) -> list:
        """Returns a list of IDs corresponding to the items in :attr:`self`.
        
        Args:
            key: The key to search for. Defaults to "id".
            exclude_negative_key: If True, excludes the key with negative value.
                Defaults to True.
            
        Returns:
            A list of IDs.
        """
        ids = []
        for c in self:
            if hasattr(c, key):
                if exclude_negative_key and c[key] <  0:
                    continue
                ids.append(c[key])
        return ids
    
    @property
    def name2label(self) -> dict[str, dict]:
        """A dictionary mapping items' names (keys) to items (values)."""
        return {c["name"]: c for c in self.classes}

    def names(self, exclude_negative_key: bool = True) -> list:
        """Returns a list of names corresponding to the items in :attr:`self`.
        
        Args:
            exclude_negative_key: If True, excludes the key with negative value.
                Defaults to True.
            
        Returns:
            A list of IDs.
        """
        names = []
        for c in self:
            if hasattr(c, "id"):
                if exclude_negative_key and c["id"] <  0:
                    continue
                names.append(c["name"])
            else:
                names.append("")
        return names
    
    def num_classes(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
    ) -> int:
        """Counts the number of items.
        
        Args:
            key: The key to search for. Defaults to "id".
            exclude_negative_key: If True, excludes the key with negative value.
                Defaults to True.
            
        Returns:
            The number of items (classes) in the dataset.
        """
        count = 0
        for c in self:
            if hasattr(c, key):
                if exclude_negative_key and c[key] <  0:
                    continue
                count += 1
        return count

    def get_class(self, key: str = "id", value: Any = None) -> dict | None:
        """Returns the item (class-label) matching the given :param:`key` and
        :param:`value`.
        """
        for c in self:
            if hasattr(c, key) and (value == c[key]):
                return c
        return None
    
    def get_class_by_name(self, name: str) -> dict | None:
        """Returns the item (class-label) with the :param:`key` is "name" and
        value matching the given :param:`name`.
        """
        return self.get_class(key="name", value=name)

    def get_id(self, key: str = "id", value: Any = None) -> int | None:
        """Returns the ID of the item (class-label) matching the given
        :param:`key` and :param:`value`.
        """
        classlabel: dict = self.get_class(key=key, value=value)
        return classlabel["id"] if classlabel is not None else None
    
    def get_id_by_name(self, name: str) -> int | None:
        """Returns the name of the item (class-label) with the :param:`key` is
        "name" and value matching the given :param:`name`.
        """
        classlabel = self.get_class_by_name(name=name)
        return classlabel["id"] if classlabel is not None else None
    
    def get_name(self, key: str = "id", value: Any = None) -> str | None:
        """Returns the name of the item (class-label) with the :param:`key` is
        "name" and value matching the given :param:`name`.
        """
        c = self.get_class(key=key, value=value)
        return c["name"] if c is not None else None
    
    @property
    def tensor(self) -> torch.Tensor | None:
        return None
    
    def print(self):
        """Prints all items (class-labels) using in rich format."""
        if len(self) <= 0:
            console.log("[yellow]No class is available.")
            return
        console.log("Classlabels:")
        rich.print_table(self.classes)


# TODO: - Delete later
'''
class ClassLabels(list, Label):
    """A list of all the class-labels defined in a dataset.
    
    Attributes:
        classlabels: A list of all class-labels (classes) in the dataset.
    """

    def __init__(self, classlabels: list[ClassLabel], *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(classlabels, list) \
               and all(isinstance(c, ClassLabel) for c in classlabels)
        self.classlabels = classlabels

    @classmethod
    def from_list(cls, l: list[dict | munch.Munch]) -> ClassLabels:
        """Creates a :class:`ClassLabels` object from a list. Each item in the
        list is a dictionary describing a :class:`ClassLabel` object.
        """
        assert isinstance(l, list)
        if all(isinstance(i, dict) for i in l):
            l = [ClassLabel(**c) for c in l]
        return cls(classlabels=l)
    
    @classmethod
    def from_dict(cls, d: dict) -> ClassLabels:
        """Creates a :class:`ClassLabels` object from a dictionary :param:`d`.
        The dictionary must contain the key "classlabels" and it's corresponding
        value is a list of dictionary. Each item in the list
        :param:`d["classlabels"]` is a dictionary describing a
        :class:`ClassLabel` object.
        """
        assert isinstance(d, dict) and hasattr(d, "classlabels")
        return cls.from_list(d["classlabels"])
        
    @classmethod
    def from_file(cls, path: PathType) -> ClassLabels:
        """Creates a :class:`ClassLabels` object from the content of a ".json"
        file specified by the :param:`path`.
        """
        path = pathlib.Path(path)
        assert path.is_json_file()
        return cls.from_dict(file_handler.load_from_file(path))
    
    @classmethod
    def from_value(cls, value: ClassLabelsType) -> ClassLabels | None:
        """Creates a :class:`ClassLabels` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, ClassLabels):
            return value
        if isinstance(value, dict | munch.Munch):
            return cls.from_dict(value)
        if isinstance(value, list):
            return cls.from_list(value)
        if isinstance(value, PathType):
            return cls.from_file(value)
        error_console.log(
            f"`value` must be `ClassLabels`, `dict`, `str`, or "
            f"`Path`. But got: {type(value)}."
        )
        return None
        
    @property
    def classes(self) -> list[ClassLabel]:
        """An alias of :attr:`classlabels`."""
        return self.classlabels
    
    @property
    def list(self) -> list:
        """An alias of :meth:`classes`."""
        return self.classlabels

    def color_legend(self, height: int | None = None) -> np.array:
        """Creates a legend figure of all the classlabels.
        
        Args:
            height: The height of the legend. If None, it will be
                25px * len(:attr:`classlabels`).
        
        Returns:
            An RGB color legend figure.
        """
        num_classes = len(self.classes)
        row_height  = 25 if (height is None) else int(height / num_classes)
        legend      = np.zeros(((num_classes * row_height) + 25, 300, 3), dtype=np.uint8)

        # Loop over the class names + colors
        for i, label in enumerate(self.classes):
            color = label.color  # Draw the class name + color on the legend
            color = color[::-1]  # Convert to BGR format since OpenCV operates on BGR format.
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
        key                 : str  = "id",
        exclude_negative_key: bool = True,
    ) -> list:
        """Returns a list of colors corresponding to the items in
        :attr:`classlabels`.
        
        Args:
            key: The key to search for. Defaults to "id".
            exclude_negative_key: If True, excludes the key with negative value.
                Defaults to True.
            
        Returns:
            A list of colors.
        """
        labels_colors = []
        for label in self.classes:
            if hasattr(label, key) and hasattr(label, "color"):
                if exclude_negative_key and label[key] <  0:
                    continue
                labels_colors.append(label.color)
        return labels_colors

    @property
    def id2label(self) -> dict[int, dict]:
        """A dictionary mapping classlabels' IDs as keys and classlabels as
        values.
        """
        return {label["id"]: label for label in self.classes}

    def ids(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
    ) -> list:
        """Returns a list of IDs corresponding to the items in
        :attr:`classlabels`.
        
        Args:
            key: The key to search for. Defaults to "id".
            exclude_negative_key: If True, excludes the key with negative value.
                Defaults to True.
            
        Returns:
            A list of IDs.
        """
        ids = []
        for c in self.classes:
            if hasattr(c, key):
                if exclude_negative_key and c[key] <  0:
                    continue
                ids.append(c[key])
        return ids
    
    @property
    def name2label(self) -> dict[str, dict]:
        """A dictionary mapping classlabels' names as keys and classlabels as
        values.
        """
        return {c["name"]: c for c in self.classes}

    def names(self, exclude_negative_key: bool = True) -> list:
        """Returns a list of names corresponding to the items in
        :attr:`classlabels`.
        
        Args:
            exclude_negative_key: If True, excludes the key with negative value.
                Defaults to True.
            
        Returns:
            A list of IDs.
        """
        names = []
        for c in self.classes:
            if hasattr(c, "id"):
                if exclude_negative_key and c["id"] <  0:
                    continue
                names.append(c["name"])
            else:
                names.append("")
        return names
    
    def num_classes(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
    ) -> int:
        """Counts the number of classlabels in the dataset
        
        Args:
            key: The key to search for. Defaults to "id".
            exclude_negative_key: If True, excludes the key with negative value.
                Defaults to True.
            
        Returns:
            The number of classes in the dataset.
        """
        count = 0
        for c in self.classes:
            if hasattr(c, key):
                if exclude_negative_key and c[key] <  0:
                    continue
                count += 1
        return count

    def get_class(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> dict | None:
        """Returns the classlabel matching the given :param:`key` and
        :param:`value`.
        
        Args:
            key: The key to search for. Defaults to "id".
            value: The value of the :param:`key` to search for. Defaults to None.
        
        Returns:
            A dictionary of the classlabel that matches the :param:`key` and
            :param:`value`. Return None if such classlabel cannot be found.
        """
        for c in self.classes:
            if hasattr(c, key) and (value == c[key]):
                return c
        return None
    
    def get_class_by_name(self, name: str) -> dict | None:
        """Returns the classlabel matching the given :param:`name`.
        
        Args:
            name: The name of the classlabel you want to get.
        
        Returns:
            A dictionary of the classlabel matching the given :param:`name`.
            Return None if such classlabel cannot be found.
        """
        
        return self.get_class(key="name", value=name)
    
    def get_id(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> int | None:
        """Returns the id of the class label that matches the given key and
        value.
        
        Args:
           key: The key to search for. Defaults to "id".
           value: The value of the key to search for. Defaults to None.
        
        Returns:
            The id of the class.
        """
        classlabel: dict = self.get_class(key=key, value=value)
        return classlabel["id"] if classlabel is not None else None
    
    def get_id_by_name(self, name: str) -> int | None:
        """Given a class name, return the class id.
        
        Args:
            name: The name of the class you want to get the ID of.
        
        Returns:
            The id of the class.
        """
        classlabel = self.get_class_by_name(name=name)
        return classlabel["id"] if classlabel is not None else None
    
    def get_name(
        self,
        key  : str              = "id",
        value: int | str | None = None
    ) -> str | None:
        """Gets the name of a class given a key and value.
        
        Args:
            key: The key to search for. Defaults to "id".
            value: The value of the key to search for. Defaults to None.
        
        Returns:
            The name of the class.
        """
        c = self.get_class(key=key, value=value)
        return c["name"] if c is not None else None
       
    def show_color_legend(self, height: int | None = None):
        """Shows a pretty color lookup legend using OpenCV drawing functions.

        Args:
            height: Height of the color legend image. Defaults to None.
        """
        color_legend = self.color_legend(height=height)
        # plt.imshow(color_legend.permute(1, 2, 0))
        matplotlib.plt.imshow(color_legend)
        matplotlib.plt.title("Color Legend")
        matplotlib.plt.show()

    @property
    def tensor(self) -> torch.Tensor | None:
        """Returns the label in tensor format."""
        return None
    
    def print(self):
        """Prints all classes using `rich` format."""
        if not (self.classes and len(self.classes) > 0):
            console.log("[yellow]No class is available.")
            return
        
        console.log("Classlabels:")
        rich.print_table(self.classes)
'''


def majority_voting(labels: list[ClassLabel]) -> ClassLabel:
    """Counts the number of appearance of each class-label, and returns the
    label with the highest count.
    
    Args:
        labels: A list of :class:`ClassLabel`s.
    
    Returns:
        The :class:`ClassLabel` object that has the most votes.
    """
    # Count number of appearance of each label.
    unique_labels = munch.Munch()
    label_voting  = munch.Munch()
    for label in labels:
        k = label.get("id")
        v = label_voting.get(k)
        if v:
            label_voting[k] = v + 1
        else:
            unique_labels[k] = label
            label_voting[k]  = 1
    
    # Get k (label's id) with max v
    max_id = max(label_voting, key=label_voting.get)
    return unique_labels[max_id]

# endregion
