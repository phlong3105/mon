#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Operations for class-label. A class-label is a dictionary of label's
properties defined in the dataset.
"""

from __future__ import annotations

from typing import Optional
from typing import Union

import cv2
import numpy as np
from munch import Munch

from one.core import console
from one.core import print_table
from one.io import load

__all__ = [
    "ClassLabels",
    "majority_voting",
]


# MARK: - Functional

def majority_voting(object_labels: list[dict]) -> dict:
    """Get label that has max appearances in the object's labels list."""
    # NOTE: Count number of appearance of each label.
    unique_labels = Munch()
    label_voting  = Munch()
    for label in object_labels:
        key   = label.get("id")
        value = label_voting.get(key)
        if value:
            label_voting[key] = value + 1
        else:
            unique_labels[key] = label
            label_voting[key]  = 1
    
    # NOTE: get key (label's id) with max value
    max_id = max(label_voting, key=label_voting.get)
    return unique_labels[max_id]


# MARK: - Modules

class ClassLabels:
    """ClassLabels object is a wrapper around a list of label dictionaries.
    It takes care of all the hassle when working with labels.
    
    Based on:
    https://www.tensorflow.org/datasets/api_docs/python/tfds/features/ClassLabel

    Attributes:
        class_labels (list):
            List of all class_labels.
    """

    # MARK: Magic Functions

    def __init__(self, class_labels: list):
        self._class_labels = class_labels

    # MARK: Configure

    @staticmethod
    def create_from_dict(label_dict: dict) -> ClassLabels:
        """Create a `ClassLabels` object from a dictionary that contains all
        class_labels.
        """
        if hasattr(label_dict, "class_labels"):
            class_labels = label_dict.get("class_labels")
            class_labels = Munch.fromDict(class_labels)
            return ClassLabels(class_labels=class_labels)
        else:
            raise ValueError(f"`label_dict` must contain key `class_labels`. "
                             f"Cannot defined labels!")
    
    @staticmethod
    def create_from_file(label_path: str) -> ClassLabels:
        """Create a `ClassLabels` object from a file that contains all
        class_labels.
        """
        labels_dict = load(path=label_path)
        class_labels = labels_dict["class_labels"]
        class_labels = Munch.fromDict(class_labels)
        return ClassLabels(class_labels=class_labels)
        
    # MARK: Property

    @property
    def class_labels(self) -> list:
        """Return the list of all labels' dictionaries."""
        return self._class_labels

    def color_legend(self, height: Optional[int] = None) -> np.ndarray:
        """Return a color legend using OpenCV drawing functions.

		References:
			https://www.pyimagesearch.com/2018/09/03/semantic-segmentation-with-opencv-and-deep-learning/

		Args:
			height (int, optional):
				Height of the color legend image. Defaults: `None`.

		Returns:
			legend (np.ndarray):
				Color legend image.
		"""
        num_classes = len(self.class_labels)
        row_height  = 25 if (height is None) else int(height / num_classes)
        legend      = np.zeros(
            ((num_classes * row_height) + 25, 300, 3), dtype=np.uint8
        )

        # NOTE: Loop over the class names + colors
        for i, label in enumerate(self.class_labels):
            # Draw the class name + color on the legend
            color = label.color
            # Convert to BGR format since OpenCV operates on BGR format.
            color = color[::-1]
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
        exclude_max_key     : bool = True
    ) -> list:
        """Return the list of all labels' colors.
        
        Args:
            key (str):
                Label's key to search from `labels`
            exclude_negative_key (bool):
                If `True` only count class's label with key >= 0.
            exclude_max_key (bool):
			    If `True` only count class's label with key < 255.
        """
        labels_colors = []
        for label in self.class_labels:
            if hasattr(label, key) and hasattr(label, "color"):
                if (exclude_negative_key and label[key] < 0  ) or \
                   (exclude_max_key      and label[key] >= 255):
                    continue
                labels_colors.append(label.color)

        return labels_colors

    @property
    def id2label(self) -> dict[int, dict]:
        """Return a dictionary of id to label object."""
        return {label["id"]: label for label in self.class_labels}

    def ids(
        self,
        key                 : str = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> list:
        """Return the list of all labels' ids at `key`.
        
        Args:
            key (str):
                Label's key to search from `labels`.
            exclude_negative_key (bool):
                If `True` only count class's label with key >= 0.
            exclude_max_key (bool):
                If `True` only count class's label with key < 255.
        """
        labels_ids = []
        for label in self.class_labels:
            if hasattr(label, key):
                if (exclude_negative_key and label[key] < 0   ) or \
                   (exclude_max_key      and label[key] >= 255):
                    continue
                labels_ids.append(label[key])

        return labels_ids

    @property
    def list(self) -> list:
        """Alias to `class_labels()`."""
        return self.class_labels

    @property
    def name2label(self) -> dict[str, dict]:
        """Return a dictionary of {`name`: `label object`}."""
        return {label["name"]: label for label in self.class_labels}

    def names(self, exclude_negative_key: bool = True, exclude_max_key: bool = True) -> list:
        """Return the list of all class_labels' names.
        
        Args:
            exclude_negative_key (bool):
                If `True` only count class's label with key >= 0.
            exclude_max_key (bool):
                If `True` only count class's label with key < 255.
        """
        names = []
        for label in self.class_labels:
            if hasattr(label, "id"):
                if (exclude_negative_key and label["id"] < 0   ) or \
                   (exclude_max_key      and label["id"] >= 255):
                    continue
                names.append(label["name"])
            else:
                names.append("")
        return names
    
    def num_classes(
        self,
        key                 : str  = "id",
        exclude_negative_key: bool = True,
        exclude_max_key     : bool = True
    ) -> int:
        """Return the number of classes.

        Args:
            key (str):
                Label's key to search from `labels`. Defaults: `id`.
            exclude_negative_key (bool):
                If `True` only count class's label with key >= 0. Defaults: `True`.
            exclude_max_key (bool):
			    If `True` only count class's label with key < 255. Defaults: `True`.
        """
        count = 0
        for class_labels in self.class_labels:
            if hasattr(class_labels, key):
                if (exclude_negative_key and class_labels[key] < 0   ) or \
                   (exclude_max_key      and class_labels[key] >= 255):
                    continue
                count += 1
        return count

    # MARK: Custom Accessors

    def get_class_label(self, key: str = "id", value: Union[int, str, None] = None) -> Optional[dict]:
        """Get the class_label based on the given (`key`, `value`) pair."""
        for class_label in self.class_labels:
            if hasattr(class_label, key) and (value == class_label[key]):
                return class_label
        return None
    
    def get_class_label_by_name(self, name: str) -> Optional[dict]:
        """Get the class_label based on the given `name`."""
        return self.get_class_label(key="name", value=name)
    
    def get_id(self, key: str = "id", value: Union[int, str, None] = None) -> Optional[int]:
        """Get the id based on the given (`key`, `value`) pair."""
        class_label: dict = self.get_class_label(key=key, value=value)
        return class_label["id"] if class_label is not None else None
    
    def get_id_by_name(self, name: str) -> Optional[int]:
        """Get the id based on the given `name`."""
        class_label = self.get_class_label_by_name(name=name)
        return class_label["id"] if class_label is not None else None
    
    def get_name(self, key: str = "id", value: Union[int, str, None] = None) -> Optional[str]:
        """Get the class_label's name based on the given (`key`, `value`) pair.
        """
        class_label = self.get_class_label(key=key, value=value)
        return class_label["name"] if class_label is not None else None
       
    # MARK: Visualize

    def show_color_legend(self, height: Optional[int] = None):
        """Show a pretty color lookup legend using OpenCV drawing functions.

        Args:
            height (int, optional):
        		Height of the color legend image.
        """
        color_legend = self.color_legend(height=height)
        cv2.imshow(winname="Color Legend", mat=color_legend)
        cv2.waitKey(1)
    
    # MARK: Print
    
    def print(self):
        """Print all class_labels using `rich` package."""
        if not (self.class_labels and len(self.class_labels) > 0):
            console.log("[yellow]No class-label is available.")
            return
        
        console.log("[red]Classlabels:")
        print_table(self.class_labels)
