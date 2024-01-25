#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements classification labels."""

from __future__ import annotations

__all__ = [
    "ClassificationLabel",
    "ClassificationsLabel",
]

import numpy as np

from mon import core
from mon.data.base.label import base

console = core.console


# region Classification

class ClassificationLabel(base.Label):
    """A classification label for an image.
    
    See Also: :class:`mon.data.base.label.base.Label`.
    
    Args:
        id_: A class ID of the classification data. Default: ``-1`` means unknown.
        label: A label string. Default: ``''``.
        confidence: A confidence value for the data. Default: ``1.0``.
        logits: Logits associated with the labels. Default: ``None``.
    """
    
    def __init__(
        self,
        id_       : int   = -1,
        label     : str   = "",
        confidence: float = 1.0,
        logits    : np.ndarray | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f":param:`conf` must be between ``0.0`` and ``1.0``, "
                f"but got {confidence}."
            )
        if id_ <= 0 and label == "":
            raise ValueError(
                f"Either :param:`id` or name must be defined, "
                f"but got {id_} and {label}."
            )
        self.id_        = id_
        self.label      = label
        self.confidence = confidence
        self.logits     = np.array(logits) if logits is not None else None
    
    @classmethod
    def from_value(cls, value: ClassificationLabel | dict) -> ClassificationLabel:
        """Create a :class:`ClassificationLabel` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, dict):
            return ClassificationLabel(**value)
        elif isinstance(value, ClassificationLabel):
            return value
        else:
            raise ValueError(
                f"value must be a ClassificationLabel class or a dict, but got "
                f"{type(value)}."
            )
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [self.id_, self.label]
        

class ClassificationsLabel(list[ClassificationLabel], base.Label):
    """A list of classification labels for an image. It is used for multi-labels
    or multi-classes classification tasks.
    
    See Also: :class:`mon.data.base.label.base.Label`.
    
    Args:
        seq: A list of :class:`ClassificationLabel` objects.
    """

    def __init__(self, seq: list[ClassificationLabel | dict]):
        super().__init__(ClassificationLabel.from_value(value=i) for i in seq)
    
    def __setitem__(self, index: int, item: ClassificationLabel | dict):
        super().__setitem__(index, ClassificationLabel.from_value(item))
    
    def insert(self, index: int, item: ClassificationLabel | dict):
        super().insert(index, ClassificationLabel.from_value(item))
    
    def append(self, item: ClassificationLabel | dict):
        super().append(ClassificationLabel.from_value(item))
    
    def extend(self, other: list[ClassificationLabel | dict]):
        super().extend([ClassificationLabel.from_value(item) for item in other])
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [i.data for i in self]  

    @property
    def ids(self) -> list[int]:
        return [i.id_ for i in self]  

    @property
    def labels(self) -> list[str]:
        return [i.label for i in self]  

# endregion
