#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements keypoint labels."""

from __future__ import annotations

__all__ = [
    "COCOKeypointsLabel",
    "KeypointLabel",
    "KeypointsLabel",
]

from mon import core
from mon.data.base.label import base

console = core.console


# region Keypoint

class KeypointLabel(base.Label):
    """A list keypoints label for a single object in an image.
    
    See Also: :class:`mon.data.base.label.base.Label`.
    
    Args:
        id_: The class ID of the polyline data. Default: ``-1`` means unknown.
        index: An index for the polyline. Default: ``-1``.
        label: The label string. Default: ``''``.
        confidence: A confidence value for the data. Default: ``1.0``.
        points: A list of lists of :math:`(x, y)` points in :math:`[0, 1] x [0, 1]`.
    """
    
    def __init__(
        self,
        id_       : int   = -1,
        index     : int   = -1,
        label     : str   = "",
        confidence: float = 1.0,
        points    : list  = [],
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
        self.index      = index
        self.label      = label
        self.confidence = confidence
        self.points     = points

    @classmethod
    def from_value(cls, value: KeypointLabel | dict) -> KeypointLabel:
        """Create a :class:`KeypointLabel` object from an arbitrary
        :param:`value`.
        """
        if isinstance(value, dict):
            return KeypointLabel(**value)
        elif isinstance(value, KeypointLabel):
            return value
        else:
            raise ValueError(
                f":param:`value` must be a :class:`KeypointLabel` class or a "
                f":class:`dict`, but got {type(value)}."
            )
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [
            self.points, self.id_, self.label, self.confidence, self.index
        ]


class KeypointsLabel(list[KeypointLabel], base.Label):
    """A list of keypoint labels for multiple objects in an image.
    
    See Also: :class:`mon.data.base.label.base.Label`.
    
    Args:
        seq: A list of :class:`KeypointLabel` objects.
    """
    
    def __init__(self, seq: list[KeypointLabel | dict]):
        super().__init__(KeypointLabel.from_value(value=i) for i in seq)
    
    def __setitem__(self, index: int, item: KeypointLabel | dict):
        super().__setitem__(index, KeypointLabel.from_value(item))
    
    def insert(self, index: int, item: KeypointLabel | dict):
        super().insert(index, KeypointLabel.from_value(item))
    
    def append(self, item: KeypointLabel | dict):
        super().append(KeypointLabel.from_value(item))
    
    def extend(self, other: list[KeypointLabel | dict]):
        super().extend([KeypointLabel.from_value(item) for item in other])
    
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

    @property
    def points(self) -> list:
        return [i.points for i in self]  
    

class COCOKeypointsLabel(KeypointsLabel):
    """A list of keypoint labels for multiple objects in COCO format.
    
    See Also: :class:`KeypointsLabel`.
    """
    pass

# endregion
