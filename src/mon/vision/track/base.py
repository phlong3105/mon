#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for tracks and trackers."""

from __future__ import annotations

__all__ = [

]

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from mon import core
from mon.globals import MOTIONS, OBJECTS

console = core.console


# region Track

class Track(ABC):
    """The base class for all tracks.
    
    Definition: A track represents the trajectory or path that an object takes
    as it moves through a sequence of frames in a video or across multiple
    sensor readings. It consists of a series of positional data points
    corresponding to the object's location at different times.
    
    Components of a Track:
        Object Identification: Identifying the object of interest to be tracked.
        Positional Data: The coordinates (e.g., x and y positions in 2D space)
            of the object in each frame.
        Time Stamps: The specific times at which the object's positions are recorded.
        Track ID: A unique identifier for each tracked object to distinguish
            it from others in the scene.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# endregion


# region Tracker

class Tracker(ABC):
    """The base class for all trackers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

# endregion
