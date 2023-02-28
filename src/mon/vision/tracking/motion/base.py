#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all motion models.

Notes:
    Discussion 01: Should we go with Mixin or Composition approach?
        - Mixin:
            - Adv.:
                - Clear behaviour of the moving object.
            - Dis-adv.:
                - Cannot change behavior (if we want for some reasons) at
                  runtime.
                - If we want to assign multiple motion models to a moving object
                  type, we must define several sub-classes.
                
        - Composition:
            - Adv.:
                - Can dynamically change the behavior at runtime.
            - Dis adv.:
                - Must provide a way to create the moving object with an
                  appropriate motion. We should write a factory function or use
                  the Factory approach.
"""

from __future__ import annotations

__all__ = [
    "Motion",
]

from abc import ABC, abstractmethod
from typing import Any


# region Motion

class Motion(ABC):
    """The base class for all motion models. It is used for predicting the next
    position of the tracking object.

    Args:
        hits: A number of frames having that track appear. Defaults to 0.
        hit_streak: A number of consecutive frames having that track appear.
            Defaults to 0.
        age: A number of frames while the track is alive. Defaults to 0.
        time_since_update: A number of consecutive frames having that track
            disappear. Defaults to 0.
    """
    
    def __init__(
        self,
        hits             : int = 0,
        hit_streak       : int = 0,
        age              : int = 0,
        time_since_update: int = 0,
    ):
        self.hits              = hits
        self.hit_streak        = hit_streak
        self.age               = age
        self.time_since_update = time_since_update
        self.history           = []
        # Store all the `predict` position of track in z-bounding bbox value,
        # these positions appear while no bounding matches the track if any
        # bounding bbox matches the track, then history = [].
    
    @abstractmethod
    def update(self, instance: Any, *args, **kwargs):
        """Update the state of the motion model with observed features.
    
		Args:
			instance: An instance of the tracking object. Get the specific
			    features used to update the motion model from new measurement of
			    the object.
		"""
        pass
    
    @abstractmethod
    def predict(self):
        """Advance the state of the motion model and return the estimation."""
        pass
    
    @abstractmethod
    def current(self):
        """Return the current motion model estimate."""
        pass
    
# endregion
