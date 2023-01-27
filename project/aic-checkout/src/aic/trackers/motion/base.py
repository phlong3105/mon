#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for modeling the motion of an individual tracked object.
"""

from __future__ import annotations

import abc

from aic.objects.detection import Detection

__all__ = [
    "Motion",
]


# MARK: - Motion

class Motion(metaclass=abc.ABCMeta):
    """Motion implements the base template to model how an individual tracked
    object moves. It is used for predicting the next position of the tracked
    object.

    Attributes:
        hits (int):
            Number of frame has that track appear.
        hit_streak (int):
            Number of `consecutive` frame has that track appear.
        age (int):
            Number of frame while the track is alive, from
            Candidate -> Deleted.
        time_since_update (int):
            Number of `consecutive` frame that track disappear.
        history (list):
            Store all the `predict` position of track in z-bounding box value,
            these position appear while no bounding matches the track if any
            bounding box matches the track, then history = [].
    """

    # MARK: Magic Functions

    def __init__(
        self,
        hits             : int = 0,
        hit_streak       : int = 0,
        age              : int = 0,
        time_since_update: int = 0,
        *args, **kwargs
    ):
        self.hits              = hits
        self.hit_streak        = hit_streak
        self.age               = age
        self.time_since_update = time_since_update
        self.history           = []

    # MARK: Update

    @abc.abstractmethod
    def update_motion_state(self, detection: Detection, *args, **kwargs):
        """Updates the state of the motion model with observed features.

		Args:
			detection (Detection):
				Get the specific features used to update the motion model from
				new measurement of the object.
		"""

    @abc.abstractmethod
    def predict_motion_state(self):
        """Advances the state of the motion model and returns the predicted
        estimate.
        """

    @abc.abstractmethod
    def current_motion_state(self):
        """Returns the current motion model estimate."""
