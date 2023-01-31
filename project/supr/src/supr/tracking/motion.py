#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements motion models tracking objects."""

from __future__ import annotations

__all__ = [
    "KFBoxMotion", "Motion", "box_x_to_xyxy", "box_xyxy_to_z",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from filterpy import kalman

from supr import constant


# region Helper Function

def box_xyxy_to_z(box: np.ndarray) -> np.ndarray:
    """Convert a bounding box from [x1, y1, x2, y2] to the format used by Kalman
    Filter [cx, cy, s, r], where:
        x1, y1 is the top left.
        x2, y2 is the bottom right.
        cx, cy is the centre of the box.
        s is the scale/area.
        r is the aspect ratio.
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = box[0] + w / 2.0
    y = box[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def box_x_to_xyxy(x: np.ndarray, score: float = None) -> np.ndarray:
    """Covert a bounding box from the format used in Kalman Filter
    [cx, cy, s, r] to [x1, y1, x2, y2], where:
        x1, y1 is the top left.
        x2, y2 is the bottom right.
        cx, cy is the centre of the box.
        s is the scale/area.
        r is the aspect ratio.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([
            x[0] - w / 2.0, x[1] - h / 2.0,
            x[0] + w / 2.0, x[1] + h / 2.0
        ]).reshape((1, 4))
    else:
        return np.array([
            x[0] - w / 2.0, x[1] - h / 2.0,
            x[0] + w / 2.0, x[1] + h / 2.0,
            score
        ]).reshape((1, 5))

# endregion


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
        *args, **kwargs
    ):
        self.hits              = hits
        self.hit_streak        = hit_streak
        self.age               = age
        self.time_since_update = time_since_update
        self.history           = []
        # Store all the `predict` position of track in z-bounding box value,
        # these positions appear while no bounding matches the track if any
        # bounding box matches the track, then history = [].

    @abstractmethod
    def update_motion_state(self, instance: Any, *args, **kwargs):
        """Update the state of the motion model with observed features.

		Args:
			instance: A tracking object. Get the specific features used to
			    update the motion model from new measurement of the object.
		"""
        pass

    @abstractmethod
    def predict_motion_state(self):
        """Advance the state of the motion model and return the estimation."""
        pass

    @abstractmethod
    def current_motion_state(self):
        """Return the current motion model estimate."""
        pass

# endregion


# region Kalman Filter

@constant.MOTION.register(name="kf_box_motion")
class KFBoxMotion(Motion):
    """Model a moving object motion by using Kalman Filter on its bounding box
    features. See more: :class:`Motion`.
    
    Attributes:
        kf: A Kalman Filter model.
    
    Args:
        box: An initial box in [x1, y1, x2, y2] format to initialize Kalman
            Filter.
        hits: A number of frames having that track appear. Defaults to 0.
        hit_streak: A number of consecutive frames having that track appear.
            Defaults to 0.
        age: A number of frames while the track is alive. Defaults to 0.
        time_since_update: A number of consecutive frames having that track
            disappear. Defaults to 0.
    """

    def __init__(
        self,
        box              : np.ndarray | None = None,
        hits             : int = 0,
        hit_streak       : int = 0,
        age              : int = 0,
        time_since_update: int = 0,
    ):
        super().__init__(
            hits              = hits,
            hit_streak        = hit_streak,
            age               = age,
            time_since_update = time_since_update,
        )

        # Define Kalman Filter (constant velocity model)
        self.kf   = kalman.KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # Give high uncertainty to the unobservable initial velocities
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Here we assume that the `MovingObject` has already been init()
        if box is not None:
            self.kf.x[0:4] = box_xyxy_to_z(box)
        
    def update_motion_state(self, moving_object: Any, **kwargs):
        """Updates the state of the motion model with observed box.

		Args:
			moving_object: A tracking object. Get the specific features used to
			    update the motion model from new measurement of the object.
		"""
        assert hasattr(moving_object, "box")
        self.time_since_update = 0
        self.history           = []
        self.hits              += 1
        self.hit_streak        += 1
        self.kf.update(box_xyxy_to_z(moving_object.box))

    def predict_motion_state(self) -> np.ndarray:
        """Advance the state of the motion model and return the estimation."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(box_x_to_xyxy(self.kf.x))
        return self.history[-1]

    def current_motion_state(self) -> np.ndarray:
        """Return the current motion model estimate."""
        return box_x_to_xyxy(self.kf.x)
