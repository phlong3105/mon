#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modeling the motion of an individual tracked object.
"""

from __future__ import annotations

import abc
from abc import abstractmethod
from typing import Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from one.core import MOTIONS
from one.data import Detection

__all__ = [
    "KFBoxMotion",
    "Motion",
]


# MARK: - Functional

def box_xyxy_to_z(xyxy: np.ndarray) -> np.ndarray:
    """Converting bounding box for Kalman Filter. Takes a bounding box in the
    form [x1, y1, x2, y2] and returns z in the form z=[cx, cy, s, r]
    Where:
        x1, y1 is the top left
        x2, y2 is the bottom right
        cx, cy is the centre of the box
        s is the scale/area
        r is the aspect ratio
    """
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    x = xyxy[0] + w / 2.0
    y = xyxy[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def box_x_to_xyxy(x: np.ndarray, score: float = None) -> np.ndarray:
    """Return bounding box from Kalman Filter. Takes a bounding box in the
    centre form [cx, cy, s, r] and returns it in the form [x1, y1, x2, y2]
    Where:
        x1, y1 is the top left
        x2, y2 is the bottom right
        cx, cy is the centre of the box
        s is the scale/area
        r is the aspect ratio
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
    

# MARK: - Modules

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
            Number of `consecutive` frame that track disappears.
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

    @abstractmethod
    def update(self, measurement, *args, **kwargs):
        """Updates the state of the motion model with observed features.

		Args:
			measurement:
				Get the specific features used to update the motion model from
				new measurement of the object.
		"""
        pass

    @abstractmethod
    def predict(self):
        """Advances the state of the motion model and returns the predicted
        estimate.
        """
        pass

    @abstractmethod
    def current(self):
        """Returns the current motion model estimate."""
        pass


@MOTIONS.register(name="kf_box_motion")
class KFBoxMotion(Motion):
    """This class represents the motion model as Kalman Filter of an individual
    tracked object observed as box.
    
    Attributes:
        hits (int):
            Number of frame has that track appear.
        hit_streak (int):
            Number of `consecutive` frame has that track appear.
        age (int):
            Number of frame while the track is alive, from Tentative -> Deleted.
        time_since_update (int):
            Number of `consecutive` frame that track disappears.
        kf (KalmanFilter):
            Kalman Filter model.
    
    Args:
        box (np.ndarray):
            Box to initialize Kalman Filter. They are expected to be in
            (x1, y1, x2, y2) format with `0 <= x1 < x2` and `0 <= y1 < y2`.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        box              : Optional[np.ndarray] = None,
        hits             : int = 0,
        hit_streak       : int = 0,
        age              : int = 0,
        time_since_update: int = 0,
        *args, **kwargs
    ):
        super().__init__(
            hits              = hits,
            hit_streak        = hit_streak,
            age               = age,
            time_since_update = time_since_update,
            *args, **kwargs
        )

        # NOte: Define Kalman Filter (constant velocity model)
        self.kf   = KalmanFilter(dim_x=7, dim_z=4)
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
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable
                                     # initial velocities
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Here we assume that the `MovingObject` has already been init()
        if box is not None:
            self.kf.x[0:4] = box_xyxy_to_z(box)
        
    # MARK: Update

    def update(self, measurement: Detection, **kwargs):
        """Updates the state of the motion model with observed box.

		Args:
			measurement (Detection):
				Get the specific features used to update the motion model from
				new `Detection`.
		"""
        self.time_since_update  = 0
        self.history            = []
        self.hits              += 1
        self.hit_streak        += 1
        self.kf.update(box_xyxy_to_z(measurement.box))

    def predict(self) -> np.ndarray:
        """Advances the state of the motion model and returns the predicted
        estimate.

        Returns:
            prediction (np.ndarray):
                Predicted boxes. They are expected to be in (x1, y1, x2, y2)
                format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(box_x_to_xyxy(self.kf.x))
        return self.history[-1]

    def current(self) -> np.ndarray:
        """Returns the current motion model estimate.

        Returns:
            current (np.ndarray):
                Current boxes. They are expected to be in (x1, y1, x2, y2)
                format with `0 <= x1 < x2` and `0 <= y1 < y2`.
        """
        return box_x_to_xyxy(self.kf.x)
