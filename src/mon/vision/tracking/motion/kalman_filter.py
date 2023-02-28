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
    "KFBBoxMotion", "KFORBMotion",
]

from typing import Any

import numpy as np
from filterpy import kalman

from mon.globals import MOTIONS
from mon.vision.tracking.motion import base


# region Helper Function

def box_xyxy_to_z(box: np.ndarray) -> np.ndarray:
    """Convert a bounding bbox from XYXY to the format used by Kalman Filter
    CXCYSR, where:
        X1, Y1 is the top left.
        X2, Y2 is the bottom right.
        CX, CY is the centre of the bbox.
        S is the scale/area.
        R is the aspect ratio.
    """
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = box[0] + w / 2.0
    y = box[1] + h / 2.0
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def box_x_to_xyxy(x: np.ndarray, score: float | None = None) -> np.ndarray:
    """Covert a bounding bbox from the format used in Kalman Filter CXCYSR to
    XYXY, where:
        X1, Y1 is the top left.
        X2, Y2 is the bottom right.
        CX, CY is the centre of the bbox.
        S is the scale/area.
        R is the aspect ratio.
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


# region Kalman Filter

@MOTIONS.register(name="kf_bbox_motion")
class KFBBoxMotion(base.Motion):
    """Model a moving object motion by using Kalman Filter on its bounding bbox
    features.
    
    Attributes:
        kf: A Kalman Filter model.
    
    Args:
        instance: An initial instance of the tracking object to initialize the
            Kalman Filter.
        hits: A number of frames having that track appear. Defaults to 0.
        hit_streak: A number of consecutive frames having that track appear.
            Defaults to 0.
        age: A number of frames while the track is alive. Defaults to 0.
        time_since_update: A number of consecutive frames having that track
            disappear. Defaults to 0.
    
    See more: :class:`mon.vision.tracking.motion.base.Motion`.
    """

    def __init__(
        self,
        instance         : Any = None,
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
        if instance is not None:
            if not hasattr(instance, "bbox"):
                raise ValueError("instance must contain 'bbox' attribute.")
            self.kf.x[0:4] = box_xyxy_to_z(instance.bbox)
            self.kf.predict()
        
    def update(self, instance: Any, **kwargs):
        """Updates the state of the motion model with observed bbox.

		Args:
			instance: An instance of the tracking object. Get the specific
			    features used to update the motion model from new measurement of
			    the object.
		"""
        if not hasattr(instance, "bbox"):
            raise ValueError("instance must contain 'bbox' attribute.")
        self.time_since_update  = 0
        self.history            = []
        self.hits              += 1
        self.hit_streak        += 1
        self.kf.update(box_xyxy_to_z(instance.bbox))

    def predict(self) -> np.ndarray:
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

    def current(self) -> np.ndarray:
        """Return the current motion model estimate."""
        return box_x_to_xyxy(self.kf.x)


@MOTIONS.register(name="kf_orb_motion")
class KFORBMotion(base.Motion):
    """Model a moving object motion by using Kalman Filter on its ORB features.
    
    Attributes:
        kf: A Kalman Filter model.
    
    Args:
        instance: An initial instance of the tracking object to initialize the
            Kalman Filter.
        hits: A number of frames having that track appear. Defaults to 0.
        hit_streak: A number of consecutive frames having that track appear.
            Defaults to 0.
        age: A number of frames while the track is alive. Defaults to 0.
        time_since_update: A number of consecutive frames having that track
            disappear. Defaults to 0.
    
    See more: :class:`mon.vision.tracking.motion.base.Motion`.
    """
    
    def __init__(
        self,
        instance         : Any = None,
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
        # Define Kalman Filter
        self.kf   = kalman.KalmanFilter(dim_x=32, dim_z=32)
        self.kf.F = np.eye(32)  # State transition matrix
        self.kf.H = np.eye(32)  # Measurement matrix
        self.kf.R = np.eye(32) * 0.1  # Measurement noise covariance matrix
        self.kf.Q = np.eye(32) * 0.001  # Process noise covariance matrix
        
        # Here we assume that the `MovingObject` has already been init()
        if instance is not None:
            if not hasattr(instance, "feature"):
                raise ValueError("instance must contain 'feature' attribute.")
            self.kf.x = instance.feature.flatten()  # Convert descriptors to feature vectors
            self.kf.predict()
        
    def update(self, instance: Any, **kwargs):
        """Updates the state of the motion model with observed bbox.

		Args:
			instance: An instance of the tracking object. Get the specific
			    features used to update the motion model from new measurement of
			    the object.
		"""
        if not hasattr(instance, "feature"):
            raise ValueError("instance must contain 'feature' attribute.")
        self.time_since_update  = 0
        self.history            = []
        self.hits              += 1
        self.hit_streak        += 1
        self.kf.update(instance.feature.flatten())  # Convert descriptors to feature vectors
    
    def predict(self) -> np.ndarray:
        """Advance the state of the motion model and return the estimation."""
        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(self.kf.x)
        return self.history[-1]

    def current(self) -> np.ndarray:
        """Return the current motion model estimate."""
        return self.kf.x

    
# endregion
