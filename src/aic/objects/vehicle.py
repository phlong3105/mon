#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Newly detected object from detector model. Attributes includes:
bounding box, confident score, class, uuid, ...
"""

from __future__ import annotations

import numpy as np

from aic.builder import OBJECTS
from aic.camera.roi import ROI
from aic.objects.base import BaseMovingObject
from aic.objects.base import MovingState
from one import euclidean_distance

__all__ = [
    "Vehicle",
]


# MARK: - Vehicle

@OBJECTS.register(name="vehicle")
class Vehicle(BaseMovingObject):
    """Moving Vehicle.
    """

    # MARK: Update

    def update_trajectory(self):
        """Update trajectory with measurement's center point."""
        traveled_distance = euclidean_distance(self.trajectory[-1], self.current_box_center)
        if traveled_distance >= self.min_traveled_distance:
            self.trajectory = np.append(self.trajectory, [self.current_box_center], axis=0)

    def update_moving_state(self, rois: list[ROI], **kwargs):
        """Update the current state of the road_objects.
        One recommendation of the state diagram is as follow:

                (exist >= 10 frames)  (road_objects cross counting line)   (after being counted
                (in roi)                                               by counter)
        _____________          _____________                  ____________        ___________        ________
        | Candidate | -------> | Confirmed | ---------------> | Counting | -----> | Counted | -----> | Exit |
        -------------          -------------                  ------------        -----------        --------
              |                       |                                                                  ^
              |_______________________|__________________________________________________________________|
                                (mark by tracker when road_objects's max age > threshold)
        """
        roi = next((roi for roi in rois if roi.id_ == self.current_roi_id), None)
        if roi is None:
            return

        # NOTE: From Candidate --> Confirmed
        if self.is_candidate:
            entering_distance = roi.is_box_in_or_touch_roi(
                box_xyxy=self.current_box, compute_distance=True
            )
            if (self.hit_streak >= Vehicle.min_hit_streak and
                entering_distance >= Vehicle.min_entering_distance and
                self.traveled_distance >= Vehicle.min_traveled_distance):
                self.moving_state = MovingState.Confirmed

        # NOTE: From Confirmed --> Counting
        elif self.is_confirmed:
            if roi.is_box_in_or_touch_roi(box_xyxy=self.current_box) <= 0:
                self.moving_state = MovingState.Counting

        # NOTE: From Counting --> ToBeCounted
        elif self.is_counting:
            if (roi.is_center_in_or_touch_roi(box_xyxy=self.current_box) < 0 or
                self.time_since_update >= self.max_age):
                self.moving_state = MovingState.ToBeCounted

        # NOTE: From ToBeCounted --> Counted
        # Perform when counting the vehicle

        # NOTE: From Counted --> Exiting
        elif self.is_counted:
            if (roi.is_center_in_or_touch_roi(box_xyxy=self.current_box, compute_distance=True) <= 0 or
                self.time_since_update >= Vehicle.max_age):
                self.moving_state = MovingState.Exiting
