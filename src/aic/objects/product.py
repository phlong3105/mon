#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Retail Product.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from aic.builder import OBJECTS
from aic.camera.roi import ROI
from aic.objects.base import BaseMovingObject
from aic.objects.base import BaseObject
from aic.objects.base import MovingState
from aic.pose_estimators import Hands
from one import AppleRGB
from one import euclidean_distance

__all__ = [
    "Product",
]


# MARK: - Vehicle

# noinspection PyMethodOverriding
@OBJECTS.register(name="product")
class Product(BaseMovingObject):
    """Retail Product.
    """
    
    # MARK: Class Attributes
    
    min_touched_landmarks: int = 1  # Min hand landmarks touching the object so that it is considered hand-handling.
    max_untouches_age    : int = 3  # Max frames the product is untouched before considering for deletion.

    # MARK: Magic Functions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.untouches = 0
        
    # MARK: Update

    def update_trajectory(self):
        """Update trajectory with measurement's center point."""
        traveled_distance = euclidean_distance(self.trajectory[-1], self.current_box_center)
        if traveled_distance >= self.min_traveled_distance:
            self.trajectory = np.append(self.trajectory, [self.current_box_center], axis=0)

    def update_moving_state(self, rois: list[ROI], hands: Optional[Hands], **kwargs):
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
            if (self.hit_streak >= Product.min_hit_streak and
                entering_distance >= Product.min_entering_distance and
                self.traveled_distance >= Product.min_traveled_distance):
                self.moving_state = MovingState.Confirmed
            
        # NOTE: From Confirmed --> Counting
        elif self.is_confirmed:
            # NOTE: Here we want to look for non-hand-handling objects
            # Method 1
            # if (roi.is_box_in_or_touch_roi(box_xyxy=self.detections[0].box) > 0 or
            #     self.traveled_distance_between(-1, -2) <= Product.min_traveled_distance):
            #    self.moving_state = MovingState.Counted
            
            # Method 2
            if hands is not None:
                num_lms_touches = 0
                box_points      = self.current_box_corners_points
                for landmarks in hands.multi_hand_landmarks:
                    for l in landmarks:
                        if int(cv2.pointPolygonTest(box_points, l, True)) >= 0:
                            num_lms_touches += 1
                if num_lms_touches < Product.min_touched_landmarks:
                    self.untouches += 1
                else:
                    self.untouches = 0

            if roi.is_box_in_or_touch_roi(box_xyxy=self.current_box) <= 0:
                if self.untouches > Product.max_untouches_age:
                    self.moving_state = MovingState.Counted
                # elif (roi.is_box_in_or_touch_roi(box_xyxy=self.first_box) > 0 or
                #       self.traveled_distance_between(-1, -2) <= Product.min_traveled_distance):
                #    pass
                    # self.moving_state = MovingState.Counted
                else:
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
                self.time_since_update >= Product.max_age):
                self.moving_state = MovingState.Exiting
        
        # print(self.untouches, self.moving_state)
        
    # MARK: Visualize

    def draw(self, drawing: np.ndarray, **kwargs) -> np.ndarray:
        """Draw the object into the `drawing`.

        Args:
            drawing (np.ndarray):
                Drawing canvas.
        """
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = self.label_by_majority["color"]
            
        if self.is_confirmed:
            BaseObject.draw(self, drawing=drawing, label=False, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counting:
            BaseObject.draw(self, drawing=drawing, label=True, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_to_be_counted:
            BaseObject.draw(self, drawing=drawing, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counted:
            # BaseObject.draw(self, drawing=drawing, label=False, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_exiting:
            # BaseObject.draw(self, drawing=drawing, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
