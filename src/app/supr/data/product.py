#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the class for all retail products."""

from __future__ import annotations

__all__ = [
    "Product",
]

import cv2
import numpy as np

import mon
from supr import rmoi
from supr.data import base, hand
from supr.globals import MovingState, OBJECTS


# region Product

# noinspection PyMethodOverriding
@OBJECTS.register(name="product")
class Product(base.MovingObject):
    """The retail product class.
    
    See more: :class:`base.MovingObject`.
    """
    
    min_touched_landmarks: int = 1  # Min hand landmarks touching the object so that it is considered hand-handling.
    max_untouches_age    : int = 3  # Max frames the product is untouched before considering for deletion.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.untouches = 0
    
    def update_moving_state(
        self,
        rois : list[rmoi.ROI],
        hands: hand.Hands | None = None,
        **kwargs
    ):
        """Update the current state of the road_objects. One recommendation of
        the state diagram is as follows:
        _____________      _____________      ____________      ___________      ________
        | Candidate | ---> | Confirmed | ---> | Counting | ---> | Counted | ---> | Exit |
        -------------      -------------      ------------      -----------      --------
              |                 |                                                   ^
              |_________________|___________________________________________________|
                    (mark by a tracker when road_objects's max age > threshold)
        """
        roi = next((roi for roi in rois if roi.id_ == self.current.roi_id), None)
        if roi is None:
            return
        
        # From Candidate --> Confirmed
        if self.is_candidate:
            entering_distance = roi.is_box_in_roi(
                bbox             = self.current.bbox,
                compute_distance = True,
            )
            if (
                self.hit_streak >= self.min_hit_streak
                and entering_distance >= self.min_entering_distance
                and self.traveled_distance >= self.min_traveled_distance
            ):
                self.moving_state = MovingState.CONFIRMED
            
        # From Confirmed --> Counting
        elif self.is_confirmed:
            # NOTE: Here we want to look for non-hand-handling objects
            # Method 1
            # if (roi.is_box_in_or_touch_roi(box_xyxy=self.detections[0].bbox) > 0 or
            #     self.traveled_distance_between(-1, -2) <= Product.min_traveled_distance):
            #    self.moving_state = MovingState.Counted
            
            # Method 2
            if hands is not None:
                num_lms_touches = 0
                box_points      = self.current.box_corners_points
                for landmarks in hands.multi_hand_landmarks:
                    for l in landmarks:
                        if int(cv2.pointPolygonTest(box_points, l, True)) >= 0:
                            num_lms_touches += 1
                if num_lms_touches < self.min_touched_landmarks:
                    self.untouches += 1
                else:
                    self.untouches = 0

            if roi.is_box_in_roi(bbox=self.current.bbox) <= 0:
                if self.untouches > self.max_untouches_age:
                    self.moving_state = MovingState.COUNTED
                # elif (roi.is_box_in_or_touch_roi(box_xyxy=self.first_box) > 0 or
                #      self.traveled_distance_between(-1, -2) <= Product.min_traveled_distance):
                #    pass
                    # self.moving_state = MovingState.Counted
                else:
                    self.moving_state = MovingState.COUNTING
            
        # From Counting --> ToBeCounted
        elif self.is_counting:
            if (
                roi.is_box_center_in_roi(bbox=self.current.bbox) < 0
                or self.time_since_update >= self.max_age
            ):
                self.moving_state = MovingState.TO_BE_COUNTED

        # From ToBeCounted --> Counted
        # Perform when counting the vehicle

        # From Counted --> Exiting
        elif self.is_counted:
            if (
                roi.is_box_center_in_roi(bbox=self.current.bbox, compute_distance=True) <= 0
                or self.time_since_update >= Product.max_age
            ):
                self.moving_state = MovingState.EXITING
        
        # print(self.untouches, self.moving_state)
        
    def draw(self, drawing: np.ndarray, **kwargs) -> np.ndarray:
        """Draw the current object on the :param:`image`."""
        if self.moi_uid is not None:
            color = mon.AppleRGB.values()[self.moi_uid]
        else:
            color = self.majority_label["color"]
        if self.is_confirmed:
            base.MovingObject.draw(self, drawing=drawing, label=False, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counting:
            base.MovingObject.draw(self, drawing=drawing, label=True, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_to_be_counted:
            base.MovingObject.draw(self, drawing=drawing, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counted:
            # base.MovingObject.draw(self, image=image, label=False, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_exiting:
            # base.MovingObject.draw(self, image=image, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
        return drawing
    
# endregion
