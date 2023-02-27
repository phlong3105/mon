#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the class for all retail products."""

from __future__ import annotations

__all__ = [
    "Product",
]

import cv2

import mon
from mon.globals import MovingState, OBJECTS
from supr import rmoi
from supr.data import hand


# region Product

@OBJECTS.register(name="product")
class Product(mon.MovingObject):
    """Retail product.
    
    See more: :class:`supr.data.base.MovingObject`.
    """

    min_touched_landmarks = 1  # Minimum hand landmarks touching the object so that it is considered hand-handling.
    min_confirms          = 3  # Minimum frames that the object is considered for counting.
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_confirms = 0
    
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
            entering_distance = roi.is_box_center_in_roi(bbox=self.current.bbox, compute_distance=True)
            if self.hit_streak >= self.min_hit_streak \
                and entering_distance >= self.min_entering_distance \
                and self.traveled_distance >= self.min_traveled_distance:
                self.moving_state = MovingState.CONFIRMED
        
        # From Confirmed --> Counting
        elif self.is_confirmed:
            # Method 1
            if hands is None:
                self.num_confirms += 1
                if self.num_confirms >= self.min_confirms:
                    self.moving_state = MovingState.TO_BE_COUNTED
                # if roi.is_box_in_roi(bbox=self.current.bbox) <= 0:
                #     self.moving_state = MovingState.COUNTING
            # Method 2
            else:
                num_lms_touches = 0
                box_points      = self.current.box_corners_points
                for landmarks in hands.multi_hand_landmarks:
                    for l in landmarks:
                        if int(cv2.pointPolygonTest(box_points, l, True)) >= 0:
                            num_lms_touches += 1
                if num_lms_touches >= self.min_touched_landmarks:
                    self.num_confirms += 1
                if self.num_confirms >= self.min_confirms:
                    self.moving_state = MovingState.TO_BE_COUNTED
                
        # From Counting --> ToBeCounted
        elif self.is_counting:
            self.moving_state = MovingState.TO_BE_COUNTED
        
        # From ToBeCounted --> Counted
        # Perform when counting the vehicle in :class:`supr.camera.base.Camera`
        # object.
        
        # From Counted --> Exiting
        elif self.is_counted:
            if roi.is_box_center_in_roi(bbox=self.current.bbox, compute_distance=True) <= 0:
                self.moving_state = MovingState.EXITING
            
# endregion
