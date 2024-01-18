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
from mon.globals import AppleRGB, MovingState, OBJECTS
from supr import rmoi
from supr.obj import hand


# region Product

@OBJECTS.register(name="product")
class Product(mon.MovingObject):
    """Retail product.
    
    See more: :class:`mon.vision.tracking.obj.base.MovingObject`.
    """

    min_touched_landmarks = 1    # Minimum hand landmarks touching the object so that it is considered hand-handling.
    min_confirms          = 3    # Minimum frames that the object is considered for counting.
    min_counting_distance = 10   # Minimum distance to the ROI's center that the object is considered for counting.
    min_counting_iou      = 0.9  # Minimum IoU value between the ROI and the object that is considered for counting.
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_confirms   = 0
        self.counting_point = None

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
                    p1   = roi.center
                    p2   = self.current.bbox_center
                    dx   = p2[0] - p1[0]
                    dy   = p2[1] - p1[1]
                    sign = -1 if p2[0] < p1[0] or p2[1] < p1[1] else 0
                    d    = sign * mon.math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    iou  = 1 - roi.calculate_iou(bbox=self.current.bbox)
                    if iou >= self.min_counting_iou \
                        and d >= self.min_counting_distance \
                        or (abs(dx) <= 30):
                        self.moving_state   = MovingState.TO_BE_COUNTED
                        self.counting_point = p2
                # if roi.is_box_in_roi(bbox=self.current.bbox) <= 0:
                #     self.moving_state = MovingState.COUNTING
            # Method 2
            else:
                num_lms_touches = 0
                box_points      = self.current.bbox_corners_points
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
    
    def draw(
        self,
        image  : np.ndarray,
        bbox   : bool             = True,
        polygon: bool             = False,
        label  : bool             = True,
        color  : list[int] | None = None
    ) -> np.ndarray:
        """Draw the current object and its trajectory on the :param:`image`."""
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = color or self.majority_label["color"]
        
        if self.is_candidate:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = False,
                color   = [255, 255, 255]
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = [255, 255, 255],
                thickness  = 3,
            )
        elif self.is_confirmed:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = False,
                color   = [255, 255, 255]
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = [255, 255, 255],
                thickness  = 3,
            )
        elif self.is_counting:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = label,
                color   = [255, 255, 255]
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = [255, 255, 255],
                thickness  = 3,
            )
        elif self.is_counted:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = label,
                color   = color
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = color,
                thickness  = 3,
            )
        elif self.is_exiting:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = label,
                color   = color
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = color,
                thickness  = 3,
            )
        return image
    
    def draw_instance(
        self,
        image  : np.ndarray,
        bbox   : bool             = True,
        polygon: bool             = False,
        label  : bool             = True,
        color  : list[int] | None = None
    ) -> np.ndarray:
        """Draw the current object on the :param:`image`."""
        color = color or self.majority_label["color"]
        if bbox:
            b = self.current.bbox
            cv2.rectangle(
                img       = image,
                pt1       = (int(b[0]), int(b[1])),
                pt2       = (int(b[2]), int(b[3])),
                color     = color,
                thickness = 2
            )
            b_center = self.current.bbox_center.astype(int)
            cv2.circle(
                img       = image,
                center    = tuple(b_center),
                radius    = 3,
                thickness = -1,
                color     = color
            )
            if self.counting_point is not None:
                counting_point = self.counting_point.astype(int)
                cv2.circle(
                    img       = image,
                    center    = tuple(counting_point),
                    radius    = 9,
                    thickness = -1,
                    color     = color
                )
        if polygon:
            pts = self.current.polygon.reshape((-1, 1, 2))
            cv2.polylines(img=image, pts=pts, isClosed=True, color=color, thickness=2)
        if label:
            box_tl     = self.current.bbox[0:2]
            curr_label = self.majority_label
            text       = f"{curr_label['name']}, {curr_label['id'] + 1}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            org        = (int(box_tl[0]) + 5, int(box_tl[1]))
            cv2.putText(
                img       = image,
                text      = text,
                fontFace  = font,
                fontScale = 1.0,
                org       = org,
                color     = color,
                thickness = 2,
            )
        return image
    
# endregion
