#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the SORT: Observation-Centric SORT on video
Multi-Object Tracking.
"""

from __future__ import annotations

__all__ = [
    "KalmanBoxTrack",
    "SORT",
]

from timeit import default_timer as timer
from typing import Sequence

import numpy as np
import torch
from filterpy.kalman import KalmanFilter

from mon import core
from mon.globals import TrackState
from mon.vision.track import base

console = core.console


# region Matching

def speed_direction(bbox1: np.ndarray, bbox2: np.ndarray):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed    = np.array([cy2 - cy1, cx2 - cx1])
    norm     = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_batch(detections: np.ndarray, tracks: np.ndarray):
    tracks   = tracks[..., np.newaxis]
    cx1, cy1 = (detections[:, 0] + detections[:, 2]) / 2.0, (detections[:, 1] + detections[:, 3]) / 2.0
    cx2, cy2 = (    tracks[:, 0] +     tracks[:, 2]) / 2.0, (    tracks[:, 1] +     tracks[:, 3]) / 2.0
    dx       = cx1 - cx2
    dy       = cy1 - cy2
    norm     = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx       = dx / norm
    dy       = dy / norm
    return dy, dx  # size: num_track x num_det


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert a bounding box in the form of `[x1, y1, x2, y2]` and
    returns ``z`` in the form `[x, y, s, r]` where ``x``, ``y`` is the
    centre of the box and ``s`` is the scale/area and ``r`` is the aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score: float | None = None) -> np.ndarray:
    """Convert a bounding box in the centre form of `[x, y, s, r]` and
    returns it in the form of `[x1, y1, x2, y2]` where ``x1``, ``y1`` is
    the top left and ``x2``, ``y2`` is the bottom right.
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


def k_previous_observations(observations, current_age: int, k: int):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if current_age - dt in observations:
            return observations[current_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]

# endregion


# region Track

class KalmanBoxTrack(base.Track):
    """This class use Kalman Filter to represent the internal state of individual
    tracked objects based on bbox.
    """
    
    def __init__(
        self,
        bbox   : np.ndarray,
        id_    : int | None = None,
        state  : TrackState = TrackState.NEW,
        delta_t: int        = 3,
        origin : bool       = False,
    ):
        super().__init__(
            id_        = id_,
            state      = state,
            detections = [],
        )
        # Define constant velocity model
        self.kf   = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1]]
        )
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0]]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # Give high uncertainty to the unobservable initial velocities
        self.kf.P         *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4]      = convert_bbox_to_z(bbox)
        #
        self.hits              = 0
        self.hit_streak        = 0
        self.age               = 0
        self.time_since_update = 0
        self.predict_history   = []
    
    def update(
        self,
        frame_id  : int,
        bbox      : np.ndarray,
        confidence: float,
        classlabel: int | None  = None,
        timestamp : int | float = timer(),
        *args, **kwargs
    ):
        """Updates the state vector with observed bbox."""
        self.predict_history    = []
        self.time_since_update  = 0
        self.hits              += 1
        self.hit_streak        += 1
        self.kf.update(convert_bbox_to_z(bbox))
        # Append track's history
        self.history.append(
            base.Detection(
                frame_id   = frame_id,
                bbox       = bbox,
                confidence = confidence,
                classlabel = classlabel,
                timestamp  = timestamp,
                *args, **kwargs
            )
        )
        # Update tracking state
        if self.state == TrackState.NEW and self.hits >= 2:
            self.state = TrackState.TRACKED
    
    def predict(self):
        """Advances the state vector and returns the predicted bounding box
        estimate.
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
            self.kf.predict()
            self.age += 1
            if self.time_since_update > 0:
                self.hit_streak = 0
            self.time_since_update += 1
            self.predict_history.append(convert_x_to_bbox(self.kf.x))
            return self.predict_history[-1]
    
    def current_state(self):
        """Returns the current bounding box estimate."""
        return convert_x_to_bbox(self.kf.x)

# endregion


# region Tracker

class SORT(base.Tracker):
    
    def __init__(
        self,
        det_threshold: float,
        max_age      : int   = 30,
        min_hits     : int   = 3,
        iou_threshold: float = 0.3,
        association  : str   = "giou",
    ):
        super().__init__()
        self.det_threshold = det_threshold
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.association   = association
        self.tracks: list[KalmanBoxTrack] = []
    
    def update(
        self,
        detections: torch.Tensor | np.ndarray,
        input_size: int | Sequence[int],
        image_size: int | Sequence[int],
        frame_id  : int | None = None,
    ):
        """Requires: this method must be called once for each frame even with
        empty detections (use np.empty((0, 5)) for frames without detections).
        
        Args:
            detections: A :obj:`torch.Tensor` or :obj:`numpy.ndarray` of
                detections in the format of `[[x1, y1, x2, y2, score, class], ...]`.
            input_size: The size of the input image in the format ``[H, W]``.
            image_size: The size of the original image in the format ``[H, W]``.
            frame_id  : The frame number.
        """
        self.frame_count += 1
        # Post-process detections
        detections    = detections.cpu().numpy() if isinstance(detections, torch.Tensor) else detections
        scores        = detections[:,   4] * detections[:, 5]
        bboxes        = detections[:, 0:4]  # [x1, y1, x2, y2]
        classes       = detections[:,   5]
        # Scale the detections
        input_size    = core.parse_hw(input_size)
        image_size    = core.parse_hw(image_size)
        inp_h, inp_w  = input_size[0], input_size[1]
        img_h, img_w  = image_size[0], image_size[1]
        scale         = min(float(img_h) / float(inp_h), float(img_w) / float(inp_w))
        bboxes       /= scale
        # Filter detection with low confidence score
        dets          = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        remain_ids    = scores > self.det_threshold
        dets          = dets[remain_ids]
        classes       = classes[remain_ids]
        # Get predicted locations from existing tracks
        trks          = np.zeros((len(self.tracks), 5))
        to_del        = []
        ret           = []
        for t, trk in enumerate(trks):
            pos    = self.tracks[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
            detections    = dets,
            tracks        = trks,
            iou_threshold = self.iou_threshold,
            association   = self.association,
        )
        # Update matched trackers with assigned detections
        for m in matched:
            self.tracks[m[1]].update(
                frame_id   = frame_id or self.frame_count,
                bbox       = dets[m[0], 0:4],
                confidence = dets[m[0],   4],
                classlabel = classes[m[0]],
            )
        # Create and initialize new tracks for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTrack(dets[i, :])
            self.tracks.append(trk)
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            d = trk.current_state()[0]
            if (
                (trk.time_since_update < 1) and
                (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
            ):
                ret.append(np.concatenate((d, [trk.id_])).reshape(1, -1))
            i -= 1
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

# endregion
