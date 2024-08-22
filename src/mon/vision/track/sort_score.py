#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the SORT: A Simple, Online and Realtime Tracker."""

from __future__ import annotations

__all__ = [
    "KalmanBoxScoreTrack",
    "SORTScore",
]

from timeit import default_timer as timer

import numpy as np
import torch
from filterpy.kalman import KalmanFilter

from mon import core
from mon.globals import TrackState
from mon.vision import geometry
from mon.vision.track import base, sort

console = core.console


# region Matching

def score_diff_batch(bboxes1, bboxes2) -> np.ndarray:
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    score2  = bboxes2[..., 4]
    score1  = bboxes1[..., 4]
    return abs(score2 - score1)


def associate_detections_to_tracks(
    detections           : np.ndarray,
    tracks               : np.ndarray,
    iou_threshold        : float = 0.3,
    association          : str   = "giou",
    tcm_first_step       : bool  = False,
    tcm_first_step_weight: float = 1.0,
):
    """Assigns detections to tracked objects (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections, and unmatched_tracks
    """
    if len(tracks) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    
    # iou_matrix = iou_batch(detections, tracks)
    if association == "giou":
        iou_matrix = geometry.bbox_giou(detections, tracks)
    else:
        iou_matrix = geometry.bbox_iou(detections, tracks)
    
    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if tcm_first_step:
                cost_matrix     = iou_matrix - score_diff_batch(detections, tracks) * tcm_first_step_weight
                matched_indices = sort.linear_assignment(-cost_matrix)
            else:
                matched_indices = sort.linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_tracks = []
    for t, trk in enumerate(tracks):
        if t not in matched_indices[:, 1]:
            unmatched_tracks.append(t)
    
    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_tracks.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_tracks)

# endregion


# region Track

class KalmanBoxScoreTrack(base.Track):
    """This class use Kalman Filter to represent the internal state of individual
    tracked objects based on bbox and confidence score.
    """
    
    def __init__(
        self,
        bbox         : np.ndarray,
        det_threshold: float,
        id_          : int | None = None,
        state        : TrackState = TrackState.NEW,
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
        self.kf.x[:4]      = sort.convert_bbox_to_z(bbox)
        #
        self.kf_score   = KalmanFilter(dim_x=2, dim_z=1)
        self.kf_score.F = np.array([[1, 1],
                                    [0, 1]])
        self.kf_score.H = np.array([[1, 0]])
        self.kf_score.R[0:, 0:]  *= 10.0
        self.kf_score.P[1:, 1:]  *= 1000.0  # Give high uncertainty to the unobservable initial velocities
        self.kf_score.P          *= 10.0
        self.kf_score.Q[-1, -1 ] *= 0.01
        self.kf_score.Q[1:,  1:] *= 0.01
        self.kf_score.x[:1]       = bbox[-1]
        #
        self.det_threshold     = det_threshold
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
        self.kf.update(sort.convert_bbox_to_z(bbox))
        self.kf_score.update(confidence)
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
        """Advances the state vector and returns the predicted bounding box and
        confidence score estimate.
        """
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
            self.kf.predict()
            self.kf_score.predict()
            self.age += 1
            if self.time_since_update > 0:
                self.hit_streak = 0
            self.time_since_update += 1
            self.predict_history.append(sort.convert_x_to_bbox(self.kf.x))
            return self.predict_history[-1], np.clip(self.kf_score.x[0], self.det_threshold, 1.0)
    
    def current_state(self):
        """Returns the current bounding box estimate."""
        return sort.convert_x_to_bbox(self.kf.x)
    
# endregion


# region Tracker

class SORTScore(base.Tracker):
    
    def __init__(
        self,
        det_threshold        : float,
        max_age              : int   = 30,
        min_hits             : int   = 3,
        iou_threshold        : float = 0.3,
        association          : str   = "giou",
        tcm_first_step       : bool  = True,
        tcm_first_step_weight: float = 1.0,
    ):
        super().__init__()
        self.det_threshold         = det_threshold
        self.max_age               = max_age
        self.min_hits              = min_hits
        self.iou_threshold         = iou_threshold
        self.association           = association
        self.tcm_first_step        = tcm_first_step
        self.tcm_first_step_weight = tcm_first_step_weight
        self.tracks: list[KalmanBoxScoreTrack] = []
    
    def update(
        self,
        detections: torch.Tensor | np.ndarray,
        # input_size: int | Sequence[int],
        # image_size: int | Sequence[int],
        frame_id  : int | None = None,
    ):
        """Requires: this method must be called once for each frame even with
        empty detections (use np.empty((0, 5)) for frames without detections).
        
        Args:
            detections: A :obj:`torch.Tensor` or :obj:`numpy.ndarray` of
                detections in the format of `[[x1, y1, x2, y2, score, class], ...]`.
            input_size: The size of the input image in the format `[h, w]`.
            image_size: The size of the original image in the format `[h, w]`.
            frame_id  : The frame number.
        """
        self.frame_count += 1
        # Post-process detections
        detections    = detections.cpu().numpy() if isinstance(detections, torch.Tensor) else detections
        scores        = detections[:,   4] * detections[:, 5]
        bboxes        = detections[:, 0:4]  # [x1, y1, x2, y2]
        classes       = detections[:,   5]
        '''
        # Scale the detections
        input_size    = core.parse_hw(input_size)
        image_size    = core.parse_hw(image_size)
        inp_h, inp_w  = input_size[0], input_size[1]
        img_h, img_w  = image_size[0], image_size[1]
        scale         = min(float(img_h) / float(inp_h), float(img_w) / float(inp_w))
        bboxes       /= scale
        '''
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
            pos, trk_score = self.tracks[t].predict()[0]
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], trk_score]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.tracks.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_tracks(
            detections            = dets,
            tracks                = trks,
            iou_threshold         = self.iou_threshold,
            association           = self.association,
            tcm_first_step        = self.tcm_first_step,
            tcm_first_step_weight = self.tcm_first_step_weight
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
            trk = KalmanBoxScoreTrack(dets[i, :], self.det_threshold)
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
