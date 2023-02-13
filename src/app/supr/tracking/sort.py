#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Simple Online Realtime Tracker (SORT) tracker."""

from __future__ import annotations

__all__ = [
    "SORT",
]

import numpy as np

from supr.globals import TRACKERS
from supr.tracking import base

np.random.seed(0)


# region Helper Function

def compute_box_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> np.ndarray:
    """From SORT: Compute IOU between two sets of boxes.
    
    Return intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in [x1, y1, x2, y2] format with
    `0 <= x1 < x2` and `0 <= y1 < y2`.

    Args:
        bbox1: The first set of boxes of shape [N, 4].
        bbox2: The second set of boxes of shape [M, 4].
    
    Returns:
        The NxM matrix containing the pairwise IoU values for every element in
        boxes1 and boxes2.
    """
    assert bbox1.ndim == 2
    assert bbox2.ndim == 2
    if isinstance(bbox1, np.ndarray) and type(bbox1) == type(bbox2):
        bbox1 = np.expand_dims(bbox1, 1)
        bbox2 = np.expand_dims(bbox2, 0)
        xx1   = np.maximum(bbox1[..., 0], bbox2[..., 0])
        yy1   = np.maximum(bbox1[..., 1], bbox2[..., 1])
        xx2   = np.minimum(bbox1[..., 2], bbox2[..., 2])
        yy2   = np.minimum(bbox1[..., 3], bbox2[..., 3])
        w     = np.maximum(0.0, xx2 - xx1)
        h     = np.maximum(0.0, yy2 - yy1)
    else:
        raise TypeError
    wh  = w * h
    iou = wh / ((bbox1[..., 2] - bbox1[..., 0]) *
                (bbox1[..., 3] - bbox1[..., 1]) +
                (bbox2[..., 2] - bbox2[..., 0]) *
                (bbox2[..., 3] - bbox2[..., 1]) - wh)
    return iou


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

# endregion


# region SORT

@TRACKERS.register(name="sort")
class SORT(base.Tracker):
    """SORT (Simple Online Realtime Tracker).
    
    See more: :class:`supr.tracking.base.Tracker`.
    """
    
    def update(self, instances: list | np.ndarray = ()):
        """Update :attr:`tracks` with new detections. This method will call the
        following methods:
            - :meth:`assign_instances_to_tracks`
            - :meth:`update_matched_tracks`
            - :meth:`create_new_tracks`
            - :meth`:delete_dead_tracks`

        Args:
            instances: A list of new instances. Defaults to ().
        """
        self.frame_count += 1  # Should be the same with VideoReader.index

        # Extract boxes from instances.
        if len(instances) > 0:
            # dets - a numpy array of detections in the format
            # [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],...]
            insts = np.array([np.append(np.float64(i.bbox), np.float64(i.confidence)) for i in instances])
        else:
            insts = np.empty((0, 5))
        
        # Extract previously predicted boxes from existing trackers.
        trks  		= np.zeros((len(self.tracks), 5))
        del_indexes = []
        for t, trk in enumerate(trks):
            pos    = self.tracks[t].motion.predict_motion_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                del_indexes.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        # Delete
        for t in reversed(del_indexes):
            self.tracks.pop(t)
            
        matched_indexes, unmatched_inst_indexes, unmatched_trks_indexes = \
            self.assign_instances_to_tracks(instances=insts, tracks=trks)
        
        # Update matched trackers with assigned detections
        self.update_matched_tracks(
            matched_indexes = matched_indexes,
            instances       = instances
        )
        # Create and initialise new trackers for unmatched detections
        self.create_new_tracks(
            unmatched_inst_indexes = unmatched_inst_indexes,
            instances              = instances
        )
        # Delete all dead tracks
        self.delete_dead_tracks()

    def assign_instances_to_tracks(
        self,
        instances: list | np.ndarray,
        tracks   : list | np.ndarray,
    ) -> tuple[
        list | np.ndarray,
        list | np.ndarray,
        list | np.ndarray
    ]:
        """Assigns new :param:`instances` to :param:`tracks`.

        Args:
            instances: A list of new instances
            tracks: A list of existing tracks.

        Returns:
            A list of tracks' indexes that have been matched with new instances.
            A list of new instances' indexes of that have NOT been matched with
                any tracks.
            A list of tracks' indexes that have NOT been matched with new
                instances.
        """
        if len(tracks) == 0:
            return \
                np.empty((0, 2), dtype=int), \
                np.arange(len(instances)),\
                np.empty((0, 5), dtype=int)

        iou_matrix = compute_box_iou(bbox1=instances, bbox2=tracks)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_inst_indexes = []
        for d, det in enumerate(instances):
            if d not in matched_indices[:, 0]:
                unmatched_inst_indexes.append(d)
        unmatched_trks_indexes = []
        for t, trk in enumerate(instances):
            if t not in matched_indices[:, 1]:
                unmatched_trks_indexes.append(t)

        # filter out matched with low IOU
        matched_indexes = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_inst_indexes.append(m[0])
                unmatched_trks_indexes.append(m[1])
            else:
                matched_indexes.append(m.reshape(1, 2))

        if len(matched_indexes) == 0:
            matched_indexes = np.empty((0, 2), dtype=int)
        else:
            matched_indexes = np.concatenate(matched_indexes, axis=0)

        return \
            matched_indexes,\
            np.array(unmatched_inst_indexes),\
            np.array(unmatched_trks_indexes)

    def update_matched_tracks(
        self,
        matched_indexes: list | np.ndarray,
        instances      : list | np.ndarray
    ):
        """Update existing tracks that have been matched with new instances.

        Args:
            matched_indexes: A list of tracks' indexes that have been matched
                with new instances.
            instances: A list of new instances.
        """
        for m in matched_indexes:
            track_idx     = m[1]
            detection_idx = m[0]
            self.tracks[track_idx].update(instances[detection_idx])
            # IF you don't call the function above, then call the following
            # functions:
            # self.tracks[track_idx].update_go_from_detection(measurement=detections[detection_idx])
            # self.tracks[track_idx].update_motion_state()

    def delete_dead_tracks(self):
        """Delete all dead tracks."""
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            # Get the current bounding bbox of Kalman Filter
            d = trk.motion.current()[0]
            """
            if (
                (trk.time_since_update < 1) and
                (trk.hit_streak >= self.min_hits or
                 self.frame_count <= self.min_hits)
            ):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            """
            i -= 1
            # NOTE: Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.tracks.pop(i)

# endregion
