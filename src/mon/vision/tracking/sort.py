#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Simple Online Realtime Tracker (SORT) tracker."""

from __future__ import annotations

__all__ = [
    "SORT", "SORTBBox",
]

import numpy as np

from mon.foundation import console
from mon.globals import TRACKERS
from mon.vision import geometry
from mon.vision.tracking import base, motion as mmotion

np.random.seed(0)


# region Helper Function

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
    
    See more: :class:`mon.vision.model.track.base.Tracker`.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.motion_type, type(mmotion.KFBBoxMotion)):
            track = SORTBBox(*args, **kwargs)
            self.__class__ = track.__class__
            self.__dict__  = track.__dict__
        else:
            raise RuntimeError
    
    def update(self, instances: list | np.ndarray = ()):
        console.log(f"This function should not be called!")
        pass
    
    def assign_instances_to_tracks(
        self,
        instances: list | np.ndarray,
        tracks   : list | np.ndarray,
    ):
        console.log(f"This function should not be called!")
        pass
    
    def update_matched_tracks(
        self,
        matched_indexes: list | np.ndarray,
        instances      : list | np.ndarray
    ):
        console.log(f"This function should not be called!")
        pass
    
    def delete_dead_tracks(self):
        console.log(f"This function should not be called!")
        pass


@TRACKERS.register(name="sort_bbox")
class SORTBBox(base.Tracker):
    """SORT (Simple Online Realtime Tracker) for bounding box.
    
    See more: :class:`mon.vision.model.track.base.Tracker`.
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
            insts = np.array([
                np.append(np.float64(i.bbox), np.float64(i.confidence))
                for i in instances
            ])
        else:
            insts = np.empty((0, 5))
        
        # Extract previously predicted boxes from existing trackers.
        trks  		= np.zeros((len(self.tracks), 5))
        del_indexes = []
        for t, trk in enumerate(trks):
            pos    = self.tracks[t].motion.predict()[0]
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

        iou_matrix = geometry.get_bbox_iou2(bbox1=instances, bbox2=tracks)
        
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
            track_idx    = m[1]
            instance_idx = m[0]
            self.tracks[track_idx].update(instance=instances[instance_idx])
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
