#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SORT tracker.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from aic.builder import TRACKERS
from aic.objects.detection import Detection
from aic.trackers.base import BaseTracker
from one import compute_box_iou_old
from one import ListOrTuple3T

np.random.seed(0)

__all__ = [
	"SORT",
]


# MARK: - SORT

@TRACKERS.register(name="sort")
class SORT(BaseTracker):
	"""SORT (Simple Online Realtime Tracker)."""

	# MARK: Magic Functions

	def __init__(self, name: str = "sort", *args, **kwargs):
		super().__init__(name=name, *args, **kwargs)

	# MARK: Update

	def update(self, detections: list[Detection]):
		"""Update `self.tracks` with new detections.

		Args:
			detections (list[Detection]):
                List of newly detected detections.

		Requires:
			This method must be called once for each frame even with empty
			detections, just call update with empty container.
		"""
		self.frame_count += 1  # Should be the same with VideoReader.index

		# NOTE: Extract and convert box from detections for easier use.
		if len(detections) > 0:
			# dets - a numpy array of detections in the format
			# [[x1,y1,x2,y2,score], [x1,y1,x2,y2,score],...]
			insts = np.array([
				np.append(np.float64(d.box), np.float64(d.confidence)) for d in detections
			])
		else:
			insts = np.empty((0, 5))
		
		# NOTE: Get predicted locations from existing trackers.
		trks  		= np.zeros((len(self.tracks), 5))
		del_indexes = []
		for t, trk in enumerate(trks):
			pos    = self.tracks[t].motion.predict_motion_state()[0]
			trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
			if np.any(np.isnan(pos)):
				del_indexes.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

		# NOTE: Find 3 lists of matches, unmatched_detections and
		# unmatched_trackers
		for t in reversed(del_indexes):
			self.tracks.pop(t)
		matched_indexes, unmatched_inst_indexes, unmatched_trks_indexes = \
			self.associate_instances_to_tracks(insts, trks)
		
		# NOTE: Update matched trackers with assigned detections
		self.update_matched_tracks(matched_indexes=matched_indexes, instances=detections)

		# NOTE: Create and initialise new trackers for unmatched detections
		self.create_new_tracks(unmatched_inst_indexes=unmatched_inst_indexes, instances=detections)

		# NOTE: Remove dead tracks
		self.delete_dead_tracks()

	def associate_instances_to_tracks(self, instances: np.ndarray, tracks: np.ndarray) -> ListOrTuple3T[np.ndarray]:
		"""Assigns `detections` to `self.tracks`.

        Args:
            instances (np.ndarray):
                Newly detected detections.
            tracks (np.ndarray):
                Current tracks.

        Returns:
            matched_indexes (np.ndarray):
            unmatched_inst_indexes (np.ndarray):
            unmatched_trks_indexes (np.ndarray):
        """
		if len(tracks) == 0:
			return (np.empty((0, 2), dtype=int), np.arange(len(instances)),
					np.empty((0, 5), dtype=int))

		iou_matrix = compute_box_iou_old(instances, tracks)
		
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

		return (matched_indexes,
				np.array(unmatched_inst_indexes),
				np.array(unmatched_trks_indexes))

	def update_matched_tracks(self, matched_indexes: Union[list, np.ndarray], instances: list[Detection]):
		"""Update tracks that have been matched with new detected detections.

		Args:
			matched_indexes (np.ndarray):
				Indexes of `self.tracks` that are matched with new detections.
			instances (list):
                Newly detected detections.
		"""
		for m in matched_indexes:
			track_idx     = m[1]
			detection_idx = m[0]
			self.tracks[track_idx].update(instances[detection_idx])

			# IF you don't call the function above, then call the following
			# functions:
			# self.tracks[track_idx].update_go_from_detection(measurement=detections[detection_idx])
			# self.tracks[track_idx].update_motion_state()

	def create_new_tracks(self, unmatched_inst_indexes: Union[list, np.ndarray], instances: list[Detection]):
		"""Create new tracks.

        Args:
            unmatched_inst_indexes (list, np.ndarray):
                Indexes of `detections` that have not been matched with any
                tracks.
            instances (list[Detection]):
                Newly detected detections.
        """
		for i in unmatched_inst_indexes:
			new_trk = self.object_type(
				detection = instances[i],
				motion    = self.motion_model(box=instances[i].box)
			)
			self.tracks.append(new_trk)

	def delete_dead_tracks(self):
		"""Delete dead tracks."""
		i = len(self.tracks)
		for trk in reversed(self.tracks):
			# Get the current bounding box of Kalman Filter
			d = trk.motion.current_motion_state()[0]
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


# MARK: - Utils

def linear_assignment(cost_matrix):
	"""

	Args:
		cost_matrix:

	Returns:

	"""
	try:
		import lap
		_, x, y = lap.lapjv(cost_matrix, extend_cost=True)
		return np.array([[y[i], i] for i in x if i >= 0])  #
	except ImportError:
		from scipy.optimize import linear_sum_assignment
		x, y = linear_sum_assignment(cost_matrix)
		return np.array(list(zip(x, y)))
