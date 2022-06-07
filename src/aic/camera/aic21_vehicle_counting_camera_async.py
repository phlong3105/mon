#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Camera class for counting vehicles moving through ROIs that matched
predefined MOIs.
"""

from __future__ import annotations

import threading
import uuid
from queue import Queue
from timeit import default_timer as timer
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

from aic.builder import CAMERAS
from aic.camera.aic21_vehicle_counting_camera import AICVehicleCountingCamera
from aic.camera.moi import MOI
from aic.camera.roi import ROI
from aic.detectors import BaseDetector
from aic.io import AIC21VehicleCountingWriter
from aic.objects import MovingState
from aic.trackers import BaseTracker
from one import AppleRGB
from one import ClassLabels
from one import CVVideoLoader
from one import CVVideoWriter

__all__ = [
	"AICVehicleCountingCameraAsync",
]


# MARK: - AICVehicleCountingCameraAsync

@CAMERAS.register(name="aic21_vehicle_counting_camera_async")
class AICVehicleCountingCameraAsync(AICVehicleCountingCamera):
	"""AIC Counting Camera Asynchronous implements the functions for Multi-Class
	Multi-Movement Vehicle Counting (MMVC).
	
	Attributes:
		id_ (int, str):
			Camera's unique ID.
		dataset (str):
			Dataset name. It is also the name of the directory inside
			`data_dir`. Default: `None`.
		subset (str):
			Subset name. One of: [`dataset_a`, `dataset_b`].
		name (str):
			Camera name. It is also the name of the camera's config files.
			Default: `None`.
		class_labels (ClassLabels):
			Classlabels.
		rois (list[ROI]):
			List of ROIs.
		mois (list[MOI]):
			List of MOIs.
		detector (BaseDetector):
			Detector model.
		tracker (BaseTracker):
			Tracker object.
		moving_object_cfg (dict):
			Config dictionary of moving object.
		data_loader (CVVideoLoader):
			Data loader object.
		data_writer (CVVideoWriter):
			Data writer object.
		result_writer (AIC21VehicleCountingWriter):
			Result writer object.
		queue_size (int):
			Queue size for multithread.
		verbose (bool):
			Verbosity mode. Default: `False`.
		save_image (bool):
			Should save individual images? Default: `False`.
		save_video (bool):
			Should save video? Default: `False`.
		save_results (bool):
			Should save results? Default: `False`.
		root_dir (str):
			Root directory is the full path to the dataset.
		configs_dir (str):
			`configs` directory located inside the root directory.
		rmois_dir (str):
			`rmois` directory located inside the root directory.
		outputs_dir (str):
			`outputs` directory located inside the root directory.
		video_dir (str):
			`video` directory located inside the root directory.
		mos (list):
			List of current moving objects in the camera.
		start_time (float):
			Start timestamp.
		pbar (tqdm):
			Progress bar.
	"""

	# MARK: Magic Functions

	def __init__(
		self,
		data_dir     : str,
		dataset      : str,
		subset       : str,
		name         : str,
		class_labels : Union[ClassLabels,  dict],
		rois         : Union[list[ROI],    dict],
		mois         : Union[list[MOI],    dict],
		detector     : Union[BaseDetector, dict],
		tracker      : Union[BaseTracker,  dict],
		moving_object: dict,
		data_loader  : Union[CVVideoLoader, dict],
		data_writer  : Union[CVVideoWriter, dict],
		result_writer: Union[AIC21VehicleCountingWriter, dict],
		id_          : Union[int, str] = uuid.uuid4().int,
		queue_size   : int             = 10,
		verbose      : bool            = False,
		save_image   : bool            = False,
		save_video   : bool            = False,
		save_results : bool            = True,
		*args, **kwargs
	):
		super().__init__(
			data_dir      = data_dir,
			dataset       = dataset,
			subset        = subset,
			name          = name,
			class_labels  = class_labels,
			rois          = rois,
			mois          = mois,
			detector      = detector,
			tracker       = tracker,
			moving_object = moving_object,
			data_loader   = data_loader,
			data_writer   = data_writer,
			result_writer = result_writer,
			id_           = id_,
			verbose       = verbose,
			save_image    = save_image,
			save_video    = save_video,
			save_results  = save_results,
			*args, **kwargs
		)
		self.queue_size = queue_size
		self.pbar       = None
		
		# NOTE: Queue
		self.frames_queue     = Queue(maxsize=self.queue_size)
		self.detections_queue = Queue(maxsize=self.queue_size)
		self.counting_queue   = Queue(maxsize=self.queue_size)

	# MARK: Run

	def run(self):
		"""Main run loop."""
		self.run_routine_start()
	
		# NOTE: Threading for video reader
		thread_data_reader = threading.Thread(target=self.run_data_reader)
		thread_data_reader.start()
	
		# NOTE: Threading for detector
		thread_detector = threading.Thread(target=self.run_detector)
		thread_detector.start()
	
		# NOTE: Threading for tracker
		thread_tracker = threading.Thread(target=self.run_tracker)
		thread_tracker.start()
	
		# NOTE: Threading for result writer
		thread_result_writer = threading.Thread(target=self.run_result_writer)
		thread_result_writer.start()
	
		# NOTE: Joins threads when all terminate
		thread_data_reader.join()
		thread_detector.join()
		thread_tracker.join()
		thread_result_writer.join()
		
		self.run_routine_end()

	def run_routine_start(self):
		"""Perform operations when run routine starts. We start the timer."""
		self.mos        = []
		self.pbar       = tqdm(total=len(self.data_loader), desc=f"{self.name}")
		self.start_time = timer()
		self.result_writer.start_time = self.start_time
		
		if self.verbose:
			cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)
	
	def run_data_reader(self):
		"""Run data reader thread and push images and frame_indexes to queue.
		"""
		for images, indexes, files, rel_paths in self.data_loader:
			if len(indexes) > 0:
				# NOTE: Push frame index and images to queue
				self.frames_queue.put([images, indexes, files, rel_paths])

		# NOTE: Push None to queue to act as a stopping condition for next thread
		self.frames_queue.put([None, None, None, None])
	
	def run_detector(self):
		"""Run detector thread and push measurement results to queue."""
		while True:
			# NOTE: Get frame indexes and images from queue
			(images, indexes, files, rel_paths) = self.frames_queue.get()
			if indexes is None:
				break
			if images is None:
				break
				
			# NOTE: Detect batch of images
			batch_detections = self.detector.detect(indexes=indexes, images=images)
	
			# NOTE: Associate detections with ROIs
			for idx, detections in enumerate(batch_detections):
				ROI.associate_detections_to_rois(detections=detections, rois=self.rois)
				batch_detections[idx] = [i for i in detections if (i.roi_id is not None)]
				
			# NOTE: Push detections to queue
			self.detections_queue.put([images, batch_detections])

		# NOTE: Push None to queue to act as a stopping condition for next thread
		self.detections_queue.put([None, None])
	
	def run_tracker(self):
		"""Run tracker thread."""
		while True:
			# NOTE: Get batch detections from queue
			images, batch_detections = self.detections_queue.get()
			if batch_detections is None:
				break

			# NOTE: Track batch detections
			for idx, detections in enumerate(batch_detections):
				self.tracker.update(detections=detections)
				self.mos = self.tracker.tracks
	
				# NOTE: Update moving objects' moving state
				for mo in self.mos:
					mo.update_moving_state(rois=self.rois)
					mo.timestamp = timer()
	
				# NOTE: Associate moving objects with MOI
				in_roi_mos = [
					o for o in self.mos if (o.is_confirmed or o.is_counting or o.is_to_be_counted)
				]
				# print(len(in_roi_mos))
				MOI.associate_moving_objects_to_mois(
					objs=in_roi_mos, mois=self.mois, shape_type="polygon"
				)
				to_be_counted_mos = [
					o for o in in_roi_mos if (o.is_to_be_counted and o.is_countable is False)
				]
				MOI.associate_moving_objects_to_mois(
					objs=to_be_counted_mos, mois=self.mois, shape_type="linestrip"
				)
	
				# NOTE: Count
				countable_mos = [o for o in in_roi_mos if (o.is_countable and o.is_to_be_counted)]
				for mo in countable_mos:
					mo.moving_state = MovingState.Counted

				# NOTE: Push countable moving objects to queue
				self.counting_queue.put([images, countable_mos])
				
			self.pbar.update(len(images))

		# NOTE: Push None to queue to act as a stopping condition for next thread
		self.counting_queue.put([None, None])
	
	def run_result_writer(self):
		"""Run result writer thread."""
		while True:
			# NOTE: Get countable moving objects from queue
			images, countable_mos = self.counting_queue.get()
			if countable_mos is None:
				break

			if self.save_results:
				self.result_writer.write(moving_objects=countable_mos)
			for image in images:
				self.postprocess(image=image)
			
	def run_routine_end(self):
		"""Perform operations when run routine ends."""
		if self.save_results:
			self.result_writer.dump()
			
		self.mos = []
		self.pbar.close()
		cv2.destroyAllWindows()
	
	def postprocess(self, image: np.ndarray, *args, **kwargs):
		"""Perform some postprocessing operations when a run step end.

		Args:
			image (np.ndarray):
				Image.
		"""
		if not self.verbose and not self.save_image and not self.save_video:
			return

		elapsed_time = timer() - self.start_time
		result       = self.draw(drawing=image, elapsed_time=elapsed_time)
		if self.verbose:
			cv2.imshow(self.name, result)
			cv2.waitKey(1)
		if self.save_video:
			self.data_writer.write(image=result)

	# MARK: Visualize

	def draw(self, drawing: np.ndarray, elapsed_time: float) -> np.ndarray:
		"""Visualize the results on the drawing.

		Args:
			drawing (np.ndarray):
				Drawing canvas.
			elapsed_time (float):
				Elapsed time per iteration.

		Returns:
			drawing (np.ndarray):
				Drawn canvas.
		"""
		# NOTE: Draw ROI
		[r.draw(drawing=drawing) for r in self.rois]
		# NOTE: Draw MOIs
		[m.draw(drawing=drawing) for m in self.mois]
		# NOTE: Draw Vehicles
		[o.draw(drawing=drawing) for o in self.mos]
		# NOTE: Draw frame index
		fps  = self.data_loader.index / elapsed_time
		text = (f"Frame: {self.data_loader.index}: "
				f"{format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)")
		font = cv2.FONT_HERSHEY_SIMPLEX
		org  = (20, 30)
		cv2.rectangle(img=drawing, pt1=(10, 0), pt2=(600, 40),
					  color=AppleRGB.BLACK.value, thickness=-1)
		cv2.putText(
			img       = drawing,
			text      = text,
			fontFace  = font,
			fontScale = 1.0,
			org       = org,
			color     = AppleRGB.WHITE.value,
			thickness = 2
		)
		return drawing
