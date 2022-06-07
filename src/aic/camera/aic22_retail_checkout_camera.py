#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Camera class for counting vehicles moving through ROIs that matched
predefined MOIs.
"""

from __future__ import annotations

import math
import os
import uuid
from timeit import default_timer as timer
from typing import Union

import cv2
import numpy as np

from aic.builder import CAMERAS
from aic.builder import DETECTORS
from aic.builder import TRACKERS
from aic.camera.base import BaseCamera
from aic.camera.moi import MOI
from aic.camera.roi import ROI
from aic.detectors import BaseDetector
from aic.io import AIC22RetailCheckoutWriter
from aic.objects import MovingState
from aic.objects import Product
from aic.pose_estimators import Hands
from aic.pose_estimators import HandsEstimator
from aic.trackers import BaseTracker
from one import AppleRGB
from one import ClassLabels
from one import CVVideoLoader
from one import CVVideoWriter
from one import error_console
from one import is_basename
from one import is_json_file
from one import is_list_of
from one import is_stem
from one import is_video_file
from one import progress_bar

__all__ = [
    "AIC22RetailCheckoutCamera",
]


# MARK: - AIC22RetailCheckoutCamera

@CAMERAS.register(name="aic22_retail_checkout_camera")
class AIC22RetailCheckoutCamera(BaseCamera):
    """AIC22 Retail Checkout Camera.
	
	Attributes:
		id_ (int, str):
			Camera's unique ID.
		dataset (str):
			Dataset name. It is also the name of the directory inside `data_dir`.
			Default: `None`.
		subset (str):
			Subset name. One of: [`test_a`, `test_b`].
		name (str):
			Camera name. It is also the name of the camera's config files.
			Default: `None`.
		class_labels (ClassLabels):
			List of all labels' dicts.
		rois (list[ROI]):
			List of ROIs.
		mois (list[MOI]):
			List of MOIs.
		detector (BaseDetector):
			Detector model.
		tracker (BaseTracker):
			Tracker object.
		hands_estimator (HandsEstimator):
			`HandsEstimator` object.
		moving_object_cfg (dict):
			Config dictionary of moving object.
		data_loader (CVVideoLoader):
			Data loader object.
		data_writer (CVVideoWriter):
			Data writer object.
		result_writer (AIC22RetailCheckoutWriter):
			Result writer object.
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
	"""

    # MARK: Magic Functions

    def __init__(
        self,
        data_dir       : str,
        dataset        : str,
        subset         : str,
        name           : str,
        class_labels   : Union[ClassLabels,    dict],
        rois           : Union[list[ROI],      dict],
        mois           : Union[list[MOI],      dict],
        detector       : Union[BaseDetector,   dict],
        tracker        : Union[BaseTracker,    dict],
        hands_estimator: Union[HandsEstimator, dict],
        moving_object  : dict,
        data_loader    : Union[CVVideoLoader, dict],
        data_writer    : Union[CVVideoWriter, dict],
        result_writer  : Union[AIC22RetailCheckoutWriter, dict],
        id_            : Union[int, str] = uuid.uuid4().int,
        verbose        : bool            = False,
        save_image     : bool            = False,
        save_video     : bool            = False,
        save_results   : bool            = True,
        *args, **kwargs
    ):
        super().__init__(id_=id_, dataset=dataset, name=name)
        self.subset            = subset
        self.data              = kwargs.get("data", None)
        self.moving_object_cfg = moving_object
        self.verbose           = verbose
        self.save_image        = save_image
        self.save_video        = save_video
        self.save_results      = save_results

        self.init_dirs(data_dir=data_dir)
        self.init_class_labels(class_labels=class_labels)
        self.init_rois(rois=rois)
        self.init_mois(mois=mois)
        self.init_detector(detector=detector)
        self.init_tracker(tracker=tracker)
        self.init_hands_estimator(hands_estimator=hands_estimator)
        self.init_moving_object()
        self.init_data_loader(data_loader=data_loader)
        self.init_data_writer(data_writer=data_writer)
        self.init_result_writer(result_writer=result_writer)
        
        self.hands      = None
        self.mos        = []
        self.start_time = None

    # MARK: Configure

    def init_dirs(self, data_dir: str):
        """Initialize dirs."""
        if not os.path.isdir(data_dir):
            raise ValueError(f"`data_dir` must be a valid location. But got: {data_dir}")
        self.root_dir    = os.path.join(data_dir, self.dataset)
        self.configs_dir = os.path.join(self.root_dir, "configs")
        self.rmois_dir   = os.path.join(self.root_dir, "rmois")
        self.outputs_dir = os.path.join(self.root_dir, "output")
        self.video_dir   = os.path.join(self.root_dir, self.subset)

    def init_class_labels(self, class_labels: Union[ClassLabels, dict]):
        """Initialize class_labels.

        Args:
            class_labels (class_labels, dict):
                class_labels object or a config dictionary.
        """
        if isinstance(class_labels, ClassLabels):
            self.class_labels = class_labels
        elif isinstance(class_labels, dict):
            file = class_labels["file"]
            if is_json_file(file):
                self.class_labels = ClassLabels.create_from_file(file)
            elif is_basename(file):
                file              = os.path.join(self.root_dir, file)
                self.class_labels = ClassLabels.create_from_file(file)
        else:
            file              = os.path.join(self.root_dir, f"class_labels.json")
            self.class_labels = ClassLabels.create_from_file(file)
            error_console.log(f"Cannot initialize class_labels from {class_labels}. "
                              f"Attempt to load from {file}.")

    def init_rois(self, rois: Union[list[ROI], dict]):
        """Initialize rois.

        Args:
            rois (list[ROI], dict):
                List of ROIs or a config dictionary.
        """
        if is_list_of(rois, item_type=ROI):
            self.rois = rois
        elif isinstance(rois, dict):
            file = rois["file"]
            if os.path.isfile(file):
                self.rois = ROI.load_from_file(**rois)
            elif is_basename(file):
                self.rois = ROI.load_from_file(rmois_dir=self.rmois_dir, **rois)
        else:
            file      = os.path.join(self.rmois_dir, f"{self.name}.json")
            self.rois = ROI.load_from_file(file=file)
            error_console.log(f"Cannot initialize rois from {rois}. "
                              f"Attempt to load from {file}.")

    def init_mois(self, mois: Union[list[MOI], dict]):
        """Initialize rois.

        Args:
            mois (list[MOI], dict):
                List of MOIs or a config dictionary.
        """
        if is_list_of(mois, item_type=MOI):
            self.mois = mois
        elif isinstance(mois, dict):
            file = mois["file"]
            if os.path.isfile(file):
                self.mois = MOI.load_from_file(**mois)
            elif is_basename(file):
                self.mois = MOI.load_from_file(rmois_dir=self.rmois_dir, **mois)
        else:
            file      = os.path.join(self.rmois_dir, f"{self.name}.json")
            self.mois = MOI.load_from_file(file=file)
            error_console.log(f"Cannot initialize mois from {mois}. "
                              f"Attempt to load from  {file}.")

    def init_detector(self, detector: Union[BaseDetector, dict]):
        """Initialize detector.

        Args:
            detector (BaseDetector, dict):
                Detector object or a detector's config dictionary.
        """
        if isinstance(detector, BaseDetector):
            self.detector = detector
        elif isinstance(detector, dict):
            detector["class_labels"] = self.class_labels
            self.detector = DETECTORS.build(**detector)
        else:
            raise ValueError(f"Cannot initialize detector with {detector}.")
        
    def init_tracker(self, tracker: Union[BaseDetector, dict]):
        """Initialize tracker.

        Args:
            tracker (BaseTracker, dict):
                Tracker object or a tracker's config dictionary.
        """
        if isinstance(tracker, BaseTracker):
            self.tracker = tracker
        elif isinstance(tracker, dict):
            self.tracker = TRACKERS.build(**tracker)
        else:
            raise ValueError(f"Cannot initialize tracker with {tracker}.")
    
    def init_hands_estimator(self, hands_estimator: Union[HandsEstimator, dict]):
        """Initialize tracker.

        Args:
            hands_estimator (HandsEstimator, dict):
                HandsEstimator object or a config dictionary.
        """
        if isinstance(hands_estimator, HandsEstimator):
            self.hands_estimator = hands_estimator
        elif isinstance(hands_estimator, dict):
            self.hands_estimator = HandsEstimator(**hands_estimator)
        else:
            raise ValueError(f"Cannot initialize `hands_estimator` with {hands_estimator}.")
    
    def init_moving_object(self):
        """Configure the Moving Object class attribute.
        """
        cfg = self.moving_object_cfg
        Product.min_traveled_distance = cfg["min_traveled_distance"]
        Product.min_entering_distance = cfg["min_entering_distance"]
        Product.min_hit_streak        = cfg["min_hit_streak"]
        Product.max_age               = cfg["max_age"]
        Product.min_touched_landmarks = cfg["min_touched_landmarks"]
        Product.max_untouches_age     = cfg["max_untouches_age"]

    def init_data_loader(self, data_loader: Union[CVVideoLoader, dict]):
        """Initialize data loader.

        Args:
            data_loader (CVVideoLoader, dict):
                Data loader object or a data loader's config dictionary.
        """
        if isinstance(data_loader, CVVideoLoader):
            self.data_loader = data_loader
        elif isinstance(data_loader, dict):
            data = data_loader.get("data", "")
            if is_video_file(data):
                data_loader["data"] = data
            elif is_basename(data):
                data_loader["data"] = os.path.join(self.video_dir, f"{data}")
            elif is_stem(data):
                data_loader["data"] = os.path.join(self.video_dir, f"{data}.mp4")
            else:
                data_loader["data"] = os.path.join(self.video_dir, f"{self.name}.mp4")
            self.data_loader = CVVideoLoader(**data_loader)
        else:
            raise ValueError(f"Cannot initialize data loader with {data_loader}.")

    def init_data_writer(self, data_writer: Union[CVVideoWriter, dict]):
        """Initialize data writer.

        Args:
            data_writer (CVVideoWriter, dict):
                Data writer object or a data writer's config dictionary.
        """
        if isinstance(data_writer, CVVideoWriter):
            self.data_writer = data_writer
        elif isinstance(data_writer, dict):
            dst = data_writer.get("dst", "")
            if is_video_file(dst):
                data_writer["dst"] = dst
            elif is_basename(dst):
                data_writer["dst"] = os.path.join(self.outputs_dir, f"{dst}")
            elif is_stem(dst):
                data_writer["dst"] = os.path.join(self.outputs_dir, f"{dst}.mp4")
            else:
                data_writer["dst"] = os.path.join(self.outputs_dir, f"{self.name}.mp4")
            data_writer["save_image"] = self.save_image
            data_writer["save_video"] = self.save_video
            self.data_writer          = CVVideoWriter(**data_writer)

    def init_result_writer(self, result_writer: Union[AIC22RetailCheckoutWriter, dict]):
        """Initialize data writer.

        Args:
            result_writer (AICCountingWriter, dict):
                Result writer object or a result writer's config dictionary.
        """
        if isinstance(result_writer, AIC22RetailCheckoutWriter):
            self.result_writer = result_writer
        elif isinstance(result_writer, dict):
            dst = result_writer.get("dst", "")
            if os.path.isfile(dst):
                result_writer["dst"] = dst
            elif is_basename(dst):
                result_writer["dst"] = os.path.join(self.outputs_dir, f"{dst}")
            elif is_stem(dst):
                result_writer["dst"] = os.path.join(self.outputs_dir, f"{dst}.txt")
            else:
                result_writer["dst"] = os.path.join(self.outputs_dir, f"{self.name}.txt")
            result_writer["camera_name"] = result_writer.get("camera_name", self.name)
            result_writer["subset"]      = self.subset
            self.result_writer           = AIC22RetailCheckoutWriter(**result_writer)

    # MARK: Run

    def run(self):
        """Main run loop."""
        self.run_routine_start()

        with progress_bar() as pbar:
            for images, indexes, files, rel_paths in pbar.track(
                self.data_loader,
                total       = self.data_loader.batch_len(),
                description = f"[bright_yellow]{self.name}"
            ):
                if len(indexes) == 0:
                    break
                if images is None:
                    break
                    
                # NOTE: Must rotate the image 180 to correct the orientation
                if isinstance(self.data_loader, CVVideoLoader):
                    if images.ndim == 4:
                        images = [cv2.rotate(i, cv2.ROTATE_180) for i in images]
                        images = np.array(images)
                    elif images.ndim == 3:
                        images = cv2.rotate(images, cv2.ROTATE_180)
                    else:
                        continue
                    
                # NOTE: Detect batch of images
                batch_detections = self.detector.detect(indexes=indexes, images=images)
                batch_hands      = self.hands_estimator.estimate(indexes=indexes, images=images)
                
                # NOTE: Associate detections with ROIs
                for idx, detections in enumerate(batch_detections):
                    ROI.associate_detections_to_rois(detections=detections, rois=self.rois)
                    batch_detections[idx] = [i for i in detections if (i.roi_id is not None)]
                    
                # NOTE: Track batch detections
                for idx, detections in enumerate(batch_detections):
                    # Update tracker
                    self.tracker.update(detections=detections)
                    self.mos  : list[Product] = self.tracker.tracks
                    self.hands: Hands         = batch_hands[idx]
                    
                    # NOTE: Update moving objects' moving state
                    for mo in self.mos:
                        mo.update_moving_state(rois=self.rois, hands=self.hands)
                        if mo.is_confirmed and self.data is not None:
                            mo.timestamp = math.floor(
                                (mo.current_frame_index - self.tracker.min_hits)
                                / self.data["frame_rate"]
                            )
                        
                    # NOTE: Count
                    countable_mos = [o for o in self.mos if o.is_to_be_counted]
                    if self.save_results:
                        self.result_writer.write(moving_objects=countable_mos)
                    for mo in countable_mos:
                        mo.moving_state = MovingState.Counted
                    
                    # NOTE: Visualize and Debug
                    self.postprocess(image=images[idx])
            
        self.run_routine_end()

    def run_routine_start(self):
        """Perform operations when run routine starts. We start the timer."""
        self.mos                      = []
        self.start_time               = timer()
        self.result_writer.start_time = self.start_time

        if self.verbose:
            cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)

    def run_routine_end(self):
        """Perform operations when run routine ends."""
        if self.save_results:
            self.result_writer.dump()
            
        self.mos = []
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
        # NOTE: Draw Products
        [o.draw(drawing=drawing) for o in self.mos]
        # NOTE: Draw Hands
        if self.hands:
            self.hands.draw(drawing=drawing)
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
        drawing = drawing[:, :, ::-1]  # RGB to BGR
        return drawing
