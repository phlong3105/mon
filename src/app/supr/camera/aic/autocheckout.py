#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the automated retail checkout camera."""

from __future__ import annotations

__all__ = [
    "AutoCheckoutCamera",
]

import uuid
from timeit import default_timer as timer
from typing import Any

import cv2
import numpy as np

import mon
from mon.globals import DETECTORS, MovingState, OBJECTS, TRACKERS
from supr import io, obj, rmoi
from supr.camera.base import Camera
from supr.globals import CAMERAS


# region AutoCheckoutCamera

@CAMERAS.register(name="auto_checkout_camera")
class AutoCheckoutCamera(Camera):
    """Automated Retail Checkout Camera.
    
    Args:
        rois: A list of :class:`supr.rmoi.ROI` objects.
        mois: A list of class:`supr.rmoi.MOI` objects.
        classlabels: A list of all the class-labels defined in a dataset.
        num_classes: A number of classes.
        image_loader: An image loader object.
        image_writer An image writer object.
        detector: A :class:`supr.detect.base.Detector` object.
        tracker: A :class:`supr.track.base.Tracker` object.
        moving_object:
        
    See Also: :class:`supr.camera.base.Camera`.
    """
    
    def __init__(
        self,
        root           : mon.Path,
        subset         : str,
        name           : str,
        rois           : Any,
        mois           : Any,
        classlabels    : Any,
        num_classes    : int | None,
        image_loader   : Any,
        image_writer   : Any,
        result_writer  : Any,
        moving_object  : Any,
        detector       : Any,
        tracker        : Any,
        tray_detector  : Any       = None,
        hands_estimator: Any       = None,
        id_            : int | str = uuid.uuid4().int,
        save_image     : bool      = False,
        save_video     : bool      = False,
        save_result    : bool      = True,
        verbose        : bool      = False,
        *args, **kwargs
    ):
        super().__init__(
            id_          = id_,
            root         = root,
            subset       = subset,
            name         = name,
            image_loader = image_loader,
            image_writer = image_writer,
            save_image   = save_image,
            save_video   = save_video,
            save_result  = save_result,
            verbose      = verbose
        )
        self.rois            = rmoi.ROI.from_value(value=rois)
        self.mois            = rmoi.MOI.from_value(value=mois)
        self.classlabels     = mon.ClassLabels.from_value(value=classlabels)
        self.num_classes     = num_classes or len(self.classlabels) \
                               if self.classlabels is not None else 0
        self.result_writer   = result_writer
        self.detector        = detector
        self.tracker         = tracker
        self.tray_detector   = tray_detector
        self.hands_estimator = hands_estimator
        self.start_time      = None
        self.hands           = None
        self.moving_objects  = []
        self.init_moving_object(moving_object)
        
    @property
    def result_writer(self) -> io.AIC23AutoCheckoutWriter:
        return self._result_writer
    
    @result_writer.setter
    def result_writer(self, result_writer: Any):
        if not self.save_result:
            self._result_writer = None
        elif isinstance(result_writer, io.AIC23AutoCheckoutWriter):
            self._result_writer = result_writer
        elif isinstance(result_writer, dict):
            destination = mon.Path(result_writer.get("destination", None))
            if destination.is_dir():
                destination = destination / f"{self.name}.txt"
            elif destination.is_basename() or destination.is_stem():
                destination = self.result_dir / f"{destination}.txt"
            if not destination.is_txt_file(exist=False):
                raise ValueError(
                    f"destination must be a valid path to a .txt file, but got "
                    f"{destination}."
                )
            result_writer["destination"] = destination
            self._result_writer = io.AIC23AutoCheckoutWriter(**result_writer)
        else:
            raise ValueError(
                f"Cannot initialize result writer with {result_writer}."
            )
    
    @property
    def detector(self) -> mon.Detector:
        return self._detector
    
    @detector.setter
    def detector(self, detector: Any):
        if isinstance(detector, mon.Detector):
            self._detector = detector
        elif isinstance(detector, dict):
            detector["classlabels"] = self.classlabels
            self._detector = DETECTORS.build(**detector)
        else:
            raise ValueError(f"Cannot initialize detector with {detector}.")
    
    @property
    def tracker(self) -> mon.Tracker:
        return self._tracker
    
    @tracker.setter
    def tracker(self, tracker: Any):
        if isinstance(tracker, mon.Tracker):
            self._tracker = tracker
        elif isinstance(tracker, dict):
            self._tracker = TRACKERS.build(**tracker)
        else:
            raise ValueError(f"Cannot initialize tracker with {tracker}.")
    
    @property
    def hands_estimator(self) -> obj.HandsEstimator | None:
        return self._hands_estimator
    
    @hands_estimator.setter
    def hands_estimator(self, hands_estimator: Any):
        if hands_estimator is None:
            self._hands_estimator = None
        elif isinstance(hands_estimator, obj.HandsEstimator):
            self._hands_estimator = hands_estimator
        elif isinstance(hands_estimator, dict):
            self._hands_estimator = obj.HandsEstimator(**hands_estimator)
        else:
            raise ValueError(
                f"Cannot initialize hands estimator with {hands_estimator}."
            )

    @property
    def tray_detector(self) -> mon.Detector:
        return self._tray_detector

    @tray_detector.setter
    def tray_detector(self, tray_detector: Any):
        if isinstance(tray_detector, mon.Detector):
            self._tray_detector = tray_detector
        elif isinstance(tray_detector, dict):
            self._tray_detector = DETECTORS.build(**tray_detector)
        else:
            raise ValueError(f"Cannot initialize tray detector with {tray_detector}.")

    # noinspection PyMethodMayBeStatic
    def init_moving_object(self, moving_object: dict | None):
        if moving_object is None:
            return
        object_class = OBJECTS.get(moving_object["name"])
        if object_class is None:
            return
        object_class.min_traveled_distance = moving_object.get("min_traveled_distance", 0.0)
        object_class.min_entering_distance = moving_object.get("min_entering_distance", 100.0)
        object_class.min_hit_streak        = moving_object.get("min_hit_streak"       , 10)
        object_class.max_age               = moving_object.get("max_age"              , 1)
        object_class.min_touched_landmarks = moving_object.get("min_touched_landmarks", 1)
        object_class.min_confirms          = moving_object.get("min_confirms"         , 3)
        object_class.min_counting_distance = moving_object.get("min_counting_distance", 10)
        
    def on_run_start(self):
        """Called at the beginning of run loop."""
        self.moving_objects = []
        self.start_time     = timer()
        self.result_writer.start_time = self.start_time
        mon.mkdirs(
            paths    = [self.subset_dir, self.result_dir],
            parents  = True,
            exist_ok = True
        )
        if self.verbose:
            cv2.namedWindow(self.name, cv2.WINDOW_KEEPRATIO)
        
    def run(self):
        """Main run loop."""
        self.on_run_start()
        
        with mon.get_progress_bar() as pbar:
            for images, indexes, files, rel_paths in pbar.track(
                self.image_loader,
                total       = self.image_loader.batch_len(),
                description = f"[bright_yellow]{self.name}"
            ):
                if len(indexes) == 0 or images is None:
                    break
                
                # Detect Trays
                if self.tray_detector:
                    batch_trays = self.tray_detector.detect(indexes=indexes, images=images)
                else:
                    batch_trays = None
                
                # Detect Objects
                batch_instances = self.detector.detect(indexes=indexes, images=images)
                if self.hands_estimator:
                    batch_hands = self.hands_estimator.estimate(indexes=indexes, images=images)
                else:
                    batch_hands = None
                    
                # Process batches
                for idx, instances in enumerate(batch_instances):
                    # Update ROI (optional)
                    if batch_trays:
                        tray = batch_trays[idx]
                        if len(tray) > 0:
                            tray = np.array(tray[0, 0:4])
                            tray = mon.get_bbox_corners_points(bbox=tray)
                            self.rois[0].points = tray
                            
                    # Assign ROI
                    roi_ids = [rmoi.get_roi_for_box(bbox=i.bbox, rois=self.rois) for i in instances]
                    for instance, roi_id in zip(instances, roi_ids):
                        instance.roi_id = roi_id
                    
                    # Track
                    self.tracker.update(instances=instances)
                    self.moving_objects: list[obj.Product] = self.tracker.tracks
                    self.hands = batch_hands[idx] if batch_hands else None
                    
                    # Update moving objects' moving state
                    for mo in self.moving_objects:
                        mo.update_moving_state(rois=self.rois, hands=self.hands)
                        if mo.is_to_be_counted:
                            # fps          = getattr(self.image_loader, "fps", 30)
                            # timestamp    = (mo.current.frame_index - self.tracker.min_hits) / fps
                            # mo.timestamp = mon.math.floor(timestamp)
                            mo.timestamp = mo.current.frame_index
                    
                    # Count
                    countable = [o for o in self.moving_objects if o.is_to_be_counted]
                    if self.save_result:
                        self.result_writer.append_results(products=countable)
                    for mo in countable:
                        mo.moving_state = MovingState.COUNTED
                    self.run_step_end(index=indexes[idx], image=images[idx])
                
        self.on_run_end()
    
    def run_step_end(self, index: int, image: np.ndarray):
        """Perform some postprocessing operations when a run step end."""
        if not (self.verbose or self.save_image or self.save_video):
            return
        elapsed_time = timer() - self.start_time
        image = self.draw(index=index, image=image, elapsed_time=elapsed_time)
        if self.save_video:
            self.image_writer.write(image=image)
        if self.verbose:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.name, image)
            cv2.waitKey(1)
    
    def on_run_end(self):
        """Called at the end of run loop."""
        if self.save_result:
            self.result_writer.write_to_file()
        self.mos = []
        if self.verbose:
            cv2.destroyAllWindows()
    
    def draw(
        self,
        index       : int,
        image       : np.ndarray,
        elapsed_time: float
    ) -> np.ndarray:
        """Visualize the results on the image.

        Args:
            index: Current frame index.
            image: Drawing canvas.
            elapsed_time: Elapsed time per iteration.
        """
        # Draw ROI
        for r in self.rois:
            image = r.draw(image=image)
        # Draw MOIs
        for m in self.mois:
            image = m.draw(image=image)
        # Draw Products
        for o in self.moving_objects:
            image = o.draw(image=image)
        # Draw Hands
            if self.hands:
                self.hands.draw(image=image)
        # Draw frame index
        fps  = index / elapsed_time
        text = f"Frame: {index}: {format(elapsed_time, '.3f')}s ({format(fps, '.1f')} fps)"
        cv2.rectangle(
            img       = image,
            pt1       = (10,   0),
            pt2       = (600, 40),
            color     = (0, 0, 0),
            thickness = -1
        )
        cv2.putText(
            img       = image,
            text      = text,
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1.0,
            org       = (20, 30),
            color     = (255, 255, 255),
            thickness = 2
        )
        return image
    
# endregion
