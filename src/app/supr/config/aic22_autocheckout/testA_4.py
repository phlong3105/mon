#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the configuration for `testA_4` video."""

from __future__ import annotations

import cv2

import mon
from supr.globals import DATA_DIR

# region Config

_current_dir    = mon.Path(__file__).absolute().parent
camera          = "auto_checkout_camera"

id_             = 1          # Camera's unique ID.
subset          = "testA"    # Subset. One of: ['testA', 'testB'].
name            = "testA_4"  # Camera name = data name.
root            = DATA_DIR/"aic22-autocheckout"  # Root directory of the data.
rois            = [
    {
        "id_"       : 1,
        "points"    : [
            [600.0 , 300.0],
            [1350.0, 300.0],
            [1350.0, 900.0],
            [600.0 , 900.0]
        ],
        "shape_type": "polygon",
    },
]      # Region of Interest.
mois            = [
    {
        "id_"               : 1,
        "points"            : [],
        "shape_type"        : "line",
        "distance_function" : "hausdorff",  # Distance function.
        "distance_threshold": 200,          # Maximum distance for counting a track.
        "angle_threshold"   : 45,           # Maximum angle for counting a track.
    },
]      # Movement of Interest.
classlabels     = _current_dir/"classlabels.json"
num_classes     = 117
image_loader    = {
    "source"        : root/subset/f"{name}.mp4",  # Data source.
    "max_samples"   : None,   # The maximum number of datapoints to be processed.
    "batch_size"    : 8,      # The number of samples in a single forward pass
    "to_rgb"        : True,   # Convert the image from BGR to RGB.
    "to_tensor"     : False,  # Convert the image from :class:`numpy.ndarray` to :class:`torch.Tensor`.
    "normalize"     : False,  # Normalize the image to [0.0, 1.0].
    "api_preference": cv2.CAP_FFMPEG,
    "verbose"       : False,  # Verbosity.
}
image_writer    = {
    "destination": name,          # A destination directory to save images.
    "image_size" : [1080, 1920],  # A desired output size of shape HW.
    "frame_rate" : 30,            # A frame rate of the output video.
    "fourcc"     : "mp4v",        # Video codec. One of: ['mp4v', 'xvid', 'mjpg', 'wmv'].
    "save_image" : False,         # If True, save each image separately.
    "denormalize": False,         # If True, convert image to [0, 255].
    "verbose"    : False,         # Verbosity.
}
result_writer   = {
    "destination": name,    # A path to the counting results file.
    "camera_name": name,    # A camera name.
    "subset"     : subset,  # The moment when the TexIO is initialized.
    "exclude"    : [116],   # A list of class ID to exclude from writing.
}
moving_object   = {
    "name"                 : "product",  # An object type.
    "min_entering_distance": 1,   # Minimum distance when an object enters the ROI to be Confirmed.
    "min_traveled_distance": 50,  # Minimum distance between first trajectory point with last trajectory point.
    "min_hit_streak"       : 3,   # Minimum number of consecutive frames that track appears.
    "max_age"              : 1,   # Maximum frame to wait until a dead track can be counted.
    "min_touched_landmarks": 1,   # Minimum hand landmarks touching the object so that it is considered hand-handling.
    "min_confirms"         : 3,   # Minimum frames that the object is considered for counting.
}
detector        = {
    "name"          : "yolov8",   # A detector name.
    "config"        : "yolov8x",  # A detector model's config.
    "weight"        : _current_dir/"yolov8x-aic22-autocheckout-117-1920.pt",
    "image_size"    : 640,        # The desired model's input size in HW format.
    "classlabels"   : classlabels,
    "conf_threshold": 0.3,        # An object confidence threshold.
    "iou_threshold" : 0.8,        # An IOU threshold for NMS.
    "max_detections": 300,        # Maximum number of detections/image.
    "device"        : 0,          # Cuda device, i.e. 0 or 0,1,2,3 or cpu.
    "to_instance"   : True,       # If True, wrap the predictions to a list of :class:`supr.data.instance.Instance` object.
}
tracker         = {
    "name"         : "sort",                 # Name of the tracker.
    "max_age"      : 1,                      # Maximum number of frame keep the object before deleting.
    "min_hits"     : 3,                      # Number of frames, which have matching bounding bbox of the detected object before the object is considered becoming the track.
    "iou_threshold": 0.3,                    # An Intersection-over-Union threshold between two tracks.
    "motion_type"  : "kf_bbox_motion",       # A motion model.
    "object_type"  : moving_object["name"],  # An object type
}
hands_estimator = {
    "static_image_mode"       : False,  # Whether to treat the input images as a batch of static and possibly unrelated images, or a video stream.
    "max_num_hands"           : 2,      # Maximum number of hands to detect.
    "model_complexity"        : 1,      # Complexity of the hand landmark model: 0 or 1.
    "min_detection_confidence": 0.2,    # Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
    "min_tracking_confidence" : 0.2,    # Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be considered tracked successfully.
}
save_image      = False      # Save processing images?
save_video      = False      # Save processing video?
save_results    = True       # Save counting results.
verbose         = True       # Verbosity.

# endregion
