#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the configuration for 'testA_3' video."""

from __future__ import annotations

import cv2

import mon
from supr.globals import DATA_DIR

# region Config

_current_dir = mon.Path(__file__).absolute().parent

id_           = 1          # Camera's unique ID.
subset        = "testA"    # Subset. One of: [`testA`, `testB`].
name          = "testA_3"  # Camera name = data name.
root          = DATA_DIR/"aic23-checkout"  # Root directory of the data.
rois          = [
    {
        "id_"       : 1,
        "points"    : [
            [550.0,  250.0],
            [1400.0, 250.0],
            [1400.0, 1000.0],
            [550.0,  1000.0]
        ],
        "shape_type": "polygon",
    },
]  # Region of Interest.
mois          = [
    {
        "id_"               : 1,
        "points"            : [],
        "shape_type"        : "line",
        "distance_function" : "hausdorff",  # Distance function.
        "distance_threshold": 200,          # Maximum distance for counting a track.
        "angle_threshold"   : 45,           # Maximum angle for counting a track.
    },
]  # Movement of Interest.
classlabels   = _current_dir/"classlabels.json"
num_classes   = 117
image_loader  = {
    "source"        : root/subset/"inpainting"/f"{name}.mp4",  # Data source.
    "max_samples"   : None,   # The maximum number of datapoints to be processed.
    "batch_size"    : 8,      # The number of samples in a single forward pass
    "to_rgb"        : True,   # Convert the image from BGR to RGB.
    "to_tensor"     : False,  # Convert the image from :class:`numpy.ndarray` to :class:`torch.Tensor`.
    "normalize"     : False,  # Normalize the image to [0.0, 1.0].
    "api_preference": cv2.CAP_FFMPEG,
    "verbose"       : False,  # Verbosity.
}
image_writer  = {
    "destination": name,
    "image_size" : [1080, 1920],
    "frame_rate" : 30,
    "fourcc"     : "mp4v",
    "save_image" : False,
    "denormalize": False,
    "verbose"    : False,
}
result_writer = {
    "destination": name,
    "camera_name": name,
    "subset"     : subset,
    "exclude"    : [116],
}
detector      = {
    "name"          : "yolov8",
    "config"        : "yolov8x",
    "weight"        : _current_dir/"aic23-checkout-synthetic117-640.pt",
    "image_size"    : 640,
    "conf_threshold": 0.2,
    "iou_threshold" : 0.5,
    "max_detections": 300,
    "device"        : 0,
}
tracker       = {
    "name"         : "sort",           # Name of the tracker.
    "max_age"      : 1,                # Maximum number of frame keep the object before deleting.
    "min_hits"     : 3,                # Number of frames, which have matching bounding bbox of the detected object before the object is considered becoming the track.
    "iou_threshold": 0.3,              # An Intersection-over-Union threshold between two tracks.
    "motion_type"  : "kf_box_motion",  # A motion model.
    "object_type"  : "product",        # An object type
}
moving_object = {}
save_image    = False  # Save processing images?
save_video    = False  # Save processing video?
save_results  = True   # Save counting results.
verbose       = True   # Verbosity.

# endregion
