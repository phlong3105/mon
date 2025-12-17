#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for FishEye8K dataset.

This module defines the FishEye8K dataset class, which is designed for object
detection tasks. It specifies the dataset's structure, including its classes,
modalities, and splits.
"""

__all__ = [
    "FishEye8K",
]

from ...core import *


@DATASETS.register(name="fisheye8k")
class FishEye8K(ImageDataset):
    """FishEye8K dataset."""
    
    _subset    : str         = "fisheye8k"
    _tasks     : list[Task]  = [Task.DETECT]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    _classes   : Classes     = Classes([
        {"name": "bus",        "id": 0, "color": [140,  24, 143]},
        {"name": "bike",       "id": 1, "color": [122,  35,   2]},
        {"name": "car",        "id": 2, "color": [ 49,   3, 150]},
        {"name": "pedestrian", "id": 3, "color": [ 81, 120, 228]},
        {"name": "truck",      "id": 4, "color": [ 72, 153, 152]},
    ])
