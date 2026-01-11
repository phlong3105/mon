#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FishEye8K dataset.

This module implements the FishEye8K dataset for fisheye object detection.
"""

__all__ = [
    "FishEye8K",
]

from ...api import *


@DATASETS.register()
class FishEye8K(ImageDataset):
    """FishEye8K dataset."""
    
    _name      : str         = "fisheye8k"
    _tasks     : list[Task]  = [Task.DETECT]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    _classlist : ClassList   = ClassList([
        {"name": "bus",        "id": 0, "color": [140,  24, 143]},
        {"name": "bike",       "id": 1, "color": [122,  35,   2]},
        {"name": "car",        "id": 2, "color": [ 49,   3, 150]},
        {"name": "pedestrian", "id": 3, "color": [ 81, 120, 228]},
        {"name": "truck",      "id": 4, "color": [ 72, 153, 152]},
    ])
