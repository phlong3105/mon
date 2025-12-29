#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ExDark dataset.

This module implements the ExDark dataset for nighttime object enhancement and
detection.

References:
    - Data: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
"""

__all__ = [
    "ExDark",
]

from ....meta import *


@DATASETS.register(name="exdark")
class ExDark(ImageDataset):
    """ExDark dataset."""
    
    _subset    : str         = "exdark"
    _tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DETECT]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classlist : ClassList   = ClassList([
        {"name": "Bicycle"  , "id":  1, "coco80_id":  2, "color": [138, 183,  33]},
        {"name": "Boat"     , "id":  2, "coco80_id":  9, "color": [ 19,  64,  83]},
        {"name": "Bottle"   , "id":  3, "coco80_id": 40, "color": [139, 160,   1]},
        {"name": "Bus"      , "id":  4, "coco80_id":  6, "color": [140,  24, 143]},
        {"name": "Car"      , "id":  5, "coco80_id":  3, "color": [ 49,   3, 150]},
        {"name": "Cat"      , "id":  6, "coco80_id": 16, "color": [ 41, 174, 251]},
        {"name": "Chair"    , "id":  7, "coco80_id": 57, "color": [ 94, 173,  36]},
        {"name": "Cup"      , "id":  8, "coco80_id": 42, "color": [ 28,  47,  55]},
        {"name": "Dog"      , "id":  9, "coco80_id": 17, "color": [ 21,   8, 251]},
        {"name": "Motorbike", "id": 10, "coco80_id":  4, "color": [122,  35,   2]},
        {"name": "People"   , "id": 11, "coco80_id":  1, "color": [ 81, 120, 228]},
        {"name": "Table"    , "id": 12, "coco80_id": 61, "color": [216, 147, 179]},
    ])
