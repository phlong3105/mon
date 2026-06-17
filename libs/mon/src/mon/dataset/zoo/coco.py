#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COCO Datasets.

This module provides the COCO datasets for object detection and instance
segmentation.
"""

from __future__ import annotations

__all__ = [
    "COCO",
]

from mon.core import Class, ClassList, DATASETS, Split, Task
from mon.dataset.base import (
    DatasetRegisterMixin,
    ImageDataset,
    ImageModality,
    ModalityList,
)


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="coco")
class COCO(ImageDataset, DatasetRegisterMixin):
    """COCO dataset."""

    name: str = "coco"
    tasks: list[Task] = [Task.DETECT]
    dirname: str = "coco"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
    ])
    classes: ClassList = ClassList([
        Class(name="background",    id=0,  category="background",  color=(  0,   0,   0)),
        Class(name="person",        id=1,  category="person",      color=( 81, 120, 228)),
        Class(name="bicycle",       id=2,  category="vehicle",     color=(138, 183,  33)),
        Class(name="car",           id=3,  category="vehicle",     color=( 49,   3, 150)),
        Class(name="motorcycle",    id=4,  category="vehicle",     color=(122,  35,   2)),
        Class(name="airplane",      id=5,  category="airplane",    color=(165, 168, 193)),
        Class(name="bus",           id=6,  category="vehicle",     color=(140,  24, 143)),
        Class(name="train",         id=7,  category="vehicle",     color=(179, 165, 212)),
        Class(name="truck",         id=8,  category="vehicle",     color=( 72, 153, 152)),
        Class(name="boat",          id=9,  category="vehicle",     color=( 19,  64,  83)),
        Class(name="traffic light", id=10, category="outdoor",     color=(122,  40,  57)),
        Class(name="fire hydrant",  id=11, category="outdoor",     color=(219,  42, 205)),
        Class(name="stop sign",     id=12, category="outdoor",     color=( 15,  90, 125)),
        Class(name="parking meter", id=13, category="outdoor",     color=(187,  80,  10)),
        Class(name="bench",         id=14, category="outdoor",     color=( 76, 226, 142)),
        Class(name="bird",          id=15, category="animal",      color=( 24,  56,  34)),
        Class(name="cat",           id=16, category="animal",      color=( 41, 174, 251)),
        Class(name="dog",           id=17, category="animal",      color=( 21,   8, 251)),
        Class(name="horse",         id=18, category="animal",      color=(106, 128, 177)),
        Class(name="sheep",         id=19, category="animal",      color=(147,  90, 131)),
        Class(name="cow",           id=20, category="animal",      color=( 65, 159, 189)),
        Class(name="elephant",      id=21, category="animal",      color=(129,  70,  30)),
        Class(name="bear",          id=22, category="animal",      color=( 38, 181,  29)),
        Class(name="zebra",         id=23, category="animal",      color=(189, 238, 167)),
        Class(name="giraffe",       id=24, category="animal",      color=(173, 154, 136)),
        Class(name="backpack",      id=25, category="accessory",   color=(205, 104,  95)),
        Class(name="umbrella",      id=26, category="accessory",   color=(163,  13, 178)),
        Class(name="handbag",       id=27, category="accessory",   color=(156,  84, 167)),
        Class(name="tie",           id=28, category="accessory",   color=( 10, 146, 166)),
        Class(name="suitcase",      id=29, category="accessory",   color=(176, 137,  78)),
        Class(name="frisbee",       id=30, category="sports",      color=(190, 118,  41)),
        Class(name="skis",          id=31, category="sports",      color=(159, 178,  24)),
        Class(name="snowboard",     id=32, category="sports",      color=(107,  85, 171)),
        Class(name="sports ball",   id=33, category="sports",      color=(186, 223, 221)),
        Class(name="kite",          id=34, category="sports",      color=(142, 218,  56)),
        Class(name="baseball bat",  id=35, category="sports",      color=( 82, 128, 254)),
        Class(name="baseball glove",id=36, category="sports",      color=( 64, 200, 173)),
        Class(name="skateboard",    id=37, category="sports",      color=(112,  66,  51)),
        Class(name="surfboard",     id=38, category="sports",      color=( 47, 131, 231)),
        Class(name="tennis racket", id=39, category="sports",      color=( 37,  70, 244)),
        Class(name="bottle",        id=40, category="kitchen",     color=(139, 160,   1)),
        Class(name="wine glass",    id=41, category="kitchen",     color=(103,  32,  74)),
        Class(name="cup",           id=42, category="kitchen",     color=( 28,  47,  55)),
        Class(name="fork",          id=43, category="kitchen",     color=(219,  18, 203)),
        Class(name="knife",         id=44, category="kitchen",     color=( 41, 125, 194)),
        Class(name="spoon",         id=45, category="kitchen",     color=( 76, 180, 131)),
        Class(name="bowl",          id=46, category="kitchen",     color=(143,   4, 187)),
        Class(name="banana",        id=47, category="food",        color=(232, 188,  11)),
        Class(name="apple",         id=48, category="food",        color=(119, 177,  17)),
        Class(name="sandwich",      id=49, category="food",        color=( 55, 214, 248)),
        Class(name="orange",        id=50, category="food",        color=(100, 254,  62)),
        Class(name="broccoli",      id=51, category="food",        color=( 15,  12,  37)),
        Class(name="carrot",        id=52, category="food",        color=(105,  24,  82)),
        Class(name="hot dog",       id=53, category="food",        color=(192, 102, 113)),
        Class(name="pizza",         id=54, category="food",        color=(242,  21, 163)),
        Class(name="donut",         id=55, category="food",        color=( 13,  42, 240)),
        Class(name="cake",          id=56, category="food",        color=( 83, 228, 215)),
        Class(name="chair",         id=57, category="furniture",   color=( 94, 173,  36)),
        Class(name="couch",         id=58, category="furniture",   color=( 63,  48,  10)),
        Class(name="potted plant",  id=59, category="furniture",   color=(199,  53,   7)),
        Class(name="bed",           id=60, category="furniture",   color=(174,  28, 109)),
        Class(name="dining table",  id=61, category="furniture",   color=(216, 147, 179)),
        Class(name="toilet",        id=62, category="furniture",   color=( 36, 181, 193)),
        Class(name="tv",            id=63, category="electronics", color=( 54,  95, 132)),
        Class(name="laptop",        id=64, category="electronics", color=(142,  43,  85)),
        Class(name="mouse",         id=65, category="electronics", color=(150, 175,  16)),
        Class(name="remote",        id=66, category="electronics", color=(125, 179, 231)),
        Class(name="keyboard",      id=67, category="electronics", color=(249,  95, 141)),
        Class(name="cell phone",    id=68, category="electronics", color=(105,  24, 191)),
        Class(name="microwave",     id=69, category="appliance",   color=(135,  51,  82)),
        Class(name="oven",          id=70, category="appliance",   color=( 69,  21,  20)),
        Class(name="toaster",       id=71, category="appliance",   color=( 67,  30, 125)),
        Class(name="sink",          id=72, category="appliance",   color=(135, 205,  67)),
        Class(name="refrigerator",  id=73, category="appliance",   color=( 35, 219,  70)),
        Class(name="book",          id=74, category="book",        color=( 80, 203,  31)),
        Class(name="clock",         id=75, category="clock",       color=( 26,  26, 253)),
        Class(name="vase",          id=76, category="furniture",   color=(134, 219,  70)),
        Class(name="scissors",      id=77, category="kitchen",     color=(  0, 132, 236)),
        Class(name="teddy bear",    id=78, category="animal",      color=(134,  81,   4)),
        Class(name="hair drier",    id=79, category="clothing",    color=(123,  68, 172)),
        Class(name="toothbrush",    id=80, category="clothing",    color=( 58, 228, 226)),
    ])

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
