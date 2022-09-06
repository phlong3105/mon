#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
COCO dataset and datamodule.
"""

from __future__ import annotations

import argparse

from matplotlib import pyplot as plt

from one.constants import *
from one.core import *
from one.data import ClassLabels
from one.data import ClassLabels_
from one.data import YOLODetectionDataset
from one.data import DataModule
from one.data import Image
from one.plot import draw_box
from one.plot import imshow
from one.vision.shape import box_cxcywh_norm_to_xyxy
from one.vision.transformation import Resize


# H1: - Module -----------------------------------------------------------------

coco17_80_classlabels = [
    { "id": 0 ,  "supercategory": "background", "name": "background"    , "color": [0  , 0  , 0  ] },
    { "id": 1 ,  "supercategory": "person",     "name": "person"        , "color": [81 , 120, 228] },
    { "id": 2 ,  "supercategory": "vehicle",    "name": "bicycle"       , "color": [138, 183, 33 ] },
    { "id": 3 ,  "supercategory": "vehicle",    "name": "car"           , "color": [49 , 3  , 150] },
    { "id": 4 ,  "supercategory": "vehicle",    "name": "motorcycle"    , "color": [122, 35 , 2  ] },
    { "id": 5 ,  "supercategory": "vehicle",    "name": "airplane"      , "color": [165, 168, 193] },
    { "id": 6 ,  "supercategory": "vehicle",    "name": "bus"           , "color": [140, 24 , 143] },
    { "id": 7 ,  "supercategory": "vehicle",    "name": "train"         , "color": [179, 165, 212] },
    { "id": 8 ,  "supercategory": "vehicle",    "name": "truck"         , "color": [72 , 153, 152] },
    { "id": 9 ,  "supercategory": "vehicle",    "name": "boat"          , "color": [19 , 64 , 83 ] },
    { "id": 10 , "supercategory": "outdoor",    "name": "traffic light" , "color": [122, 40 , 57 ] },
    { "id": 11,  "supercategory": "outdoor",    "name": "fire hydrant"  , "color": [219, 42 , 205] },
    { "id": 12,  "supercategory": "outdoor",    "name": "stop sign"     , "color": [15 , 90 , 125] },
    { "id": 13,  "supercategory": "outdoor",    "name": "parking meter" , "color": [187, 80 , 10 ] },
    { "id": 14,  "supercategory": "outdoor",    "name": "bench"         , "color": [76 , 226, 142] },
    { "id": 15,  "supercategory": "animal",     "name": "bird"          , "color": [24 , 56 , 34 ] },
    { "id": 16,  "supercategory": "animal",     "name": "cat"           , "color": [41 , 174, 251] },
    { "id": 17,  "supercategory": "animal",     "name": "dog"           , "color": [21 , 8  , 251] },
    { "id": 18,  "supercategory": "animal",     "name": "horse"         , "color": [106, 128, 177] },
    { "id": 19,  "supercategory": "animal",     "name": "sheep"         , "color": [147, 90 , 131] },
    { "id": 20,  "supercategory": "animal",     "name": "cow"           , "color": [65 , 159, 189] },
    { "id": 21,  "supercategory": "animal",     "name": "elephant"      , "color": [129, 70 , 30 ] },
    { "id": 22,  "supercategory": "animal",     "name": "bear"          , "color": [38 , 181, 29 ] },
    { "id": 23,  "supercategory": "animal",     "name": "zebra"         , "color": [189, 238, 167] },
    { "id": 24,  "supercategory": "animal",     "name": "giraffe"       , "color": [173, 154, 136] },
    { "id": 25,  "supercategory": "accessory",  "name": "backpack"      , "color": [205, 104, 95 ] },
    { "id": 26,  "supercategory": "accessory",  "name": "umbrella"      , "color": [163, 13 , 178] },
    { "id": 27,  "supercategory": "accessory",  "name": "handbag"       , "color": [156, 84 , 167] },
    { "id": 28,  "supercategory": "accessory",  "name": "tie"           , "color": [10 , 146, 166] },
    { "id": 29,  "supercategory": "accessory",  "name": "suitcase"      , "color": [176, 137, 78 ] },
    { "id": 30,  "supercategory": "sports",     "name": "frisbee"       , "color": [190, 118, 41 ] },
    { "id": 31,  "supercategory": "sports",     "name": "skis"          , "color": [159, 178, 24 ] },
    { "id": 32,  "supercategory": "sports",     "name": "snowboard"     , "color": [107, 85 , 171] },
    { "id": 33,  "supercategory": "sports",     "name": "sports ball"   , "color": [186, 223, 221] },
    { "id": 34,  "supercategory": "sports",     "name": "kite"          , "color": [142, 218, 56 ] },
    { "id": 35,  "supercategory": "sports",     "name": "baseball bat"  , "color": [82 , 128, 254] },
    { "id": 36,  "supercategory": "sports",     "name": "baseball glove", "color": [64 , 200, 173] },
    { "id": 37,  "supercategory": "sports",     "name": "skateboard"    , "color": [112, 66 , 51 ] },
    { "id": 38,  "supercategory": "sports",     "name": "surfboard"     , "color": [47 , 131, 231] },
    { "id": 39,  "supercategory": "sports",     "name": "tennis racket" , "color": [37 , 70 , 244] },
    { "id": 40,  "supercategory": "kitchen",    "name": "bottle"        , "color": [139, 160, 1  ] },
    { "id": 41,  "supercategory": "kitchen",    "name": "wine glass"    , "color": [103, 32 , 74 ] },
    { "id": 42,  "supercategory": "kitchen",    "name": "cup"           , "color": [28 , 47 , 55 ] },
    { "id": 43,  "supercategory": "kitchen",    "name": "fork"          , "color": [219, 18 , 203] },
    { "id": 44,  "supercategory": "kitchen",    "name": "knife"         , "color": [41 , 125, 194] },
    { "id": 45,  "supercategory": "kitchen",    "name": "spoon"         , "color": [76 , 180, 131] },
    { "id": 46,  "supercategory": "kitchen",    "name": "bowl"          , "color": [143, 4  , 187] },
    { "id": 47,  "supercategory": "food",       "name": "banana"        , "color": [232, 188, 11 ] },
    { "id": 48,  "supercategory": "food",       "name": "apple"         , "color": [119, 177, 17 ] },
    { "id": 49,  "supercategory": "food",       "name": "sandwich"      , "color": [55 , 214, 248] },
    { "id": 50,  "supercategory": "food",       "name": "orange"        , "color": [100, 254, 62 ] },
    { "id": 51,  "supercategory": "food",       "name": "broccoli"      , "color": [15 , 12 , 37 ] },
    { "id": 52,  "supercategory": "food",       "name": "carrot"        , "color": [105, 24 , 82 ] },
    { "id": 53,  "supercategory": "food",       "name": "hot dog"       , "color": [192, 102, 113] },
    { "id": 54,  "supercategory": "food",       "name": "pizza"         , "color": [242, 21 , 163] },
    { "id": 55,  "supercategory": "food",       "name": "donut"         , "color": [13 , 42 , 240] },
    { "id": 56,  "supercategory": "food",       "name": "cake"          , "color": [83 , 228, 215] },
    { "id": 57,  "supercategory": "furniture",  "name": "chair"         , "color": [94 , 173, 36 ] },
    { "id": 58,  "supercategory": "furniture",  "name": "couch"         , "color": [63 , 48 , 10 ] },
    { "id": 59,  "supercategory": "furniture",  "name": "potted plant"  , "color": [199, 53 , 7  ] },
    { "id": 60,  "supercategory": "furniture",  "name": "bed"           , "color": [174, 28 , 109] },
    { "id": 61,  "supercategory": "furniture",  "name": "dining table"  , "color": [216, 147, 179] },
    { "id": 62,  "supercategory": "furniture",  "name": "toilet"        , "color": [36 , 181, 193] },
    { "id": 63,  "supercategory": "electronic", "name": "tv"            , "color": [54 , 95 , 132] },
    { "id": 64,  "supercategory": "electronic", "name": "laptop"        , "color": [142, 43 , 85 ] },
    { "id": 65,  "supercategory": "electronic", "name": "mouse"         , "color": [150, 175, 16 ] },
    { "id": 66,  "supercategory": "electronic", "name": "remote"        , "color": [125, 179, 231] },
    { "id": 67,  "supercategory": "electronic", "name": "keyboard"      , "color": [249, 95 , 141] },
    { "id": 68,  "supercategory": "electronic", "name": "cell phone"    , "color": [105, 24 , 191] },
    { "id": 69,  "supercategory": "appliance",  "name": "microwave"     , "color": [135, 51 , 82 ] },
    { "id": 70,  "supercategory": "appliance",  "name": "oven"          , "color": [69 , 21 , 20 ] },
    { "id": 71,  "supercategory": "appliance",  "name": "toaster"       , "color": [67 , 30 , 125] },
    { "id": 72,  "supercategory": "appliance",  "name": "sink"          , "color": [135, 205, 67 ] },
    { "id": 73,  "supercategory": "appliance",  "name": "refrigerator"  , "color": [35 , 219, 70 ] },
    { "id": 74,  "supercategory": "indoor",     "name": "book"          , "color": [80 , 203, 31 ] },
    { "id": 75,  "supercategory": "indoor",     "name": "clock"         , "color": [26 , 26 , 253] },
    { "id": 76,  "supercategory": "indoor",     "name": "vase"          , "color": [134, 219, 70 ] },
    { "id": 77,  "supercategory": "indoor",     "name": "scissors"      , "color": [0  , 132, 236] },
    { "id": 78,  "supercategory": "indoor",     "name": "teddy bear"    , "color": [134, 81 , 4  ] },
    { "id": 79,  "supercategory": "indoor",     "name": "hair drier"    , "color": [123, 68 , 172] },
    { "id": 80,  "supercategory": "indoor",     "name": "toothbrush"    , "color": [58 , 228, 226] }
]

coco17_91_classlabels = [
    { "id": 0 ,  "supercategory": "background", "name": "background"    , "color": [0  , 0  , 0  ] },
    { "id": 1 ,  "supercategory": "person",     "name": "person"        , "color": [81 , 120, 228] },
    { "id": 2 ,  "supercategory": "vehicle",    "name": "bicycle"       , "color": [138, 183, 33 ] },
    { "id": 3 ,  "supercategory": "vehicle",    "name": "car"           , "color": [49 , 3  , 150] },
    { "id": 4 ,  "supercategory": "vehicle",    "name": "motorcycle"    , "color": [122, 35 , 2  ] },
    { "id": 5 ,  "supercategory": "vehicle",    "name": "airplane"      , "color": [165, 168, 193] },
    { "id": 6 ,  "supercategory": "vehicle",    "name": "bus"           , "color": [140, 24 , 143] },
    { "id": 7 ,  "supercategory": "vehicle",    "name": "train"         , "color": [179, 165, 212] },
    { "id": 8 ,  "supercategory": "vehicle",    "name": "truck"         , "color": [72 , 153, 152] },
    { "id": 9 ,  "supercategory": "vehicle",    "name": "boat"          , "color": [19 , 64 , 83 ] },
    { "id": 10 , "supercategory": "outdoor",    "name": "traffic light" , "color": [122, 40 , 57 ] },
    { "id": 11,  "supercategory": "outdoor",    "name": "fire hydrant"  , "color": [219, 42 , 205] },
    { "id": 12,  "supercategory": "outdoor",    "name": "street sign"   , "color": [147, 24 , 225] },
    { "id": 13,  "supercategory": "outdoor",    "name": "stop sign"     , "color": [15 , 90 , 125] },
    { "id": 14,  "supercategory": "outdoor",    "name": "parking meter" , "color": [187, 80 , 10 ] },
    { "id": 15,  "supercategory": "outdoor",    "name": "bench"         , "color": [76 , 226, 142] },
    { "id": 16,  "supercategory": "animal",     "name": "bird"          , "color": [24 , 56 , 34 ] },
    { "id": 17,  "supercategory": "animal",     "name": "cat"           , "color": [41 , 174, 251] },
    { "id": 18,  "supercategory": "animal",     "name": "dog"           , "color": [21 , 8  , 251] },
    { "id": 19,  "supercategory": "animal",     "name": "horse"         , "color": [106, 128, 177] },
    { "id": 20,  "supercategory": "animal",     "name": "sheep"         , "color": [147, 90 , 131] },
    { "id": 21,  "supercategory": "animal",     "name": "cow"           , "color": [65 , 159, 189] },
    { "id": 22,  "supercategory": "animal",     "name": "elephant"      , "color": [129, 70 , 30 ] },
    { "id": 23,  "supercategory": "animal",     "name": "bear"          , "color": [38 , 181, 29 ] },
    { "id": 24,  "supercategory": "animal",     "name": "zebra"         , "color": [189, 238, 167] },
    { "id": 25,  "supercategory": "animal",     "name": "giraffe"       , "color": [173, 154, 136] },
    { "id": 26,  "supercategory": "accessory",  "name": "hat"           , "color": [69 , 77 , 52 ] },
    { "id": 27,  "supercategory": "accessory",  "name": "backpack"      , "color": [205, 104, 95 ] },
    { "id": 28,  "supercategory": "accessory",  "name": "umbrella"      , "color": [163, 13 , 178] },
    { "id": 29,  "supercategory": "accessory",  "name": "shoe"          , "color": [35 , 37 , 0  ] },
    { "id": 30,  "supercategory": "accessory",  "name": "eye glasses"   , "color": [0  , 60 , 2  ] },
    { "id": 31,  "supercategory": "accessory",  "name": "handbag"       , "color": [156, 84 , 167] },
    { "id": 32,  "supercategory": "accessory",  "name": "tie"           , "color": [10 , 146, 166] },
    { "id": 33,  "supercategory": "accessory",  "name": "suitcase"      , "color": [176, 137, 78 ] },
    { "id": 34,  "supercategory": "sports",     "name": "frisbee"       , "color": [190, 118, 41 ] },
    { "id": 35,  "supercategory": "sports",     "name": "skis"          , "color": [159, 178, 24 ] },
    { "id": 36,  "supercategory": "sports",     "name": "snowboard"     , "color": [107, 85 , 171] },
    { "id": 37,  "supercategory": "sports",     "name": "sports ball"   , "color": [186, 223, 221] },
    { "id": 38,  "supercategory": "sports",     "name": "kite"          , "color": [142, 218, 56 ] },
    { "id": 39,  "supercategory": "sports",     "name": "baseball bat"  , "color": [82 , 128, 254] },
    { "id": 40,  "supercategory": "sports",     "name": "baseball glove", "color": [64 , 200, 173] },
    { "id": 41,  "supercategory": "sports",     "name": "skateboard"    , "color": [112, 66 , 51 ] },
    { "id": 42,  "supercategory": "sports",     "name": "surfboard"     , "color": [47 , 131, 231] },
    { "id": 43,  "supercategory": "sports",     "name": "tennis racket" , "color": [37 , 70 , 244] },
    { "id": 44,  "supercategory": "kitchen",    "name": "bottle"        , "color": [139, 160, 1  ] },
    { "id": 45,  "supercategory": "kitchen",    "name": "plate"         , "color": [121, 97 , 34 ] },
    { "id": 46,  "supercategory": "kitchen",    "name": "wine glass"    , "color": [103, 32 , 74 ] },
    { "id": 47,  "supercategory": "kitchen",    "name": "cup"           , "color": [28 , 47 , 55 ] },
    { "id": 48,  "supercategory": "kitchen",    "name": "fork"          , "color": [219, 18 , 203] },
    { "id": 49,  "supercategory": "kitchen",    "name": "knife"         , "color": [41 , 125, 194] },
    { "id": 50,  "supercategory": "kitchen",    "name": "spoon"         , "color": [76 , 180, 131] },
    { "id": 51,  "supercategory": "kitchen",    "name": "bowl"          , "color": [143, 4  , 187] },
    { "id": 52,  "supercategory": "food",       "name": "banana"        , "color": [232, 188, 11 ] },
    { "id": 53,  "supercategory": "food",       "name": "apple"         , "color": [119, 177, 17 ] },
    { "id": 54,  "supercategory": "food",       "name": "sandwich"      , "color": [55 , 214, 248] },
    { "id": 55,  "supercategory": "food",       "name": "orange"        , "color": [100, 254, 62 ] },
    { "id": 56,  "supercategory": "food",       "name": "broccoli"      , "color": [15 , 12 , 37 ] },
    { "id": 57,  "supercategory": "food",       "name": "carrot"        , "color": [105, 24 , 82 ] },
    { "id": 58,  "supercategory": "food",       "name": "hot dog"       , "color": [192, 102, 113] },
    { "id": 59,  "supercategory": "food",       "name": "pizza"         , "color": [242, 21 , 163] },
    { "id": 60,  "supercategory": "food",       "name": "donut"         , "color": [13 , 42 , 240] },
    { "id": 61,  "supercategory": "food",       "name": "cake"          , "color": [83 , 228, 215] },
    { "id": 62,  "supercategory": "furniture",  "name": "chair"         , "color": [94 , 173, 36 ] },
    { "id": 63,  "supercategory": "furniture",  "name": "couch"         , "color": [63 , 48 , 10 ] },
    { "id": 64,  "supercategory": "furniture",  "name": "potted plant"  , "color": [199, 53 , 7  ] },
    { "id": 65,  "supercategory": "furniture",  "name": "bed"           , "color": [174, 28 , 109] },
    { "id": 66,  "supercategory": "furniture",  "name": "mirror"        , "color": [13 , 17 , 116] },
    { "id": 67,  "supercategory": "furniture",  "name": "dining table"  , "color": [216, 147, 179] },
    { "id": 68,  "supercategory": "furniture",  "name": "window"        , "color": [244, 33 , 215] },
    { "id": 69,  "supercategory": "furniture",  "name": "desk"          , "color": [131, 218, 147] },
    { "id": 70,  "supercategory": "furniture",  "name": "toilet"        , "color": [36 , 181, 193] },
    { "id": 71,  "supercategory": "furniture",  "name": "door"          , "color": [161, 163, 128] },
    { "id": 72,  "supercategory": "electronic", "name": "tv"            , "color": [54 , 95 , 132] },
    { "id": 73,  "supercategory": "electronic", "name": "laptop"        , "color": [142, 43 , 85 ] },
    { "id": 74,  "supercategory": "electronic", "name": "mouse"         , "color": [150, 175, 16 ] },
    { "id": 75,  "supercategory": "electronic", "name": "remote"        , "color": [125, 179, 231] },
    { "id": 76,  "supercategory": "electronic", "name": "keyboard"      , "color": [249, 95 , 141] },
    { "id": 77,  "supercategory": "electronic", "name": "cell phone"    , "color": [105, 24 , 191] },
    { "id": 78,  "supercategory": "appliance",  "name": "microwave"     , "color": [135, 51 , 82 ] },
    { "id": 79,  "supercategory": "appliance",  "name": "oven"          , "color": [69 , 21 , 20 ] },
    { "id": 80,  "supercategory": "appliance",  "name": "toaster"       , "color": [67 , 30 , 125] },
    { "id": 81,  "supercategory": "appliance",  "name": "sink"          , "color": [135, 205, 67 ] },
    { "id": 82,  "supercategory": "appliance",  "name": "refrigerator"  , "color": [35 , 219, 70 ] },
    { "id": 83,  "supercategory": "appliance",  "name": "blender"       , "color": [149, 140, 252] },
    { "id": 84,  "supercategory": "indoor",     "name": "book"          , "color": [80 , 203, 31 ] },
    { "id": 85,  "supercategory": "indoor",     "name": "clock"         , "color": [26 , 26 , 253] },
    { "id": 86,  "supercategory": "indoor",     "name": "vase"          , "color": [134, 219, 70 ] },
    { "id": 87,  "supercategory": "indoor",     "name": "scissors"      , "color": [0  , 132, 236] },
    { "id": 88,  "supercategory": "indoor",     "name": "teddy bear"    , "color": [134, 81 , 4  ] },
    { "id": 89,  "supercategory": "indoor",     "name": "hair drier"    , "color": [123, 68 , 172] },
    { "id": 90,  "supercategory": "indoor",     "name": "toothbrush"    , "color": [58 , 228, 226] },
    { "id": 91,  "supercategory": "indoor",     "name": "hair brush"    , "color": [149, 108, 73 ] }
]


@DATASETS.register(name="coco17_detection")
class COCO17Detection(YOLODetectionDataset):
    """
    
    Args:
        root (Path_): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image of shape [H, W, C], [H, W], or [S, S].
        classlabels (ClassLabels_ | None): ClassLabels object. Defaults to
            None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """

    def __init__(
        self,
        root            : Path_,
        split           : str,
        shape           : Ints,
        classlabels     : ClassLabels_ | None = None,
        transform       : Transforms_  | None = None,
        target_transform: Transforms_  | None = None,
        transforms      : Transforms_  | None = None,
        cache_images    : bool                = False,
        cache_data      : bool                = False,
        backend         : VisionBackend_      = VISION_BACKEND,
        verbose         : bool                = True,
        *args, **kwargs
    ):
        path = os.path.join(root, f"80_classlabels.json")
        if classlabels is None:
            if os.path.isfile(path):
                classlabels = ClassLabels.from_file(path)
            else:
                classlabels = ClassLabels.from_list(coco17_80_classlabels)
        
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
    
    def list_images(self):
        """
        List image files.
        """
        if self.split not in ["train", "val"]:
            console.log(
                f"{self.__class__.classname} dataset only supports `split`: "
                f"`train` or `val`. Get: {self.split}."
            )
        
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = self.root / "images" / f"{self.split}2017"
            for path in pbar.track(
                list(pattern.rglob("*.jpg")),
                description=f"Listing {self.__class__.classname} "
                            f"{self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))

    def annotation_files(self) -> Paths_:
        """
        Returns the path to json annotation files.
        """
        files: list[Path] = []
        for img in self.images:
            path = str(img.path)
            path = path.replace("images", "annotations_yolo")
            path = path.replace(".jpg", ".txt")
            files.append(Path(path))
        return files
    

@DATAMODULES.register(name="coco17_detection")
class COCO17DetectionDataModule(DataModule):
    """
    """
    
    def __init__(
        self,
        root: Path_ = DATA_DIR / "coco" / "coco2017",
        name: str   = "coco17_detection",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]{COCO17Detection.classname}[/red] datasets.")
        phase = ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = COCO17Detection(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = COCO17Detection(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = COCO17Detection(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """
        Load ClassLabels.
        """
        self.classlabels = ClassLabels.from_value(coco17_80_classlabels)


# H1: - Test -------------------------------------------------------------------

def test_coco17_detection():
    cfg = {
        "root": DATA_DIR / "coco" / "coco2017",
           # Root directory of dataset.
        "name": "coco17_detection",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [C, H, W], [H, W], or [S, S].
        "transform": [
            Resize(size=[3, 512, 512]),
        ],
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": None,
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": True,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = COCO17DetectionDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter = iter(dm.train_dataloader)
    start = time.time()
    input, target, meta = next(data_iter)
    end = time.time()
    console.log(end - start)

    drawings = []
    for i, img in enumerate(input):
        chw       = img.shape
        l         = target[target[:, 0] == i]
        l[:, 2:6] = box_cxcywh_norm_to_xyxy(l[:, 2:6], chw[1], chw[2])
        drawing   = draw_box(img, l, dm.class_labels.colors())
        drawings.append(drawing)
    drawings = torch.Tensor(drawings)
    imshow(winname="image", image=drawings)
    plt.show(block=True)


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str , default="test_coco17_detection", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test_coco17_detection":
        test_coco17_detection()
