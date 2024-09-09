#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""COCO Datasets.

This package implements COCO datasets for object detection and instance
segmentation.
"""

from __future__ import annotations

__all__ = [
    "COCO",
    "COCODataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "coco"
BBoxesAnnotation    = core.BBoxesAnnotation
ClassLabels         = core.ClassLabels
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
ImageAnnotation     = core.ImageAnnotation
ImageDataset        = core.ImageDataset


@DATASETS.register(name="coco")
class COCO(ImageDataset):
    """COCO dataset."""
    
    tasks : list[Task]  = [Task.DETECT]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        # "bbox" : BBoxesAnnotation,
    })
    has_test_annotations: bool        = False
    classlabels         : ClassLabels = ClassLabels([
        {"name": "background"    , "id": 0 , "supercategory": "background", "color": [0  , 0  ,   0]},
        {"name": "person"        , "id": 1 , "supercategory": "person"    , "color": [81 , 120, 228]},
        {"name": "bicycle"       , "id": 2 , "supercategory": "vehicle"   , "color": [138, 183,  33]},
        {"name": "car"           , "id": 3 , "supercategory": "vehicle"   , "color": [49 , 3  , 150]},
        {"name": "motorcycle"    , "id": 4 , "supercategory": "vehicle"   , "color": [122, 35 ,   2]},
        {"name": "airplane"      , "id": 5 , "supercategory": "vehicle"   , "color": [165, 168, 193]},
        {"name": "bus"           , "id": 6 , "supercategory": "vehicle"   , "color": [140, 24 , 143]},
        {"name": "train"         , "id": 7 , "supercategory": "vehicle"   , "color": [179, 165, 212]},
        {"name": "truck"         , "id": 8 , "supercategory": "vehicle"   , "color": [72 , 153, 152]},
        {"name": "boat"          , "id": 9 , "supercategory": "vehicle"   , "color": [19 , 64 ,  83]},
        {"name": "traffic light" , "id": 10, "supercategory": "outdoor"   , "color": [122, 40 ,  57]},
        {"name": "fire hydrant"  , "id": 11, "supercategory": "outdoor"   , "color": [219, 42 , 205]},
        {"name": "street sign"   , "id": 12, "supercategory": "outdoor"   , "color": [147, 24 , 225]},
        {"name": "stop sign"     , "id": 13, "supercategory": "outdoor"   , "color": [15 , 90 , 125]},
        {"name": "parking meter" , "id": 14, "supercategory": "outdoor"   , "color": [187, 80 ,  10]},
        {"name": "bench"         , "id": 15, "supercategory": "outdoor"   , "color": [76 , 226, 142]},
        {"name": "bird"          , "id": 16, "supercategory": "animal"    , "color": [24 , 56 ,  34]},
        {"name": "cat"           , "id": 17, "supercategory": "animal"    , "color": [41 , 174, 251]},
        {"name": "dog"           , "id": 18, "supercategory": "animal"    , "color": [21 , 8  , 251]},
        {"name": "horse"         , "id": 19, "supercategory": "animal"    , "color": [106, 128, 177]},
        {"name": "sheep"         , "id": 20, "supercategory": "animal"    , "color": [147, 90 , 131]},
        {"name": "cow"           , "id": 21, "supercategory": "animal"    , "color": [65 , 159, 189]},
        {"name": "elephant"      , "id": 22, "supercategory": "animal"    , "color": [129, 70 ,  30]},
        {"name": "bear"          , "id": 23, "supercategory": "animal"    , "color": [38 , 181,  29]},
        {"name": "zebra"         , "id": 24, "supercategory": "animal"    , "color": [189, 238, 167]},
        {"name": "giraffe"       , "id": 25, "supercategory": "animal"    , "color": [173, 154, 136]},
        {"name": "hat"           , "id": 26, "supercategory": "accessory" , "color": [69 , 77 ,  52]},
        {"name": "backpack"      , "id": 27, "supercategory": "accessory" , "color": [205, 104,  95]},
        {"name": "umbrella"      , "id": 28, "supercategory": "accessory" , "color": [163, 13 , 178]},
        {"name": "shoe"          , "id": 29, "supercategory": "accessory" , "color": [35 , 37 ,   0]},
        {"name": "eye glasses"   , "id": 30, "supercategory": "accessory" , "color": [0  , 60 ,   2]},
        {"name": "handbag"       , "id": 31, "supercategory": "accessory" , "color": [156, 84 , 167]},
        {"name": "tie"           , "id": 32, "supercategory": "accessory" , "color": [10 , 146, 166]},
        {"name": "suitcase"      , "id": 33, "supercategory": "accessory" , "color": [176, 137,  78]},
        {"name": "frisbee"       , "id": 34, "supercategory": "sports"    , "color": [190, 118,  41]},
        {"name": "skis"          , "id": 35, "supercategory": "sports"    , "color": [159, 178,  24]},
        {"name": "snowboard"     , "id": 36, "supercategory": "sports"    , "color": [107, 85 , 171]},
        {"name": "sports ball"   , "id": 37, "supercategory": "sports"    , "color": [186, 223, 221]},
        {"name": "kite"          , "id": 38, "supercategory": "sports"    , "color": [142, 218,  56]},
        {"name": "baseball bat"  , "id": 39, "supercategory": "sports"    , "color": [82 , 128, 254]},
        {"name": "baseball glove", "id": 40, "supercategory": "sports"    , "color": [64 , 200, 173]},
        {"name": "skateboard"    , "id": 41, "supercategory": "sports"    , "color": [112, 66 ,  51]},
        {"name": "surfboard"     , "id": 42, "supercategory": "sports"    , "color": [47 , 131, 231]},
        {"name": "tennis racket" , "id": 43, "supercategory": "sports"    , "color": [37 , 70 , 244]},
        {"name": "bottle"        , "id": 44, "supercategory": "kitchen"   , "color": [139, 160,   1]},
        {"name": "plate"         , "id": 45, "supercategory": "kitchen"   , "color": [121, 97 ,  34]},
        {"name": "wine glass"    , "id": 46, "supercategory": "kitchen"   , "color": [103, 32 ,  74]},
        {"name": "cup"           , "id": 47, "supercategory": "kitchen"   , "color": [28 , 47 ,  55]},
        {"name": "fork"          , "id": 48, "supercategory": "kitchen"   , "color": [219, 18 , 203]},
        {"name": "knife"         , "id": 49, "supercategory": "kitchen"   , "color": [41 , 125, 194]},
        {"name": "spoon"         , "id": 50, "supercategory": "kitchen"   , "color": [76 , 180, 131]},
        {"name": "bowl"          , "id": 51, "supercategory": "kitchen"   , "color": [143, 4  , 187]},
        {"name": "banana"        , "id": 52, "supercategory": "food"      , "color": [232, 188,  11]},
        {"name": "apple"         , "id": 53, "supercategory": "food"      , "color": [119, 177,  17]},
        {"name": "sandwich"      , "id": 54, "supercategory": "food"      , "color": [55 , 214, 248]},
        {"name": "orange"        , "id": 55, "supercategory": "food"      , "color": [100, 254,  62]},
        {"name": "broccoli"      , "id": 56, "supercategory": "food"      , "color": [15 , 12 ,  37]},
        {"name": "carrot"        , "id": 57, "supercategory": "food"      , "color": [105, 24 ,  82]},
        {"name": "hot dog"       , "id": 58, "supercategory": "food"      , "color": [192, 102, 113]},
        {"name": "pizza"         , "id": 59, "supercategory": "food"      , "color": [242, 21 , 163]},
        {"name": "donut"         , "id": 60, "supercategory": "food"      , "color": [13 , 42 , 240]},
        {"name": "cake"          , "id": 61, "supercategory": "food"      , "color": [83 , 228, 215]},
        {"name": "chair"         , "id": 62, "supercategory": "furniture" , "color": [94 , 173,  36]},
        {"name": "couch"         , "id": 63, "supercategory": "furniture" , "color": [63 , 48 ,  10]},
        {"name": "potted plant"  , "id": 64, "supercategory": "furniture" , "color": [199, 53 ,   7]},
        {"name": "bed"           , "id": 65, "supercategory": "furniture" , "color": [174, 28 , 109]},
        {"name": "mirror"        , "id": 66, "supercategory": "furniture" , "color": [13 , 17 , 116]},
        {"name": "dining table"  , "id": 67, "supercategory": "furniture" , "color": [216, 147, 179]},
        {"name": "window"        , "id": 68, "supercategory": "furniture" , "color": [244, 33 , 215]},
        {"name": "desk"          , "id": 69, "supercategory": "furniture" , "color": [131, 218, 147]},
        {"name": "toilet"        , "id": 70, "supercategory": "furniture" , "color": [36 , 181, 193]},
        {"name": "door"          , "id": 71, "supercategory": "furniture" , "color": [161, 163, 128]},
        {"name": "tv"            , "id": 72, "supercategory": "electronic", "color": [54 , 95 , 132]},
        {"name": "laptop"        , "id": 73, "supercategory": "electronic", "color": [142, 43 ,  85]},
        {"name": "mouse"         , "id": 74, "supercategory": "electronic", "color": [150, 175,  16]},
        {"name": "remote"        , "id": 75, "supercategory": "electronic", "color": [125, 179, 231]},
        {"name": "keyboard"      , "id": 76, "supercategory": "electronic", "color": [249, 95 , 141]},
        {"name": "cell phone"    , "id": 77, "supercategory": "electronic", "color": [105, 24 , 191]},
        {"name": "microwave"     , "id": 78, "supercategory": "appliance" , "color": [135, 51 ,  82]},
        {"name": "oven"          , "id": 79, "supercategory": "appliance" , "color": [69 , 21 ,  20]},
        {"name": "toaster"       , "id": 80, "supercategory": "appliance" , "color": [67 , 30 , 125]},
        {"name": "sink"          , "id": 81, "supercategory": "appliance" , "color": [135, 205,  67]},
        {"name": "refrigerator"  , "id": 82, "supercategory": "appliance" , "color": [35 , 219,  70]},
        {"name": "blender"       , "id": 83, "supercategory": "appliance" , "color": [149, 140, 252]},
        {"name": "book"          , "id": 84, "supercategory": "indoor"    , "color": [80 , 203,  31]},
        {"name": "clock"         , "id": 85, "supercategory": "indoor"    , "color": [26 , 26 , 253]},
        {"name": "vase"          , "id": 86, "supercategory": "indoor"    , "color": [134, 219,  70]},
        {"name": "scissors"      , "id": 87, "supercategory": "indoor"    , "color": [0  , 132, 236]},
        {"name": "teddy bear"    , "id": 88, "supercategory": "indoor"    , "color": [134, 81 ,   4]},
        {"name": "hair drier"    , "id": 89, "supercategory": "indoor"    , "color": [123, 68 , 172]},
        {"name": "toothbrush"    , "id": 90, "supercategory": "indoor"    , "color": [58 , 228, 226]},
        {"name": "hair brush"    , "id": 91, "supercategory": "indoor"    , "color": [149, 108,  73]}
    ])
    
    def __init__(
        self,
        root: core.Path = default_root_dir,
        *args, **kwargs
    ):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / self.split_str / "image",
        ]
        
        # Left Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} "
                                f"{self.split_str} left images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path))
        
        self.datapoints["image"] = images


@DATAMODULES.register(name="coco")
class COCODataModule(DataModule):
    
    tasks: list[Task] = [Task.DETECT]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = COCO(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = COCO(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = COCO(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
