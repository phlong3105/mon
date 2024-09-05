#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cityscapes.

This module implements the Cityscapes dataset.

References:
	https://www.cityscapes-dataset.com/
"""

from __future__ import annotations

__all__ = [
    "Cityscapes",
    "CityscapesDataModule",
]

from typing import Literal

import cv2

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console                        = core.console
default_root_dir               = DATA_DIR / "cityscapes"
ClassLabels                    = core.ClassLabels
DataModule                     = core.DataModule
DatapointAttributes            = core.DatapointAttributes
ImageAnnotation                = core.ImageAnnotation
ImageDataset                   = core.ImageDataset
SemanticSegmentationAnnotation = core.SemanticSegmentationAnnotation


# region Dataset

@DATASETS.register(name="cityscapes")
class Cityscapes(ImageDataset):
    """Cityscapes dataset.
    
    Args:
        root       : The root directory of the dataset.
        use_blurred: If ``True``, use the blurred images. Defaults: ``False``.
        use_coarse : If ``True``, use the coarse annotations. Defaults: ``False``.
    """
    
    tasks : list[Task]  = [Task.SEGMENT]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"      : ImageAnnotation,                 # leftImg8bit
        "semantic"   : SemanticSegmentationAnnotation,  # gtFine
        # "right_image": ImageAnnotation,                 # rightImg8bit
        # "disparity"  : ImageAnnotation,                 # disparity
        # "camera"     : ImageAnnotation,                 # camera
        # "vehicle"    : ImageAnnotation,                 # vehicle
        # "bbox3d"     : ImageAnnotation,                 # bbox3d
    })
    has_test_annotations: bool        = False
    classlabels         : ClassLabels = ClassLabels([
        {"name": "unlabeled"           , "id": 0 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "ego vehicle"         , "id": 1 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "rectification border", "id": 2 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "out of roi"          , "id": 3 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "static"              , "id": 4 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [0  , 0  ,   0]},
        {"name": "dynamic"             , "id": 5 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [111, 74 ,   0]},
        {"name": "ground"              , "id": 6 , "train_id": 255, "category": "void"        , "category_id": 0, "ignore_in_eval": True , "color": [81 , 0  ,  81]},
        {"name": "road"                , "id": 7 , "train_id": 0  , "category": "flat"        , "category_id": 1, "ignore_in_eval": False, "color": [128, 64 , 128]},
        {"name": "sidewalk"            , "id": 8 , "train_id": 1  , "category": "flat"        , "category_id": 1, "ignore_in_eval": False, "color": [244, 35 , 232]},
        {"name": "parking"             , "id": 9 , "train_id": 255, "category": "flat"        , "category_id": 1, "ignore_in_eval": True , "color": [250, 170, 160]},
        {"name": "rail track"          , "id": 10, "train_id": 255, "category": "flat"        , "category_id": 1, "ignore_in_eval": True , "color": [230, 150, 140]},
        {"name": "building"            , "id": 11, "train_id": 2  , "category": "construction", "category_id": 2, "ignore_in_eval": False, "color": [70 , 70 ,  70]},
        {"name": "wall"                , "id": 12, "train_id": 3  , "category": "construction", "category_id": 2, "ignore_in_eval": False, "color": [102, 102, 156]},
        {"name": "fence"               , "id": 13, "train_id": 4  , "category": "construction", "category_id": 2, "ignore_in_eval": False, "color": [190, 153, 153]},
        {"name": "guard rail"          , "id": 14, "train_id": 255, "category": "construction", "category_id": 2, "ignore_in_eval": True , "color": [180, 165, 180]},
        {"name": "bridge"              , "id": 15, "train_id": 255, "category": "construction", "category_id": 2, "ignore_in_eval": True , "color": [150, 100, 100]},
        {"name": "tunnel"              , "id": 16, "train_id": 255, "category": "construction", "category_id": 2, "ignore_in_eval": True , "color": [150, 120,  90]},
        {"name": "pole"                , "id": 17, "train_id": 5  , "category": "object"      , "category_id": 3, "ignore_in_eval": False, "color": [153, 153, 153]},
        {"name": "polegroup"           , "id": 18, "train_id": 255, "category": "object"      , "category_id": 3, "ignore_in_eval": True , "color": [153, 153, 153]},
        {"name": "traffic light"       , "id": 19, "train_id": 6  , "category": "object"      , "category_id": 3, "ignore_in_eval": False, "color": [250, 170,  30]},
        {"name": "traffic sign"        , "id": 20, "train_id": 7  , "category": "object"      , "category_id": 3, "ignore_in_eval": False, "color": [220, 220,   0]},
        {"name": "vegetation"          , "id": 21, "train_id": 8  , "category": "nature"      , "category_id": 4, "ignore_in_eval": False, "color": [107, 142,  35]},
        {"name": "terrain"             , "id": 22, "train_id": 9  , "category": "nature"      , "category_id": 4, "ignore_in_eval": False, "color": [152, 251, 152]},
        {"name": "sky"                 , "id": 23, "train_id": 10 , "category": "sky"         , "category_id": 5, "ignore_in_eval": False, "color": [70 , 130, 180]},
        {"name": "person"              , "id": 24, "train_id": 11 , "category": "human"       , "category_id": 6, "ignore_in_eval": False, "color": [220, 20 ,  60]},
        {"name": "rider"               , "id": 25, "train_id": 12 , "category": "human"       , "category_id": 6, "ignore_in_eval": False, "color": [255, 0  ,   0]},
        {"name": "car"                 , "id": 26, "train_id": 13 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 0  , 142]},
        {"name": "truck"               , "id": 27, "train_id": 14 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 0  ,  70]},
        {"name": "bus"                 , "id": 28, "train_id": 15 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 60 , 100]},
        {"name": "caravan"             , "id": 29, "train_id": 255, "category": "vehicle"     , "category_id": 7, "ignore_in_eval": True , "color": [0  , 0  ,  90]},
        {"name": "trailer"             , "id": 30, "train_id": 255, "category": "vehicle"     , "category_id": 7, "ignore_in_eval": True , "color": [0  , 0  , 110]},
        {"name": "train"               , "id": 31, "train_id": 16 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 80 , 100]},
        {"name": "motorcycle"          , "id": 32, "train_id": 17 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [0  , 0  , 230]},
        {"name": "bicycle"             , "id": 33, "train_id": 18 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": False, "color": [119, 11 ,  32]},
        {"name": "license plate"       , "id": -1, "train_id": -1 , "category": "vehicle"     , "category_id": 7, "ignore_in_eval": True , "color": [0  , 0  , 142]},
    ])
    
    def __init__(
        self,
        root       : core.Path = default_root_dir,
        use_blurred: bool = False,
        use_coarse : bool = False,
        *args, **kwargs
    ):
        self.use_blurred = use_blurred
        self.use_coarse  = use_coarse
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        image_name = "leftImg8bit_blurred"  if self.use_blurred else "leftImg8bit"
        gt_name    = "gtCoarse"             if self.use_coarse  else "gtFine"
        patterns   = [
            self.root / self.split_str / image_name,
        ]
        
        # Left Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} "
                                f"{self.split_str} {image_name} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path))
        
        # Semantic segmentation maps
        semantic: list[SemanticSegmentationAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} {gt_name} maps"
            ):
                path = img.path.replace(f"{image_name}", f"{gt_name}")
                path = path.parent / f"{path.stem}_labelIds{path.suffix}"
                semantic.append(SemanticSegmentationAnnotation(path=path.image_file(), flags=cv2.IMREAD_GRAYSCALE))
        
        self.datapoints["image"]    = images
        self.datapoints["semantic"] = semantic
        
# endregion


# region Datamodule

@DATAMODULES.register(name="cityscapes")
class CityscapesDataModule(DataModule):
    
    tasks: list[Task] = [Task.SEGMENT]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Cityscapes(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Cityscapes(split=Split.VAL, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Cityscapes(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
    
# endregion
