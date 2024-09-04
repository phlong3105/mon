#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cityscapes Datasets.

This module implements the paper: "Night-time Scene Parsing with a Large Real
Dataset". The largest real-world night-time semantic segmentation dataset with
pixel-level labels.

References:
	https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html
"""

from __future__ import annotations

__all__ = [
	"NightCity",
	"NightCityDataModule",
]

from typing import Literal

import cv2

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console                = core.console
default_root_dir       = DATA_DIR / "enhance" / "llie"
DataModule             = core.DataModule
DatapointAttributes    = core.DatapointAttributes
ImageAnnotation        = core.ImageAnnotation
ImageDataset           = core.ImageDataset
SegmentationAnnotation = core.SemanticSegmentationAnnotation


# region Dataset

@DATASETS.register(name="nightcity")
class NightCity(ImageDataset):
    """NightCity dataset consists of 4,297 real night-time images with ground
    truth pixel-level semantic annotations.
    
    References:
    	https://dmcv.sjtu.edu.cn/people/phd/tanxin/NightCity/index.html
    """
    
    tasks : list[Task]  = [Task.LLIE, Task.SEGMENT]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"  : ImageAnnotation,
        "depth"  : ImageAnnotation,
        "segment": SegmentationAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        if self.split == Split.TEST:
            patterns = [
                self.root / "nightcity" / "val" / "lq",
            ]
        else:
            patterns = [
                self.root / "nightcity" / self.split_str / "lq",
            ]
        
        # LQ images
        lq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sorted(list(pattern.rglob("*"))),
                    description=f"Listing {self.__class__.__name__} "
                                f"{self.split_str} lq images"
                ):
                    if path.is_image_file():
                        lq_images.append(ImageAnnotation(path=path))
        
        # LQ depth images
        depth_maps: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} lq depth maps"
            ):
                path = img.path.replace("/lq/", "/lq_dav2_vitb_g/")
                depth_maps.append(ImageAnnotation(path=path.image_file(), flags=cv2.IMREAD_GRAYSCALE))
        
        # LQ segmentation maps
        segments: list[SegmentationAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} lq segmentation maps"
            ):
                path = img.path.replace("/lq/", "/labelIds/")
                segments.append(SegmentationAnnotation(path=path.image_file(), flags=cv2.IMREAD_GRAYSCALE))
        
        self.datapoints["image"]   = lq_images
        self.datapoints["depth"]   = depth_maps
        self.datapoints["segment"] = segments
        
# endregion


# region Datamodule

@DATAMODULES.register(name="nightcity")
class NightCityDataModule(DataModule):
    
    tasks: list[Task] = [Task.LLIE, Task.SEGMENT]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = NightCity(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = NightCity(split=Split.VAL,   **self.dataset_kwargs)
            # self.val   = LOLV1(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = NightCity(split=Split.TEST,  **self.dataset_kwargs)
            
        self.get_classlabels()
        if self.can_log:
            self.summarize()

# endregion
