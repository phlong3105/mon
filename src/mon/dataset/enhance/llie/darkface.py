#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DarkFace Datasets."""

from __future__ import annotations

__all__ = [
    "DarkFace",
    "DarkFaceDataModule",
]

from typing import Literal

import cv2

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "llie"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
ImageAnnotation     = core.ImageAnnotation
ImageDataset        = core.ImageDataset


@DATASETS.register(name="darkface")
class DarkFace(ImageDataset):
    """DarkFace dataset consists of 6,490 low-light images."""
    
    tasks : list[Task]  = [Task.LLIE, Task.DETECT]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "darkface" / self.split_str / "lq",
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
        
        self.datapoints["image"] = lq_images
        self.datapoints["depth"] = depth_maps


@DATAMODULES.register(name="darkface")
class DarkFaceDataModule(DataModule):
    
    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = DarkFace(split=Split.TEST, **self.dataset_kwargs)
            self.val   = DarkFace(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = DarkFace(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
