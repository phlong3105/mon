#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unsupervised LOL Datasets."""

from __future__ import annotations

__all__ = [
    "ULOL",
    "ULOLMixDataModule",
]

from typing import Literal

import cv2

from mon import core
from mon.dataset.enhance.llie.lol_v1 import LOLV1
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "llie"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
ImageAnnotation     = core.ImageAnnotation
ImageDataset        = core.ImageDataset


@DATASETS.register(name="ulol")
class ULOL(ImageDataset):

    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "dicm"             / "test"         / "lq",
            self.root / "fusion"           / "test"         / "lq",
            self.root / "lime"             / "test"         / "lq",
            self.root / "lol_v1"           / self.split_str / "hq",
            self.root / "lol_v1"           / self.split_str / "lq",
            self.root / "lol_v2_real"      / self.split_str / "hq",
            self.root / "lol_v2_real"      / self.split_str / "lq",
            self.root / "lol_v2_synthetic" / self.split_str / "hq",
            self.root / "lol_v2_synthetic" / self.split_str / "lq",
            self.root / "mef"              / "test"         / "lq",
            self.root / "npe"              / "test"         / "lq",
            self.root / "sice_mix"         / self.split_str / "lq",
            self.root / "sice_mix_v2"      / self.split_str / "lq",
            self.root / "vv"               / "test"         / "lq",
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
                depth_maps.append(ImageAnnotation(
                    path  = path.image_file(),
                    flags = cv2.IMREAD_GRAYSCALE
                ))
        
        self.datapoints["image"] = lq_images
        self.datapoints["depth"] = depth_maps


@DATAMODULES.register(name="ulol")
class ULOLMixDataModule(DataModule):
    
    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train =  ULOL(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV1(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLV1(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
