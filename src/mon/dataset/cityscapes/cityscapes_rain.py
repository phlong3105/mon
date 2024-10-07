#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cityscapes.

This module implements the Cityscapes dataset.

References:
	https://www.cityscapes-dataset.com/
"""

from __future__ import annotations

__all__ = [
    "CityscapesRain",
    "CityscapesRainDataModule",
]

from typing import Literal

import cv2

from mon import core
from mon.dataset.cityscapes.cityscapes import Cityscapes
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console                        = core.console
default_root_dir               = DATA_DIR / "cityscapes"
ClassLabels                    = core.ClassLabels
DataModule                     = core.DataModule
DatapointAttributes            = core.DatapointAttributes
DepthMapAnnotation             = core.DepthMapAnnotation
ImageAnnotation                = core.ImageAnnotation
MultimodalDataset              = core.MultimodalDataset
SemanticSegmentationAnnotation = core.SemanticSegmentationAnnotation


@DATASETS.register(name="cityscapes_rain")
class CityscapesRain(Cityscapes):

    tasks : list[Task]  = [Task.DERAIN]
    splits: list[Split] = [Split.TRAIN, Split.VAL]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
        "semantic" : SemanticSegmentationAnnotation,  # gtFine
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / self.split_str /  "leftImg8bit_rain",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                path = img.path.replace("/leftImg8bit_rain/", "/leftImg8bit/")
                stem = path.stem
                path = path.parent / f"{stem.split("leftImg8bit")[0]}leftImg8bit{path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
        
        # Semantic segmentation maps
        semantic: list[SemanticSegmentationAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} semantic maps"
            ):
                path = img.path.replace("/leftImg8bit/", "/gtFine/")
                semantic.append(
                    SemanticSegmentationAnnotation(
                        path  = path.image_file(),
                        root  = img.root,
                        flags = cv2.IMREAD_GRAYSCALE
                    )
                )
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        self.datapoints["semantic"]  = semantic


@DATAMODULES.register(name="cityscapes_rain")
class CityscapesRainDataModule(DataModule):
    
    tasks: list[Task] = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = CityscapesRain(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = CityscapesRain(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = CityscapesRain(split=Split.VAL,   **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
