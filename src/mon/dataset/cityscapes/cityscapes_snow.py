#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cityscapes.

This module implements the Cityscapes dataset.

References:
	https://www.cityscapes-dataset.com/
"""

from __future__ import annotations

__all__ = [
    "CityscapesSnow",
    "CityscapesSnowDataModule",
    "CityscapesSnowL",
    "CityscapesSnowLDataModule",
    "CityscapesSnowM",
    "CityscapesSnowMDataModule",
    "CityscapesSnowS",
    "CityscapesSnowSDataModule",
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
ImageAnnotation                = core.ImageAnnotation
ImageDataset                   = core.ImageDataset
SemanticSegmentationAnnotation = core.SemanticSegmentationAnnotation


@DATASETS.register(name="cityscapes_snow")
class CityscapesSnow(Cityscapes):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
        "semantic": SemanticSegmentationAnnotation,  # gtFine
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "cityscapes_snow_s" / self.split_str / "lq",
            self.root / "cityscapes_snow_m" / self.split_str / "lq",
            self.root / "cityscapes_snow_l" / self.split_str / "lq",
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
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} hq images"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        # Semantic segmentation maps
        semantic: list[SemanticSegmentationAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                hq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} semantic maps"
            ):
                path = img.path.replace("/leftImg8bit/", "/gtFine/")
                semantic.append(SemanticSegmentationAnnotation(path=path.image_file(), flags=cv2.IMREAD_GRAYSCALE))
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images
        self.datapoints["semantic"] = semantic


@DATASETS.register(name="cityscapes_snow_s")
class CityscapesSnowS(Cityscapes):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
        "semantic": SemanticSegmentationAnnotation,  # gtFine
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "cityscapes_snow_s" / self.split_str / "lq",
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
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} hq images"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        # Semantic segmentation maps
        semantic: list[SemanticSegmentationAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                hq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} semantic maps"
            ):
                path = img.path.replace("/leftImg8bit/", "/gtFine/")
                semantic.append(SemanticSegmentationAnnotation(path=path.image_file(), flags=cv2.IMREAD_GRAYSCALE))
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images
        self.datapoints["semantic"] = semantic


@DATASETS.register(name="cityscapes_snow_m")
class CityscapesSnowM(Cityscapes):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
        "semantic": SemanticSegmentationAnnotation,  # gtFine
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "cityscapes_snow_m" / self.split_str / "lq",
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
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} hq images"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        # Semantic segmentation maps
        semantic: list[SemanticSegmentationAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                hq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} semantic maps"
            ):
                path = img.path.replace("/leftImg8bit/", "/gtFine/")
                semantic.append(SemanticSegmentationAnnotation(path=path.image_file(), flags=cv2.IMREAD_GRAYSCALE))
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images
        self.datapoints["semantic"] = semantic
        

@DATASETS.register(name="cityscapes_snow_l")
class CityscapesSnowL(Cityscapes):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
        "semantic": SemanticSegmentationAnnotation,  # gtFine
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "cityscapes_snow_l" / self.split_str / "lq",
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
        
        # HQ images
        hq_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} hq images"
            ):
                path = img.path.replace("/lq/", "/hq/")
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        # Semantic segmentation maps
        semantic: list[SemanticSegmentationAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                hq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} semantic maps"
            ):
                path = img.path.replace("/leftImg8bit/", "/gtFine/")
                semantic.append(SemanticSegmentationAnnotation(path=path.image_file(), flags=cv2.IMREAD_GRAYSCALE))
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images
        self.datapoints["semantic"] = semantic


@DATAMODULES.register(name="cityscapes_snow")
class CityscapesSnowDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = CityscapesSnow(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = CityscapesSnow(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = CityscapesSnow(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="cityscapes_snow_s")
class CityscapesSnowSDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = CityscapesSnowS(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = CityscapesSnowS(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = CityscapesSnowS(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="cityscapes_snow_m")
class CityscapesSnowMDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = CityscapesSnowM(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = CityscapesSnowM(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = CityscapesSnowM(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="cityscapes_snow_l")
class CityscapesSnowLDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = CityscapesSnowL(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = CityscapesSnowL(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = CityscapesSnowL(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
