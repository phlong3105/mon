#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""KITTI-Snow Datasets."""

from __future__ import annotations

__all__ = [
    "KITTISnow",
    "KITTISnowDataModule",
    "KITTISnowL",
    "KITTISnowLDataModule",
    "KITTISnowM",
    "KITTISnowMDataModule",
    "KITTISnowS",
    "KITTISnowSDataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "desnow"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
ImageAnnotation     = core.ImageAnnotation
ImageDataset        = core.ImageDataset


@DATASETS.register(name="kitti_snow")
class KITTISnow(ImageDataset):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow" / self.split_str / "lq",
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
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="kitti_snow_s")
class KITTISnowS(ImageDataset):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_s" / self.split_str / "lq",
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
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="kitti_snow_m")
class KITTISnowM(ImageDataset):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_m" / self.split_str / "lq",
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
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATASETS.register(name="kitti_snow_l")
class KITTISnowL(ImageDataset):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_l" / self.split_str / "lq",
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
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images
        

@DATAMODULES.register(name="kitti_snow")
class KITTISnowDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = KITTISnow(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = KITTISnow(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = KITTISnow(split=Split.TRAIN, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="kitti_snow_s")
class KITTISnowSDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = KITTISnowS(split=Split.TEST, **self.dataset_kwargs)
            self.val   = KITTISnowS(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = KITTISnowS(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="kitti_snow_m")
class KITTISnowMDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = KITTISnowM(split=Split.TEST, **self.dataset_kwargs)
            self.val   = KITTISnowM(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = KITTISnowM(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="kitti_snow_l")
class KITTISnowLDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = KITTISnowL(split=Split.TEST, **self.dataset_kwargs)
            self.val   = KITTISnowL(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = KITTISnowL(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
