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
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="kitti_snow")
class KITTISnow(MultimodalDataset):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow" / self.split_str / "image",
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
                        images.append(ImageAnnotation(path=path))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="kitti_snow_s")
class KITTISnowS(MultimodalDataset):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_s" / self.split_str / "image",
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
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="kitti_snow_m")
class KITTISnowM(MultimodalDataset):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_m" / self.split_str / "image",
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
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="kitti_snow_l")
class KITTISnowL(MultimodalDataset):

    tasks : list[Task]  = [Task.DESNOW]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "kitti_snow_l" / self.split_str / "image",
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
        
        self.datapoints["image"] = images
        

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
