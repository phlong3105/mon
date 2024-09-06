#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL-v2 Datasets."""

from __future__ import annotations

__all__ = [
    "LOLV2Real",
    "LOLV2RealDataModule",
    "LOLV2Synthetic",
    "LOLV2SyntheticDataModule",
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


# region Dataset

@DATASETS.register(name="lol_v2_real")
class LOLV2Real(ImageDataset):
    """LOL-v2 Real (VE-LOL) dataset consists of ``500`` low-light and
    normal-light image pairs. They are divided into ``400`` training pairs and
    ``100`` testing pairs. The low-light images contain noise produced during
    the photo capture process. Most of the images are indoor scenes. All the
    images have a resolution of ``400×600``.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "depth"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
        "hq_depth": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "lol_v2_real" / self.split_str / "lq",
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
        lq_depth_maps: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} lq depth maps"
            ):
                path = img.path.replace("/lq/", "/lq_dav2_vitb_g/")
                lq_depth_maps.append(ImageAnnotation(
                    path  = path.image_file(),
                    flags = cv2.IMREAD_GRAYSCALE
                ))
        
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
        
        # HQ depth images
        hq_depth_maps: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} hq depth maps"
            ):
                path = img.path.replace("/lq/", "/hq_dav2_vitb_c/")
                hq_depth_maps.append(ImageAnnotation(
                    path  = path.image_file(),
                    flags = cv2.IMREAD_GRAYSCALE
                ))
        
        self.datapoints["image"]    = lq_images
        self.datapoints["depth"]    = lq_depth_maps
        self.datapoints["hq_image"] = hq_images
        self.datapoints["hq_depth"] = hq_depth_maps


@DATASETS.register(name="lol_v2_synthetic")
class LOLV2Synthetic(ImageDataset):
    """LOL-v2 Synthetic (VE-LOL-Syn) dataset consists of ``1000`` low-light and
    normal-light image pairs. They are divided into ``900`` training pairs and
    ``100`` testing pairs. The low-light images contain noise produced during
    the photo capture process. Most of the images are indoor scenes. All the
    images have a resolution of ``400×600``.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "depth"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
        "hq_depth": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "lol_v2_synthetic" / self.split_str / "lq",
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
        lq_depth_maps: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} lq depth maps"
            ):
                path = img.path.replace("/lq/", "/lq_dav2_vitb_g/")
                lq_depth_maps.append(ImageAnnotation(
                    path  = path.image_file(),
                    flags = cv2.IMREAD_GRAYSCALE
                ))
        
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
        
        # HQ depth images
        hq_depth_maps: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                lq_images,
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split_str} hq depth maps"
            ):
                path = img.path.replace("/lq/", "/hq_dav2_vitb_c/")
                hq_depth_maps.append(ImageAnnotation(
                    path  = path.image_file(),
                    flags = cv2.IMREAD_GRAYSCALE
                ))
        
        self.datapoints["image"]    = lq_images
        self.datapoints["depth"]    = lq_depth_maps
        self.datapoints["hq_image"] = hq_images
        self.datapoints["hq_depth"] = hq_depth_maps


@DATAMODULES.register(name="lol_v2_real")
class LOLV2RealDataModule(DataModule):
    
    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = LOLV2Real(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV2Real(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLV2Real(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="lol_v2_synthetic")
class LOLV2SyntheticDataModule(DataModule):
    
    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = LOLV2Synthetic(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLV2Synthetic(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLV2Synthetic(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
