#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain13K Datasets."""

from __future__ import annotations

__all__ = [
    "Rain13K",
    "Rain13KDataModule",
]

from typing import Literal

from mon import core
from mon.dataset.enhance.derain.rain100 import Rain100
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "derain"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
ImageAnnotation     = core.ImageAnnotation
ImageDataset        = core.ImageDataset


@DATASETS.register(name="rain13k")
class Rain13K(ImageDataset):
    """Rain13K dataset consists 13,000 pairs of rain/no-rain train images."""
    
    tasks : list[Task]  = [Task.DERAIN]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"   : ImageAnnotation,
        "hq_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        if self.split in [Split.TRAIN]:
            patterns = [
                self.root / "rain13k" / self.split_str / "lq",
            ]
        elif self.split in [Split.VAL]:
            patterns = [
                self.root / "rain1200" / self.split_str / "lq",
                self.root / "rain800"  / self.split_str / "lq",
            ]
        else:
            patterns = [
                self.root / "rain100"  / self.split_str / "lq",
                self.root / "rain100h" / self.split_str / "lq",
                self.root / "rain100l" / self.split_str / "lq",
                self.root / "rain1200" / self.split_str / "lq",
                self.root / "rain1400" / self.split_str / "lq",
                self.root / "rain2800" / self.split_str / "lq",
                self.root / "rain800"  / self.split_str / "lq",
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


@DATAMODULES.register(name="rain13k")
class Rain13KDataModule(DataModule):

    tasks: list[Task] = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain13K(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain100(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            # self.test  = Rain13K(split=Split.TEST,  **self.dataset_kwargs)
            self.test  = Rain100(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
