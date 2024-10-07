#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain1200 Datasets."""

from __future__ import annotations

__all__ = [
    "Rain1200",
    "Rain1200DataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "derain"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="rain1200")
class Rain1200(MultimodalDataset):
    """Rain1200 dataset consists 1,200 pairs of rain/no-rain test images and
    12,000 pairs of rain/no-rain train images.
    """
    
    tasks : list[Task]  = [Task.DERAIN]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        if self.split in [Split.TRAIN]:
            patterns = [
                self.root / "rain1200_light"  / self.split_str / "image",
                self.root / "rain1200_medium" / self.split_str / "image",
                self.root / "rain1200_heavy"  / self.split_str / "image",
            ]
        else:
            patterns = [
                self.root / "rain1200" / self.split_str / "image",
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


@DATAMODULES.register(name="rain1200")
class Rain1200DataModule(DataModule):

    tasks: list[Task] = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain1200(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain1200(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain1200(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
