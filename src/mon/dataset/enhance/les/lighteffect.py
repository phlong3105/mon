#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LightEffect Datasets."""

from __future__ import annotations

__all__ = [
    "LightEffect",
    "LightEffectDataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "les"
ClassLabels         = core.ClassLabels
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
ImageAnnotation     = core.ImageAnnotation
ImageDataset        = core.ImageDataset


@DATASETS.register(name="lighteffect")
class LightEffect(ImageDataset):
    """LightEffect dataset consists 961 flare images."""
    
    tasks : list[Task]  = [Task.LES]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            # self.root / self.split / "light-effect" / "clear",
            self.root / "lighteffect" / self.split_str / "lq",
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
        
        self.datapoints["image"] = lq_images


@DATAMODULES.register(name="lighteffect")
class LightEffectDataModule(DataModule):
    
    tasks: list[Task] = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LightEffect(split=Split.TRAIN, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
