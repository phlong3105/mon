#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unsupervised LOL Datasets."""

from __future__ import annotations

__all__ = [
    "ULOL",
    "ULOLMixDataModule",
]

from typing import Literal

from mon import core
from mon.dataset.enhance.llie.lol_v1 import LOLv1
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "llie"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="ulol")
class ULOL(MultimodalDataset):

    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "dicm"             / "test"         / "image",
            self.root / "fusion"           / "test"         / "image",
            self.root / "lime"             / "test"         / "image",
            self.root / "lol_v1"           / self.split_str / "image",
            self.root / "lol_v1"           / self.split_str / "image",
            self.root / "lol_v2_real"      / self.split_str / "image",
            self.root / "lol_v2_real"      / self.split_str / "image",
            self.root / "lol_v2_synthetic" / self.split_str / "image",
            self.root / "lol_v2_synthetic" / self.split_str / "image",
            self.root / "mef"              / "test"         / "image",
            self.root / "npe"              / "test"         / "image",
            self.root / "sice_mix"         / self.split_str / "image",
            self.root / "sice_mix_v2"      / self.split_str / "image",
            self.root / "vv"               / "test"         / "image",
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
            self.val   = LOLv1(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLv1(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
