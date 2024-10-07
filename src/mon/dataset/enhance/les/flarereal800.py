#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FlareReal800 Datasets."""

from __future__ import annotations

__all__ = [
    "FlareReal800",
    "FlareReal800DataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "les"
ClassLabels         = core.ClassLabels
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="flarereal800")
class FlareReal800(MultimodalDataset):
    """FlareReal800 dataset consists of 800 flare/clear image pairs."""
    
    tasks : list[Task]  = [Task.LES]
    splits: list[Split] = [Split.TRAIN, Split.VAL]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "flarereal800" / self.split_str / "image",
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
    

@DATAMODULES.register(name="flarereal800")
class FlareReal800DataModule(DataModule):
    
    tasks: list[Task] = [Task.LES]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = FlareReal800(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FlareReal800(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FlareReal800(split=Split.VAL,   **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
