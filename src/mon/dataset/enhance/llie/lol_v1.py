#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL-v1 Datasets."""

from __future__ import annotations

__all__ = [
    "LOLv1",
    "LOLV1DataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "llie"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
DepthMapAnnotation  = core.DepthMapAnnotation
ImageAnnotation     = core.ImageAnnotation
MultimodalDataset   = core.MultimodalDataset


@DATASETS.register(name="lol_v1")
class LOLv1(MultimodalDataset):
    """LOL-v1 dataset consists of ``500`` low-light and normal-light image pairs.
    They are divided into ``485`` training pairs and 15 testing pairs. The
    low-light images contain noise produced during the photo capture process.
    Most of the images are indoor scenes. All the images have a resolution of
    ``400×600``.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "lol_v1" / self.split_str / "image",
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


@DATAMODULES.register(name="lol_v1")
class LOLV1DataModule(DataModule):
    
    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = LOLv1(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLv1(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLv1(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
