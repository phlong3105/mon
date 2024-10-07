#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GT-Snow Datasets."""

from __future__ import annotations

__all__ = [
    "GTSnow",
    "GTSnowDataModule",
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


@DATASETS.register(name="gtsnow")
class GTSnow(MultimodalDataset):
 
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
            self.root / "gtsnow" / self.split_str / "image",
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
        
        # Reference images
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for img in pbar.track(
                sequence    = images,
                description = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            ):
                path = str(img.path)
                path = path[:-9] + "C-000.png"
                path = path.replace("/image/", "/ref/")
                path = core.Path(path)
                ref_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images


@DATAMODULES.register(name="gtsnow")
class GTSnowDataModule(DataModule):

    tasks: list[Task] = [Task.DESNOW]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()

        self.summarize()
