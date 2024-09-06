#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GT-Rain Datasets."""

from __future__ import annotations

__all__ = [
    "GTRain",
    "GTRainDataModule",
]

from typing import Literal

from mon import core
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance" / "derain"
DataModule          = core.DataModule
DatapointAttributes = core.DatapointAttributes
ImageAnnotation     = core.ImageAnnotation
ImageDataset        = core.ImageDataset


@DATASETS.register(name="gtrain")
class GTRain(ImageDataset):
    """GT-Rain dataset consists 26,124 train and 1,793 val pairs of rain/no-rain
    images.
    """
    
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
        patterns = [
            self.root / "gtrain" / self.split_str / "lq",
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
                path = str(img.path)
                if "Gurutto_1-2" in path:
                    path = path.replace("-R-", "-C-")
                else:
                    path = path[:-9] + "C-000.png"
                path = path.replace("/lq/", "/hq/")
                path = core.Path(path)
                hq_images.append(ImageAnnotation(path=path.image_file()))
        
        self.datapoints["image"]    = lq_images
        self.datapoints["hq_image"] = hq_images

            
@DATAMODULES.register(name="gtrain")
class GTRainDataModule(DataModule):
    
    tasks: list[Task] = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = GTRain(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = GTRain(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = GTRain(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
