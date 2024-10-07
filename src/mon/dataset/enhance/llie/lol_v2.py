#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL-v2 Datasets."""

from __future__ import annotations

__all__ = [
    "LOLv2Real",
    "LOLV2RealDataModule",
    "LOLv2Synthetic",
    "LOLV2SyntheticDataModule",
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


# region Dataset

@DATASETS.register(name="lol_v2_real")
class LOLv2Real(MultimodalDataset):
    """LOL-v2 Real (VE-LOL) dataset consists of ``500`` low-light and
    normal-light image pairs. They are divided into ``400`` training pairs and
    ``100`` testing pairs. The low-light images contain noise produced during
    the photo capture process. Most of the images are indoor scenes. All the
    images have a resolution of ``400×600``.
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
            self.root / "lol_v2_real" / self.split_str / "image",
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


@DATASETS.register(name="lol_v2_synthetic")
class LOLv2Synthetic(MultimodalDataset):
    """LOL-v2 Synthetic (VE-LOL-Syn) dataset consists of ``1000`` low-light and
    normal-light image pairs. They are divided into ``900`` training pairs and
    ``100`` testing pairs. The low-light images contain noise produced during
    the photo capture process. Most of the images are indoor scenes. All the
    images have a resolution of ``400×600``.
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
            self.root / "lol_v2_synthetic" / self.split_str / "image",
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


@DATAMODULES.register(name="lol_v2_real")
class LOLV2RealDataModule(DataModule):
    
    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = LOLv2Real(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLv2Real(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLv2Real(split=Split.TEST, **self.dataset_kwargs)
        
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
            self.train = LOLv2Synthetic(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLv2Synthetic(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLv2Synthetic(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
