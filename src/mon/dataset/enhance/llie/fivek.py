#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIT-Adobe FiveK Datasets."""

from __future__ import annotations

__all__ = [
    "FiveKA",
    "FiveKADataModule",
    "FiveKB",
    "FiveKBDataModule",
    "FiveKC",
    "FiveKCDataModule",
    "FiveKD",
    "FiveKDDataModule",
    "FiveKE",
    "FiveKEDataModule",
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


@DATASETS.register(name="fivek_a")
class FiveKA(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert A ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_a" / self.split_str / "image",
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


@DATASETS.register(name="fivek_b")
class FiveKB(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert B ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_b" / self.split_str / "image",
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


@DATASETS.register(name="fivek_c")
class FiveKC(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert C ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_c" / self.split_str / "image",
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


@DATASETS.register(name="fivek_d")
class FiveKD(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert D ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_d" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} lq images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATASETS.register(name="fivek_e")
class FiveKE(MultimodalDataset):
    """MIT-Adobe FiveK dataset with Expert E ground-truth. It consists of
    5,000 low/high image pairs.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        
    def get_data(self):
        patterns = [
            self.root / "fivek_e" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} lq images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATAMODULES.register(name="fivek_a")
class FiveKADataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKA(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKA(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKA(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_b")
class FiveKBDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKB(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKB(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKB(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_c")
class FiveKCDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKC(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKC(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKC(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_d")
class FiveKDDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = FiveKD(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKD(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKD(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="fivek_e")
class FiveKEDataModule(DataModule):

    tasks: list[Task] = [Task.LLIE]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = FiveKE(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = FiveKE(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = FiveKE(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
