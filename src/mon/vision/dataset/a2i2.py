#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements A2I2 datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "A2I2Haze", "A2I2HazeDet", "A2I2HazeDataModule", "A2I2HazeDetDataModule",
]

import argparse

import torch
from torch.utils.data import random_split

from mon import core, coreimage as ci
from mon.vision import constant, visualize
from mon.vision.dataset import base
from mon.vision.transform import transform as t
from mon.vision.typing import (
    CallableType, ClassLabelsType, Ints, ModelPhaseType, PathsType, PathType,
    Strs, TransformType, VisionBackendType,
)

# region ClassLabels

a2i2_classlabels = [
    { "name": "vehicle", "id": 0, "color": [  0,   0, 142] }
]

# endregion


# region Dataset

@constant.DATASET.register(name="a2i2-haze")
class A2I2Haze(base.ImageEnhancementDataset):
    """A2I2-Haze dataset.
    
    Args:
        name: A dataset name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "a2i2-haze",
        root            : PathType               = constant.DATA_DIR / "a2i2" / "haze",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = None,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / self.split / "dehazing" / "haze-images"
            for path in pbar.track(
                list(pattern.rglob("*.jpg")),
                description=f"Listing {self.__class__.__name__} {self.split} "
                            f"images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def list_labels(self):
        """List label files."""
        self.labels: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"Listing {self.__class__.__name__} {self.split} "
                            f"labels"
            ):
                path = core.Path(str(img.path).replace("haze-images", "hazefree-images"))
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )


@constant.DATASET.register(name="a2i2-haze-det")
class A2I2HazeDet(base.ImageDetectionDataset):
    """A2I2-Haze Detection.
    
    Args:
        name: A datamodule name.
        root: A root directory where the data is stored.
        split: The data split to use. One of: ["train", "val", "test"].
            Defaults to "train".
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        classlabels: A :class:`ClassLabels` object. Defaults to None.
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        cache_data: If True, cache data to disk for faster loading next time.
            Defaults to False.
        cache_images: If True, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Defaults to False.
        backend: The image processing backend. Defaults to VISION_BACKEND.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                    = "a2i2-haze-det",
        root            : PathType               = constant.DATA_DIR / "a2i2" / "haze",
        split           : str                    = "train",
        shape           : Ints                   = (3, 256, 256),
        classlabels     : ClassLabelsType | None = a2i2_classlabels,
        transform       : TransformType   | None = None,
        target_transform: TransformType   | None = None,
        transforms      : TransformType   | None = None,
        cache_data      : bool                   = False,
        cache_images    : bool                   = False,
        backend         : VisionBackendType      = constant.VISION_BACKEND,
        verbose         : bool                   = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            split            = split,
            shape            = shape,
            classlabels      = classlabels or a2i2_classlabels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
     
    def list_images(self):
        """List image files."""
        if self.split not in ["train", "dry-run", "test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train', 'dryrun', or 'test'. "
                f"Get: {self.split}."
            )

        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            if self.split == "train":
                pattern = self.root / "train" / "detection" / "hazefree-images"
            elif self.split == "dry-run":
                pattern = self.root / "dry-run" / "2023" / "images"
            elif self.split == "test":
                pattern = self.root / "test" / "images"
                
            for path in pbar.track(
                list(pattern.rglob("*.jpg")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
            ):
                self.images.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )
    
    def list_labels(self):
        """List label files."""
        ann_files = self.annotation_files()
        
        self.labels: list[base.DetectionsLabel] = []
        with core.rich.progress_bar() as pbar:
            for i in pbar.track(
                range(len(ann_files)),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                path = core.Path(ann_files[i])
                assert path.is_txt_file()
                
                f      = open(path, "r")
                labels = [x.split() for x in f.read().splitlines()]
                shape  = self.images[i].shape
                
                detections: list[base.DetectionLabel] = []
                for _, l in enumerate(labels):
                    id              = l[0]
                    box_xyxy        = torch.Tensor(l[1:5])
                    box_cxcywh_norm = ci.box_xyxy_to_cxcywhn(
                        box    = box_xyxy,
                        height = shape[0],
                        width  = shape[1],
                    )
                    
                    if id.isnumeric():
                        id = int(id)
                    elif isinstance(self.classlabels, base.ClassLabels):
                        id = self.classlabels.get_id(key="name", value=id)
                    else:
                        id = -1
                        
                    detections.append(
                        base.DetectionLabel(
                            id         = id,
                            bbox       = box_cxcywh_norm,
                            confidence = 0.0,
                        )
                    )
                
                self.labels.append(base.DetectionsLabel(detections))
            
    def annotation_files(self) -> PathsType:
        """Return the path to '.txt' annotation files."""
        ann_files = []
        for img in self.images:
            if self.split == "train":
                path = str(img.path).replace("hazefree-images", "hazefree-labels")
            else:
                path = str(img.path).replace("images", "labels")
            path = core.Path(path.replace(".jpg", ".txt"))
            ann_files.append(path)
        return ann_files

# endregion


# region Datamodule

@constant.DATAMODULE.register(name="a2i2-haze")
class A2I2HazeDataModule(base.DataModule):
    """A2I2-Haze datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        name            : str                  = "a2i2-haze",
        root            : PathType             = constant.DATA_DIR / "a2i2" / "haze",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            full_dataset = A2I2Haze(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            train_size   = int(0.8 * len(full_dataset))
            val_size     = len(full_dataset) - train_size
            self.train, self.val = random_split(
                full_dataset, [train_size, val_size]
            )
            self.classlabels = getattr(full_dataset, "classlabels", None)
            self.collate_fn  = getattr(full_dataset, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = A2I2Haze(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass


@constant.DATAMODULE.register(name="a2i2-haze-det")
class A2I2HazeDetDataModule(base.DataModule):
    """A2I2-Haze Detection datamodule.
    
    Args:
        name: A datamodule's name.
        root: A root directory where the data is stored.
        shape: The desired datapoint shape preferably in a channel-last format.
            Defaults to (3, 256, 256).
        transform: Transformations performing on the input.
        target_transform: Transformations performing on the target.
        transforms: Transformations performing on both the input and target.
        batch_size: The number of samples in one forward pass. Defaults to 1.
        devices: A list of devices to use. Defaults to 0.
        shuffle: If True, reshuffle the datapoints at the beginning of every
            epoch. Defaults to True.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Defaults to True.
    """
     
    def __init__(
        self,
        name            : str                  = "a2i2-haze-det",
        root            : PathType             = constant.DATA_DIR / "a2i2" / "haze",
        shape           : Ints                 = (3, 256, 256),
        transform       : TransformType | None = None,
        target_transform: TransformType | None = None,
        transforms      : TransformType | None = None,
        batch_size      : int                  = 1,
        devices         : Ints | Strs          = 0,
        shuffle         : bool                 = True,
        collate_fn      : CallableType  | None = None,
        verbose         : bool                 = True,
        *args, **kwargs
    ):
        super().__init__(
            name             = name,
            root             = root,
            shape            = shape,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            batch_size       = batch_size,
            devices          = devices,
            shuffle          = shuffle,
            collate_fn       = collate_fn,
            verbose          = verbose,
            *args, **kwargs
        )
        
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        if self.classlabels is None:
            self.load_classlabels()
    
    def setup(self, phase: ModelPhaseType | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
                - "training" : prepares :attr:'train' and :attr:'val'.
                - "testing"  : prepares :attr:'test'.
                - "inference": prepares :attr:`predict`.
                - None:      : prepares all.
                Defaults to None.
        """
        core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        phase = constant.ModelPhase.from_value(phase) if phase is not None else phase

        # Assign train/val datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TRAINING]:
            self.train = A2I2HazeDet(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = A2I2HazeDet(
                root             = self.root,
                split            = "dry-run",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # Assign test datasets for use in dataloader
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = A2I2HazeDet(
                root             = self.root,
                split            = "test",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.test, "classlabels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",  None)
        
        if self.classlabels is None:
            self.load_classlabels()

        self.summarize()
        
    def load_classlabels(self):
        """Load all the class-labels of the dataset."""
        self.classlabels = base.ClassLabels.from_value(value=a2i2_classlabels)

# endregion


# region Test

def test_a2i2_haze():
    cfg = {
        "name": "a2i2-haze",
            # A datamodule's name.
        "root": constant.DATA_DIR / "a2i2" / "haze",
            # A root directory where the data is stored.
        "shape": [3, 256, 256],
            # The desired datapoint shape preferably in a channel-last format.
            # Defaults to (3, 256, 256).
        "transform": None,
            # Transformations performing on the input.
        "target_transform": None,
            # Transformations performing on the target.
        "transforms": [
            t.Resize(size=[3, 256, 256]),
        ],
            # Transformations performing on both the input and target.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": constant.VISION_BACKEND,
            # The image processing backend. Defaults to VISION_BACKEND.
        "batch_size": 8,
            # The number of samples in one forward pass. Defaults to 1.
        "devices" : 0,
            # A list of devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the datapoints at the beginning of every epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm  = A2I2HazeDataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result              = {"image": input, "target": target}
    label               = [(m["name"]) for m in meta]
    visualize.imshow_enhancement(
        winname = "image",
        image   = result,
        label   = label
    )
    visualize.plt.show(block=True)

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="test-a2i2-haze", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test-a2i2-haze":
        test_a2i2_haze()

# endregion
