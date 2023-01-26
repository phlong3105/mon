#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements KODAS datasets and datamodules."""

from __future__ import annotations

__all__ = [
    "KODASLoL19", "KODASLoL19DataModule",
]

import argparse

from mon import core
from mon.vision import constant, visualize
from mon.vision.dataset import base
from mon.vision.transform import transform as t
from mon.vision.typing import (
    CallableType, ClassLabelsType, Ints, ModelPhaseType, PathType,
    Strs, TransformType, VisionBackendType,
)


# region Dataset

@constant.DATASET.register(name="kodas-lol19")
class KODASLoL19(base.ImageEnhancementDataset):
    """KODAS-LoL dataset.
    
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
        name            : str                    = "kodas-lol19",
        root            : PathType               = constant.DATA_DIR / "kodas" / "lol19",
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
        if self.split not in ["train", "val", "test"]:
            core.console.log(
                f"{self.__class__.__name__} dataset only supports "
                f":param:`split`: 'train', 'val', or 'test'. Get: {self.split}."
            )
            
        self.images: list[base.ImageLabel] = []
        with core.rich.progress_bar() as pbar:
            pattern = self.root / self.split / "low"
            for path in pbar.track(
                list(pattern.rglob("*.png")),
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} images"
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
                description=f"Listing {self.__class__.__name__} "
                            f"{self.split} labels"
            ):
                path = core.Path(str(img.path).replace("low", "high"))
                self.labels.append(
                    base.ImageLabel(path=path, backend=self.backend)
                )

# endregion


# region Datamodule

@constant.DATAMODULE.register(name="kodas-lol19")
class KODASLoL19DataModule(base.DataModule):
    """KODAS-LoL DataModule.
    
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
        name            : str                  = "kodas-lol19",
        root            : PathType             = constant.DATA_DIR / "kodas" / "lol19",
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
        """"Use this method to do things that might write to disk, or that need
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
            self.train = KODASLoL19(
                root             = self.root,
                split            = "train",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.val = KODASLoL19(
                root             = self.root,
                split            = "val",
                shape            = self.shape,
                transform        = self.transform,
                target_transform = self.target_transform,
                transforms       = self.transforms,
                verbose          = self.verbose,
                **self.dataset_kwargs
            )
            self.classlabels = getattr(self.train, "classlabels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",  None)
            
        # Assign test datasets for use in dataloaders
        if phase in [None, constant.ModelPhase.TESTING]:
            self.test = KODASLoL19(
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
        pass

# endregion


# region Test

def test_kodas_lol19():
    cfg = {
        "name": "kodas-lol19",
            # A datamodule's name.
        "root": constant.DATA_DIR / "kodas" / "lol19",
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
    dm  = KODASLoL19DataModule(**cfg)
    dm.setup()
    # Visualize labels
    if dm.classlabels:
        dm.classlabels.print()
    # Visualize one sample
    data_iter           = iter(dm.train_dataloader)
    input, target, meta = next(data_iter)
    result              = {"image" : input, "target": target}
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
    parser.add_argument("--task", type=str , default="test-kodas-lol", help="The task to run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.task == "test-kodas-lol19":
        test_kodas_lol19()

# endregion
