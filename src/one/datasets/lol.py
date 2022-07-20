#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LoL dataset and datamodule.
"""

from __future__ import annotations

import inspect
import os
import sys
from glob import glob
from pathlib import Path
from typing import Callable
from typing import Union

import matplotlib.pyplot as plt

from one.core import console
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import Ints
from one.core import is_image_file
from one.core import ModelState
from one.core import progress_bar
from one.core import VISION_BACKEND
from one.core import VisionBackend
from one.datasets.utils import ClassLabel
from one.datasets.utils import DataModule
from one.datasets.utils import Image
from one.datasets.utils import ImageEnhancementDataset
from one.utils import data_dir
from one.visualize import show_images


# MARK: - Module

@DATASETS.register(name="lol")
class LoL(ImageEnhancementDataset):
    """LoL dataset consists of multiple datasets related to low-light
    image enhancement task.
    
    Args:
        root (str):
            Root directory of dataset.
        split (str):
            Split to use. One of: ["train", "val", "test"].
        shape (Ints):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        cache_data (bool):
            If `True`, cache data to disk for faster loading next time.
            Default: `False`.
        cache_images (bool):
            If `True`, cache images into memory for faster training
            (WARNING: large datasets may exceed system RAM). Default: `False`.
        backend (VisionBackend, str, int):
            Vision backend to process image. Default: `VISION_BACKEND`.
        verbose (bool):
            Verbosity. Default: `True`.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        split           : str                            = "train",
        shape           : Ints                           = (720, 1280, 3),
        class_labels    : Union[ClassLabel, str, Path]   = None,
        transform       : Union[Callable, None]          = None,
        target_transform: Union[Callable, None]          = None,
        transforms      : Union[Callable, None]          = None,
        cache_data      : bool                           = False,
        cache_images    : bool                           = False,
        backend         : Union[VisionBackend, str, int] = VISION_BACKEND,
        verbose         : bool                           = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_labels     = class_labels,
            transform        = transform,
            target_transform = target_transform,
            transforms       = transforms,
            cache_data       = cache_data,
            cache_images     = cache_images,
            backend          = backend,
            verbose          = verbose,
            *args, **kwargs
        )
 
    # MARK: Configure
    
    def list_images(self):
        """List image files."""
        self.images: list[Image] = []
        with progress_bar() as pbar:
            for path in pbar.track(
                glob(os.path.join(self.root, self.split, "low", "*.png")),
                description=f"[bright_yellow]Listing LoL {self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """List label files."""
        self.labels: list[Image] = []
        with progress_bar() as pbar:
            for img in pbar.track(
                self.images,
                description=f"[bright_yellow]Listing LoL {self.split} labels"
            ):
                path = img.path.replace("low", "high")
                if is_image_file(path):
                    self.labels.append(Image(path=path, backend=self.backend))
    
    def filter(self):
        """Filter unwanted samples."""
        pass
        

@DATAMODULES.register(name="lol")
class LoLDataModule(DataModule):
    """LoL DataModule."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        root: str = os.path.join(data_dir, "lol"),
        name: str = "lol",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
    # MARK: Prepare Data
    
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.class_labels is None:
            self.load_class_labels()
    
    def setup(self, model_state: Union[ModelState, None] = None):
        """There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            model_state (ModelState, None):
                ModelState to use: [None, ModelState.TRAINING, ModelState.TESTING]. Set to
                "None" to setup all train, val, and test data. Default: `None`.
        """
        console.log(f"Setup [red]LoL[/red] datasets.")
        
        # NOTE: Assign train/val datasets for use in dataloaders
        if model_state in [None, ModelState.TRAINING]:
            self.train = LoL(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            self.val = LoL(
                root=self.dataset_dir, split="val", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.train, "class_labels", None)
            self.collate_fn   = getattr(self.train, "collate_fn",   None)
            
        # NOTE: Assign test datasets for use in dataloader(s)
        if model_state in [None, ModelState.TESTING]:
            self.test = LoL(
                root=self.dataset_dir, split="test", **self.dataset_kwargs
            )
            self.class_labels = getattr(self.test, "class_labels", None)
            self.collate_fn   = getattr(self.test, "collate_fn",   None)
        
        if self.class_labels is None:
            self.load_class_labels()

        self.summarize()
        
    def load_class_labels(self):
        """Load ClassLabels."""
        pass


# MARK: - Test

def test_lol():
    cfgs = {
        "root": os.path.join(data_dir, "lol"),
           # Root directory of dataset.
        "name": "lol",
            # Dataset's name.
        "shape": [512, 512, 3],
            # Image shape as [H, W, C], [H, W], or [S, S].
        "transform": None,
            # Functions/transforms that takes in an input sample and returns a
            # transformed version. E.g, `transforms.RandomCrop`.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": None,
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": False,
            # If `True`, cache data to disk for faster loading next time.
            # Default: `False`.
        "cache_images": False,
            # If `True`, cache images into memory for faster training
            # (WARNING: large datasets may exceed system RAM). Default: `False`.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Default: `VISION_BACKEND`.
        "batch_size": 4,
            # Number of samples in one forward & backward pass.
        "devices" : 0,
            # The devices to use.
        "shuffle": True,
            # If `True`, reshuffle the data at every training epoch.
        "verbose": True,
            # Verbosity.
    }
    dm = LoLDataModule(**cfgs)
    dm.setup()
    # NOTE: Visualize labels
    if dm.class_labels:
        dm.class_labels.print()
    # NOTE: Visualize one sample
    data_iter            = iter(dm.train_dataloader)
    input, target, shape = next(data_iter)
    show_images(images=input,  nrow=2, denormalize=True)
    show_images(images=target, nrow=2, denormalize=True, figure_num=1)
    plt.show(block=True)


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]


if __name__ == "__main__":
    test_lol()
