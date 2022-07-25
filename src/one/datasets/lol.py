#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoL dataset and datamodule.
"""

from __future__ import annotations

import glob
import inspect
import os
import sys

import matplotlib.pyplot as plt

from one.core import ClassLabel_
from one.core import console
from one.core import DATA_DIR
from one.core import DataModule
from one.core import DATAMODULES
from one.core import DATASETS
from one.core import Image
from one.core import ImageEnhancementDataset
from one.core import Ints
from one.core import is_image_file
from one.core import ModelPhase
from one.core import ModelPhase_
from one.core import progress_bar
from one.core import Transforms_
from one.core import VISION_BACKEND
from one.core import VisionBackend_
from one.plot import show_images


# MARK: - Module ---------------------------------------------------------------

@DATASETS.register(name="lol")
class LoL(ImageEnhancementDataset):
    """
    LoL dataset consists of multiple datasets related to low-light image
    enhancement task.
    
    Args:
        root (str): Root directory of dataset.
        split (str): Split to use. One of: ["train", "val", "test"].
        shape (Ints): Image shape as [H, W, C], [H, W], or [S, S].
        class_label (ClassLabel_ | None): ClassLabel object. Defaults to None.
        transform (Transforms_ | None): Functions/transforms that takes in an
            input sample and returns a transformed version.
            E.g, `transforms.RandomCrop`.
        target_transform (Transforms_ | None): Functions/transforms that takes
            in a target and returns a transformed version.
        transforms (Transforms_ | None): Functions/transforms that takes in an
            input and a target and returns the transformed versions of both.
        cache_data (bool): If True, cache data to disk for faster loading next
            time. Defaults to False.
        cache_images (bool): If True, cache images into memory for faster
            training (WARNING: large datasets may exceed system RAM).
            Defaults to False.
        backend (VisionBackend_): Vision backend to process image.
            Defaults to VISION_BACKEND.
        verbose (bool): Verbosity. Defaults to True.
    """
    
    def __init__(
        self,
        root            : str,
        split           : str                = "train",
        shape           : Ints               = (720, 1280, 3),
        class_label     : ClassLabel_ | None = None,
        transform       : Transforms_ | None = None,
        target_transform: Transforms_ | None = None,
        transforms      : Transforms_ | None = None,
        cache_data      : bool               = False,
        cache_images    : bool               = False,
        backend         : VisionBackend_     = VISION_BACKEND,
        verbose         : bool               = True,
        *args, **kwargs
    ):
        super().__init__(
            root             = root,
            split            = split,
            shape            = shape,
            class_label      = class_label,
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
        """
        List image files.
        """
        self.images: list[Image] = []
        with progress_bar() as pbar:
            pattern = os.path.join(self.root, self.split, "low", "*.png")
            for path in pbar.track(
                glob.glob(pattern),
                description=f"[bright_yellow]Listing LoL {self.split} images"
            ):
                self.images.append(Image(path=path, backend=self.backend))
    
    def list_labels(self):
        """
        List label files.
        """
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
        """
        Filter unwanted samples.
        """
        pass
        

@DATAMODULES.register(name="lol")
class LoLDataModule(DataModule):
    """
    LoL DataModule.
    """
    
    def __init__(
        self,
        root: str = os.path.join(DATA_DIR, "lol"),
        name: str = "lol",
        *args, **kwargs
    ):
        super().__init__(root=root, name=name, *args, **kwargs)
        
    def prepare_data(self, *args, **kwargs):
        """
        Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        if self.class_label is None:
            self.load_class_label()
    
    def setup(self, phase: ModelPhase_ | None = None):
        """
        There are also data operations you might want to perform on every GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelPhase_ | None):
                Stage to use: [None, ModelPhase.TRAINING, ModelPhase.TESTING].
                Set to None to setup all train, val, and test data.
                Defaults to None.
        """
        console.log(f"Setup [red]LoL[/red] datasets.")
        phase = ModelPhase.from_value(phase)
        
        # Assign train/val datasets for use in dataloaders
        if phase in [None, ModelPhase.TRAINING]:
            self.train = LoL(
                root=self.dataset_dir, split="train", **self.dataset_kwargs
            )
            self.val = LoL(
                root=self.dataset_dir, split="val", **self.dataset_kwargs
            )
            self.class_label = getattr(self.train, "class_labels", None)
            self.collate_fn  = getattr(self.train, "collate_fn",   None)
            
        # Assign test datasets for use in dataloader(s)
        if phase in [None, ModelPhase.TESTING]:
            self.test = LoL(
                root=self.dataset_dir, split="test", **self.dataset_kwargs
            )
            self.class_label = getattr(self.test, "class_labels", None)
            self.collate_fn  = getattr(self.test, "collate_fn",   None)
        
        if self.class_label is None:
            self.load_class_label()

        self.summarize()
        
    def load_class_label(self):
        """Load ClassLabels."""
        pass


# MARK: - Test -----------------------------------------------------------------

def test_lol():
    cfgs = {
        "root": os.path.join(DATA_DIR, "lol"),
           # Root directory of dataset.
        "name": "lol",
            # Dataset's name.
        "shape": [3, 512, 512],
            # Image shape as [H, W, C], [H, W], or [S, S].
        "transform": None,
            # Functions/transforms that takes in an input sample and returns a
            # transformed version.
        "target_transform": None,
            # Functions/transforms that takes in a target and returns a
            # transformed version.
        "transforms": None,
            # Functions/transforms that takes in an input and a target and
            # returns the transformed versions of both.
        "cache_data": False,
            # If True, cache data to disk for faster loading next time.
            # Defaults to False.
        "cache_images": False,
            # If True, cache images into memory for faster training (WARNING:
            # large datasets may exceed system RAM). Defaults to False.
        "backend": VISION_BACKEND,
            # Vision backend to process image. Defaults to VISION_BACKEND.
        "batch_size": 4,
            # Number of samples in one forward & backward pass. Defaults to 1.
        "devices" : 0,
            # The devices to use. Defaults to 0.
        "shuffle": True,
            # If True, reshuffle the data at every training epoch.
            # Defaults to True.
        "verbose": True,
            # Verbosity. Defaults to True.
    }
    dm = LoLDataModule(**cfgs)
    dm.setup()
    # Visualize labels
    if dm.class_label:
        dm.class_label.print()
    # Visualize one sample
    data_iter            = iter(dm.train_dataloader)
    input, target, shape = next(data_iter)
    show_images(images=input,  nrow=2, denormalize=True)
    show_images(images=target, nrow=2, denormalize=True, figure_num=1)
    plt.show(block=True)


# MARK: - All ------------------------------------------------------------------

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]


# MARK: - Main -----------------------------------------------------------------

if __name__ == "__main__":
    test_lol()
