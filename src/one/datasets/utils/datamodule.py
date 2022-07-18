#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all datamodules.
"""

from __future__ import annotations

import inspect
import sys
from abc import ABCMeta
from abc import abstractmethod
from typing import Callable
from typing import Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from one.core import console
from one.core import Devices
from one.core import EvalDataLoaders
from one.core import Int3T
from one.core import Table
from one.core import TrainDataLoaders
from one.data import ClassLabels


# MARK: - DataModule

class DataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """Base class for all datamodules.
    
    Args:
        root (str):
            Root directory of dataset.
        name (str):
            Dataset's name.
        shape (Int3T):
            Image shape as [H, W, C], [H, W], or [S, S].
        transform (Callable, list, dict, None):
            Functions/transforms that takes in an input sample and returns a
            transformed version. E.g, `transforms.RandomCrop`.
        target_transform (Callable, list, dict, None):
            Functions/transforms that takes in a target and returns a
            transformed version.
        transforms (Callable, list, dict, None):
            Functions/transforms that takes in an input and a target and
            returns the transformed versions of both.
        batch_size (int):
            Number of training samples in one forward & backward pass.
            Default: `1`.
        devices (Device):
            The devices to use. Default: `0`.
        shuffle (bool):
             If `True`, reshuffle the data at every training epoch.
             Default: `True`.
        collate_fn (Callable, None):
            Collate function used to fused input items together when using
            `batch_size > 1`.
        verbose (bool):
            Verbosity. Default: `True`.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        root            : str,
        name            : str,
        shape           : Int3T,
        transform       : Union[Callable, list, dict, None] = None,
        target_transform: Union[Callable, list, dict, None] = None,
        transforms      : Union[Callable, list, dict, None] = None,
        batch_size      : int                               = 1,
        devices         : Devices                           = 0,
        shuffle         : bool                              = True,
        collate_fn      : Union[Callable, None]             = None,
        verbose         : bool                              = True,
        *args, **kwargs
    ):
        super().__init__()
        self.root             = root
        self.name             = name
        self.shape            = shape
        self.transform        = transform
        self.target_transform = target_transform
        self.transforms       = transforms
        self.batch_size       = batch_size
        self.devices          = devices
        self.shuffle          = shuffle
        self.collate_fn       = collate_fn
        self.verbose          = verbose
        self.dataset_kwargs   = kwargs
        self.train            = None
        self.val              = None
        self.test             = None
        self.predict          = None
        self.class_labels     = None
   
    # MARK: Property
    
    @property
    def devices(self) -> list:
        return self._devices

    @devices.setter
    def devices(self, devices: Devices):
        if isinstance(devices, (int, str)):
            devices = [devices]
        elif isinstance(devices, tuple):
            devices = list(devices)
        self._devices = devices
    
    @property
    def num_classes(self) -> int:
        """Return number of classes in the dataset."""
        if isinstance(self.class_labels, ClassLabels):
            return self.class_labels.num_classes()
        return 0
    
    @property
    def num_workers(self) -> int:
        """Return number of workers used in the data loading pipeline."""
        # NOTE: Set `num_workers` = 4 * the number of gpus to avoid bottleneck
        return 4 * len(self.devices)
        # return 4  # os.cpu_count()

    @property
    def train_dataloader(self) -> Union[TrainDataLoaders, None]:
        """Implement one or more PyTorch DataLoaders for training."""
        if self.train:
            return DataLoader(
                dataset            = self.train,
                batch_size         = self.batch_size,
                shuffle            = self.shuffle,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def val_dataloader(self) -> Union[EvalDataLoaders, None]:
        """Implement one or more PyTorch DataLoaders for validation."""
        if self.val:
            return DataLoader(
                dataset            = self.val,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def test_dataloader(self) -> Union[EvalDataLoaders, None]:
        """Implement one or more PyTorch DataLoaders for testing."""
        if self.test:
            return DataLoader(
                dataset            = self.test,
                batch_size         = 1,  # self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None

    @property
    def predict_dataloader(self) -> Union[EvalDataLoaders, None]:
        """Implement one or multiple PyTorch DataLoaders for prediction."""
        if self.predict:
            return DataLoader(
                dataset            = self.predict,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None
    
    # MARK: Prepare Data

    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk or that need
        to be done only from a single GPU in distributed settings.
            - Download.
            - Tokenize.
        """
        pass
    
    @abstractmethod
    def setup(self, phase: Union["ModelState", None] = None):
        """There are also data operations you might want to perform on every
        GPU.

        Todos:
            - Count number of classes.
            - Build class_labels vocabulary.
            - Perform train/val/test splits.
            - Apply transforms (defined explicitly in your datamodule or
              assigned in init).
            - Define collate_fn for you custom dataset.

        Args:
            phase (ModelState, None):
                Stage to use: [None, ModelState.TRAINING, ModelState.TESTING].
                Set to `None` to setup all train, val, and test data.
                Default: `None`.
        """
        pass

    @abstractmethod
    def load_class_labels(self):
        """Load ClassLabels."""
        pass
    
    # MARK: Utils
    
    def summarize(self):
        table = Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Desc")
        
        table.add_row("1", "train",        f"{len(self.train)               if self.train is not None else None}")
        table.add_row("2", "val",          f"{len(self.val)                 if self.val   is not None else None}")
        table.add_row("3", "test",         f"{len(self.test)                if self.test  is not None else None}")
        table.add_row("4", "class_labels", f"{self.class_labels.num_classes if self.class_labels is not None else None}")
        table.add_row("5", "batch_size",   f"{self.batch_size}")
        table.add_row("6", "shape",        f"{self.shape}")
        table.add_row("7", "num_workers",  f"{self.num_workers}")
        console.log(table)


# MARK: - Main

__all__ = [
    name for name, value in inspect.getmembers(
        sys.modules[__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__
    )
]
