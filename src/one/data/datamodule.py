#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all datamodules.
"""

from __future__ import annotations

from abc import ABCMeta
from abc import abstractmethod
from typing import Callable
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from one.core import console
from one.core import Devices
from one.core import EvalDataLoaders
from one.core import Int3T
from one.core import Table
from one.core import TrainDataLoaders
from one.core import VISION_BACKEND
from one.core import VisionBackend

__all__ = [
    "DataModule",
]


# MARK: - DataModule


class DataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """Base class for all data module present in the `torchkit.datasets`
    packages.
    
    Attributes:
        dataset_dir (str):
            Path to the dataset directory.
        name (str):
            Dataset name.
        shape (Int3T):
            Image shape as [H, W, C]. This is compatible with OpenCV format.
        batch_size (int):
            Number of training samples in one forward & backward pass.
            Default: `1`.
        devices (Device):
            The devices to use. Default: `1`.
        shuffle (bool):
             If `True`, reshuffle the data at every training epoch.
             Default: `True`.
        train (Dataset):
            Train dataset.
        val (Dataset):
            Val dataset.
        test (Dataset):
            Test dataset.
        predict (Dataset):
            Predict dataset.
        class_labels (ClassLabels, optional):
            `ClassLabels` object contains all class-labels defined in the
            dataset.
        collate_fn (callable, optional):
            Collate function used to fused input items together when using
            `batch_size > 1`.
        transforms (callable, optional):
            Function/transform that takes input sample and its target as
            entry and returns a transformed version.
        transform (callable, optional):
            Function/transform that takes input sample as entry and returns
            a transformed version.
        target_transform (callable, optional):
            Function/transform that takes in the target and transforms it.
    """

    # MARK: Magic Functions
    
    def __init__(
        self,
        dataset_dir     : str,
        name            : str,
        shape           : Int3T,
        batch_size      : int                     = 1,
        devices         : Devices                 = 1,
        shuffle         : bool                    = True,
        collate_fn      : Optional[Callable]      = None,
        vision_backend  : Optional[VisionBackend] = None,
        transforms      : Optional[Callable]      = None,
        transform       : Optional[Callable]      = None,
        target_transform: Optional[Callable]      = None,
        *args, **kwargs
    ):
        super().__init__()
        self.dataset_dir      = dataset_dir
        self.name             = name
        self.shape            = shape
        self.batch_size       = batch_size
        self.devices          = devices
        self.shuffle          = shuffle
        self.train            = None
        self.val              = None
        self.test             = None
        self.predict          = None
        self.class_labels     = None
        self.collate_fn       = collate_fn
        self.transforms       = transforms
        self.transform        = transform
        self.target_transform = target_transform
        
        if vision_backend in VisionBackend:
            self.vision_backend = vision_backend
        else:
            self.vision_backend = VISION_BACKEND
            
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
        if self.class_labels is not None:
            return self.class_labels.num_classes()
        return 0
    
    @property
    def num_workers(self) -> int:
        """Return number of workers used in the data loading pipeline."""
        # NOTE: Set `num_workers` = 4 * the number of gpus to avoid bottleneck
        return 4 * len(self.devices)
        # return 4  # os.cpu_count()

    @property
    def train_dataloader(self) -> Optional[TrainDataLoaders]:
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
    def val_dataloader(self) -> Optional[EvalDataLoaders]:
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
    def test_dataloader(self) -> Optional[EvalDataLoaders]:
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
    def predict_dataloader(self) -> Optional[EvalDataLoaders]:
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
    def setup(self, phase: Optional["ModelState"] = None):
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
            phase (ModelState, optional):
                Stage to use: [None, ModelState.TRAINING, ModelState.TESTING].
                Set to `None` to setup all train, val, and test data.
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
