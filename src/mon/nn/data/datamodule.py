#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all data modules."""

from __future__ import annotations

__all__ = [
    "DataModule",
]

from abc import ABC, abstractmethod
from typing import Any, Callable

import lightning
from torch.utils import data

from mon.core import builtins, console, rich
from mon.globals import ModelPhase
from mon.nn.data import dataset, label


# region DataModule

class DataModule(lightning.LightningDataModule, ABC):
    """The base class for all datamodules.
    
    See Also: :class:`lightning.LightningDataModule`.
    
    Attributes:
        dataset_kwargs: A :class:`dict` containing datasets' default arguments.
            Example usage: train = Dataset(split='train', **self.dataset_kwargs)
        
    Args:
        batch_size: The number of samples in one forward pass. Default: ``1``.
        devices: A list of devices to use. Default: ``0``.
        shuffle: If ``True``, reshuffle the datapoints at the beginning of every
            epoch. Default: ``True``.
        collate_fn: The function used to fused datapoint together when using
            :param:`batch_size` > 1.
        verbose: Verbosity. Default: ``True``.
    """
    
    def __init__(
        self,
        datasets  : Any = None,
        batch_size: int = 1,
        devices   : int | str | list[int | str] = 0,
        shuffle   : bool     = True,
        collate_fn: Callable = None,
        verbose   : bool     = True,
        *args, **kwargs
    ):
        super().__init__()
        self.batch_size     = batch_size
        self.devices        = devices
        self.shuffle        = shuffle
        self.collate_fn     = collate_fn
        self.verbose        = verbose
        self.dataset_kwargs = kwargs | {
            "verbose": self.verbose,
        }

        train, val, test, predict = None, None, None, None
        if isinstance(datasets, dict):
            train   = datasets.pop("train")   if "train"   in datasets else None
            val     = datasets.pop("val")     if "val"     in datasets else None
            test    = datasets.pop("test")    if "test"    in datasets else None
            predict = datasets.pop("predict") if "predict" in datasets else None
            self.dataset_kwargs = kwargs | datasets
        elif isinstance(datasets, dataset.Dataset):
            train   = datasets
            val     = datasets
            test    = datasets
            predict = datasets
        
        self.train       = train
        self.val         = val
        self.test        = test
        self.predict     = predict
        self.classlabels = None
    
    @property
    def devices(self) -> list:
        """The :class:`list` of devices."""
        return self._devices
    
    @devices.setter
    def devices(self, devices: int | str | list[int | str]):
        self._devices = builtins.to_list(devices)
    
    @property
    def num_classes(self) -> int:
        """The number of classes in the dataset."""
        if isinstance(self.classlabels, label.ClassLabels):
            return self.classlabels.num_classes()
        return 0
    
    @property
    def num_workers(self) -> int:
        """The number of workers used in the data loading pipeline.
        Set to: 4 * the number of :attr:`devices` to avoid a bottleneck.
        """
        return 4 * len(self.devices)
        # return 4  # os.cpu_count()
    
    @property
    def train_dataloader(self) -> data.DataLoader | None:
        """Returns a :class:`torch.utils.data.DataLoader` object if
        :attr:`train` exists. Otherwise, returns ``None``.
        """
        if self.train:
            return data.DataLoader(
                dataset            = self.train,
                batch_size         = self.batch_size,
                shuffle            = self.shuffle,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = False,
                collate_fn         = self.collate_fn,
                # prefetch_factor  = 4,
                persistent_workers = True,
            )
        return None
    
    @property
    def val_dataloader(self) -> data.DataLoader | None:
        """Returns a :class:`torch.utils.data.DataLoader` object if
        :attr:`val` exists. Otherwise, returns ``None``.
        """
        if self.val:
            return data.DataLoader(
                dataset            = self.val,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = False,
                collate_fn         = self.collate_fn,
                # prefetch_factor  = 4,
                persistent_workers = True,
            )
        return None
    
    @property
    def test_dataloader(self) -> data.DataLoader | None:
        """Returns a :class:`torch.utils.data.DataLoader` object if
        :attr:`test` exists. Otherwise, returns ``None``.
        """
        if self.test:
            return data.DataLoader(
                dataset            = self.test,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = False,
                collate_fn         = self.collate_fn,
                # prefetch_factor  = 4,
                persistent_workers = True,
            )
        return None
    
    @property
    def predict_dataloader(self):
        """Returns a :class:`torch.utils.data.DataLoader` object if
        :attr:`predict` exists. Otherwise, returns ``None``.
        """
        if self.predict:
            return data.DataLoader(
                dataset            = self.predict,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = True,
                collate_fn         = self.collate_fn,
                # prefetch_factor  = 4,
                persistent_workers = True,
            )
        return None
    
    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        pass
    
    @abstractmethod
    def setup(self, phase: ModelPhase | str | None = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            phase: The model phase. One of:
            
                - ``'training'`` : prepares :attr:`train` and :attr:`val`.
                - ``'testing'``  : prepares :attr:`test`.
                - ``'inference'``: prepares :attr:`predict`.
                - ``None``       : prepares all.
                - Default: ``None``.
        """
        pass
    
    @abstractmethod
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass
    
    def summarize(self):
        """Print a summary."""
        table = rich.table.Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Desc")
        table.add_row("1", "train",       f"{len(self.train) if self.train is not None else None}")
        table.add_row("2", "val",         f"{len(self.val) if self.val is not None else None}")
        table.add_row("3", "test",        f"{len(self.test) if self.test is not None else None}")
        table.add_row("4", "classlabels", f"{self.classlabels.num_classes if self.classlabels is not None else None}")
        table.add_row("5", "batch_size",  f"{self.batch_size}")
        table.add_row("6", "num_workers", f"{self.num_workers}")
        console.log(table)

# endregion
