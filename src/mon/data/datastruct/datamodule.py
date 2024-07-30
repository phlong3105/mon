#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all datamodules."""

from __future__ import annotations

__all__ = [
    "DataModule",
]

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import lightning
from torch.utils import data

from mon import core
from mon.data.datastruct.dataset import base

console = core.console


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
    
    tasks = []
    
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
        self.devices        = core.to_list(devices)
        self.shuffle        = shuffle
        self.collate_fn     = collate_fn
        self.verbose        = verbose
        self.dataset_kwargs = kwargs | {
            "verbose": verbose,
        }

        train, val, test, predict = None, None, None, None
        if isinstance(datasets, dict):
            train   = datasets.pop("train")   if "train"   in datasets else None
            val     = datasets.pop("val")     if "val"     in datasets else None
            test    = datasets.pop("test")    if "test"    in datasets else None
            predict = datasets.pop("predict") if "predict" in datasets else None
            self.dataset_kwargs = kwargs | datasets
        elif isinstance(datasets, base.Dataset):
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
    def num_classes(self) -> int:
        """The number of classes in the dataset."""
        if hasattr(self.classlabels, "num_classes"):
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
            self.classlabels = self.classlabels or getattr(self.train, "classlabels", None)
            return data.DataLoader(
                dataset            = self.train,
                batch_size         = self.batch_size,
                shuffle            = self.shuffle,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = False,
                collate_fn         = getattr(self.train, "collate_fn", None) or self.collate_fn,
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
            self.classlabels = self.classlabels or getattr(self.val, "classlabels", None)
            return data.DataLoader(
                dataset            = self.val,
                batch_size         = self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = False,
                collate_fn         = getattr(self.val, "collate_fn", None) or self.collate_fn,
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
            self.classlabels = self.classlabels or getattr(self.test, "classlabels", None)
            return data.DataLoader(
                dataset            = self.test,
                batch_size         = 1,  # self.batch_size,
                shuffle            = False,
                num_workers        = self.num_workers,
                pin_memory         = True,
                drop_last          = False,
                collate_fn         = getattr(self.test, "collate_fn", None) or self.collate_fn,
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
    
    @property
    def can_log(self) -> bool:
        if self.verbose:
            if self.trainer is None:
                return True
            elif self.trainer is not None and self.trainer.global_rank == 0:
                return True
        return False
    
    @abstractmethod
    def prepare_data(self, *args, **kwargs):
        """Use this method to do things that might write to disk, or that need
        to be done only from a single GPU in distributed settings:
            - Download.
            - Tokenize.
        """
        pass
    
    @abstractmethod
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Use this method to do things on every device:
            - Count number of classes.
            - Build classlabels vocabulary.
            - Prepare train/val/test/predict splits.
            - Apply transformations.
            - Define :attr:`collate_fn` for your custom dataset.

        Args:
            stage: The running stage. One of:
                - ``'train'``  : prepares :attr:`train` and :attr:`val`.
                - ``'test'``   : prepares :attr:`test`.
                - ``'predict'``: prepares :attr:`predict`.
                - ``None``     : prepares all.
                - Default: ``None``.
        """
        pass
    
    def split_train_val(
        self,
        dataset    : base.Dataset,
        split_ratio: float = 0.8,
        full_train : bool  = True
    ):
        train_size       = int(split_ratio * len(dataset))
        val_size         = len(dataset) - train_size
        train, self.val  = data.random_split(dataset, [train_size, val_size])
        self.train       = dataset if full_train else train
        self.classlabels = getattr(dataset, "classlabels", None)
        self.collate_fn  = getattr(dataset, "collate_fn",  None)
    
    @abstractmethod
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        pass
    
    def summarize(self):
        """Print a summary."""
        table = core.rich.table.Table(header_style="bold magenta")
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
