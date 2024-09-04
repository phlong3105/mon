#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DataModule.

This module implements the base class for all datamodules.
"""

from __future__ import annotations

__all__ = [
    "DataModule",
]

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal

import lightning
from torch.utils import data

from mon.core import dtype, rich
from mon.core.data.dataset import base
from mon.globals import Task


# region DataModule

class DataModule(lightning.LightningDataModule, ABC):
    """The base class for all datamodules.
    
    Attributes:
        dataset_kwargs: A :obj:`dict` containing datasets' default arguments.
            Example usage: train = Dataset(split='train', **self.dataset_kwargs)
        
    Args:
        batch_size: The number of samples in one forward pass. Default: ``1``.
        devices: A list of devices to use. Default: ``0``.
        shuffle: If ``True``, reshuffle the datapoints at the beginning of every
            epoch. Default: ``True``.
        collate_fn: The function used to fused datapoint together when using
            :obj:`batch_size`> `1``.
        verbose: Verbosity. Default: ``True``.
    """
    
    tasks: list[Task] = []
    
    def __init__(
        self,
        datasets  : Any      = None,
        batch_size: int      = 1,
        devices   : int | str | list[int | str] = 0,
        shuffle   : bool     = True,
        collate_fn: Callable = None,
        verbose   : bool     = True,
        *args, **kwargs
    ):
        super().__init__()
        self.batch_size     = batch_size
        self.devices        = dtype.to_list(devices)
        self.shuffle        = shuffle
        self.collate_fn     = collate_fn
        self.verbose        = verbose
        self.dataset_kwargs = kwargs | {"verbose": verbose}

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
    def num_workers(self) -> int:
        """The number of workers used in the data loading pipeline.
        Set to: 4 * the number of :obj:`devices` to avoid a bottleneck.
        """
        return 4 * len(self.devices)
        # return 4  # os.cpu_count()
    
    @property
    def train_dataloader(self) -> data.DataLoader | None:
        """Returns a :obj:`torch.utils.data.DataLoader` object if :obj:`train`
        exists. Otherwise, returns ``None``.
        """
        if self.train:
            self.classlabels = getattr(self.train, "classlabels", self.classlabels)
            return data.DataLoader(
                dataset     = self.train,
                batch_size  = self.batch_size,
                shuffle     = self.shuffle,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = False,
                collate_fn  = getattr(self.train, "collate_fn", self.collate_fn),
                # prefetch_factor  = 4,
                persistent_workers = True,
            )
        return None
    
    @property
    def val_dataloader(self) -> data.DataLoader | None:
        """Returns a :obj:`torch.utils.data.DataLoader` object if :obj:`val`
        exists. Otherwise, returns ``None``.
        """
        if self.val:
            # self.classlabels = getattr(self.val, "classlabels", self.classlabels)
            return data.DataLoader(
                dataset     = self.val,
                batch_size  = self.batch_size,
                shuffle     = False,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = False,
                collate_fn  = getattr(self.val, "collate_fn", self.collate_fn),
                # prefetch_factor  = 4,
                persistent_workers = True,
            )
        return None
    
    @property
    def test_dataloader(self) -> data.DataLoader | None:
        """Returns a :obj:`torch.utils.data.DataLoader` object if :obj:`test`
        exists. Otherwise, returns ``None``.
        """
        if self.test:
            # self.classlabels = getattr(self.test, "classlabels", self.classlabels)
            return data.DataLoader(
                dataset     = self.test,
                batch_size  = 1,  # self.batch_size,
                shuffle     = False,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = False,
                collate_fn  = getattr(self.test, "collate_fn", self.collate_fn),
                # prefetch_factor    = 4,
                persistent_workers = True,
            )
        return None
    
    @property
    def predict_dataloader(self):
        """Returns a :obj:`torch.utils.data.DataLoader` object if :obj:`predict`
        exists. Otherwise, returns ``None``.
        """
        if self.predict:
            return data.DataLoader(
                dataset     = self.predict,
                batch_size  = self.batch_size,
                shuffle     = False,
                num_workers = self.num_workers,
                pin_memory  = True,
                drop_last   = True,
                collate_fn  = self.collate_fn,
                # prefetch_factor  = 4,
                persistent_workers = True,
            )
        return None
    
    @property
    def can_log(self) -> bool:
        if self.verbose:
            if self.trainer is None:
                return True
            elif self.trainer and self.trainer.global_rank == 0:
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
            - Define :obj:`collate_fn` for your custom dataset.

        Args:
            stage: The running stage. One of:
                - ``'train'``  : prepares :obj:`train` and :obj:`val`.
                - ``'test'``   : prepares :obj:`test`.
                - ``'predict'``: prepares :obj:`predict`.
                - ``None``     : prepares all.
                - Default: ``None``.
        """
        pass
    
    def get_classlabels(self):
        """Load all the class-labels of the dataset."""
        if isinstance(self.classlabels, base.ClassLabels):
            return
        elif self.train is not None:
            self.classlabels = getattr(self.train, "classlabels", None)
        elif self.val is not None:
            self.classlabels = getattr(self.val, "classlabels", None)
        elif self.test is not None:
            self.classlabels = getattr(self.test, "classlabels", None)
        elif self.predict is not None:
            self.classlabels = getattr(self.predict, "classlabels", None)
        else:
            rich.console.log("[yellow]No classlabels found.")
        
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
    
    def summarize(self):
        """Print a summary."""
        table = rich.table.Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Desc")
        table.add_row("1", "train",       f"{len(self.train) if self.train else None}")
        table.add_row("2", "val",         f"{len(self.val)   if self.val   else None}")
        table.add_row("3", "test",        f"{len(self.test)  if self.test  else None}")
        table.add_row("4", "classlabels", f"{self.classlabels.num_classes if self.classlabels else None}")
        table.add_row("5", "batch_size",  f"{self.batch_size}")
        table.add_row("6", "num_workers", f"{self.num_workers}")
        rich.console.log(table)
    
# endregion
