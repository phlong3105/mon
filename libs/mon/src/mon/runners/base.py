#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Runners.

This module provides base runner classes.
"""

from __future__ import annotations

__all__ = [
    "Runner",
]

from abc import ABC, abstractmethod

import torch
from box import Box
from torch import nn

from mon.core import (
    Config,
    DictLike,
    Path,
    PathLike,
    Size,
    SizeLike,
    Split,
    SplitLike,
)
from mon.dataset import (
    build_dataloader,
    build_dataset,
    DataLoader,
    Dataset,
    transform as T,
)
from mon.metrics import benchmark

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Runner(ABC):
    """Base class for all runners."""

    # --- Lifecycle & Initialization ---
    def __init__(self, config: Config):
        """Initialize a new instance.

        Args:
            config (Config): The configuration object containing all necessary
                parameters for running.
        """
        # Assign attributes
        self._config = config

        # Allocate resources
        # We will initialize these attributes later to avoid a long
        # initialization time
        self._model: nn.Module | None = None

    @abstractmethod
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        pass

    def _init_dataset(
        self,
        source: DictLike | PathLike,
        split: SplitLike = Split.TEST,
        transforms: T.Compose | None = None,
    ) -> tuple[str | None, Dataset | None]:
        """Initialize and return a dataset.

        Args:
            source (DictLike | PathLike): A dataloader configuration dictionary
                or a source path.
            split (SplitLike, optional): The data split to use.
                Defaults to Split.TEST.
            transforms (T.Compose, optional): The data transformations to apply.
                Defaults to None.

        Returns:
            tuple[str | None, Dataset | None]: A tuple containing the dataset
                name (or None if not applicable) and the initialized dataset
                instance (or None if initialization failed).
        """
        return build_dataset(
            src=source,
            dataset_dir=self.config.data_dir,
            split=split,
            transforms=transforms,
        )

    def _init_dataloader(
        self,
        source: DictLike | PathLike,
        split: SplitLike = Split.TEST,
        transforms: T.Compose | None = None,
    ) -> tuple[str | None, DataLoader | None]:
        """Initialize and return a dataloader.

        Args:
            source (DictLike | PathLike): A dataloader configuration dictionary
                or a source path.
            split (SplitLike, optional): The data split to use.
                Defaults to Split.TEST.
            transforms (T.Compose, optional): The data transformations to apply.
                Defaults to None.

        Returns:
            tuple[str | None, DataLoader | None]: A tuple containing the dataloader
                name (or None if not applicable) and the initialized dataloader
                instance (or None if initialization failed).
        """
        config = self.config

        if isinstance(source, (Box , dict)):
            dataloader = DataLoader.from_config(source)
            name = dataloader.name
        elif isinstance(source, (Path, str)):
            name, dataloader = build_dataloader(
                src=source,
                dataset_dir=config.data_dir,
                split=split,
                transforms=transforms,
            )
        else:
            name, dataloader = None, None
        return name, dataloader

    # --- Properties ---
    @property
    def config(self) -> Config:
        """Return the config object."""
        return self._config

    @property
    def device(self) -> torch.device:
        """Return the device to use."""
        return self.config.device

    @property
    def save(self) -> bool:
        """Return the save flag."""
        return self.config.save

    @property
    def save_debug(self) -> bool:
        """Return the save_debug flag."""
        return self.config.save_debug

    @property
    def verbose(self) -> bool:
        """Return the verbose flag."""
        return self.config.verbose

    @property
    def model(self) -> nn.Module:
        """Return the model object."""
        return self._model

    # --- Utilities ---
    def benchmark(self, imgsz: SizeLike | None = None):
        """Run the benchmark for the model.

        Args:
            imgsz (SizeLike, optional): The input image size for benchmarking.
                Defaults to None, which means using the default size.
        """
        config = self.config
        imgsz = Size.from_value(imgsz or config.eval_imgsz)

        if config.benchmark:
            benchmark(self.model, imgsz=imgsz)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
