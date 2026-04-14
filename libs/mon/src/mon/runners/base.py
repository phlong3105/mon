#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Runners.

This module provides base runner classes.
"""

from __future__ import annotations

__all__ = [
    "Evaluator",
    "Runner",
]

from abc import ABC, abstractmethod

import torch
from box import Box
from torch import nn

from mon.core import (
    Config,
    DeviceLike,
    DictLike,
    Path,
    PathLike,
    Size,
    SizeLike,
    Split,
    SplitLike,
    sys_ctx,
)
from mon.dataset import (
    build_dataloader,
    build_dataset,
    DataLoader,
    Dataset,
    transform as T,
)

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
    def model(self) -> nn.Module:
        """Return the model object."""
        return self._model

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
            self.model.benchmark(imgsz=imgsz, verbose=self.verbose)


class Evaluator(ABC):
    """Base class for all evaluators."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: PathLike,
        target_dir: PathLike | None,
        result_file: PathLike | None,
        metrics: list[str],
        device: DeviceLike,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            input_dir (PathLike): The directory containing the input data.
            target_dir (PathLike | None): The directory containing the target
                data. If None, it will be inferred from the input directory.
            result_file (PathLike | None): The file to save the evaluation
                results. If None, results will not be saved.
            metrics (list[str]): The list of metrics to evaluate.
            device (DeviceLike): The device to use for evaluation.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        # Assign attributes
        self.verbose = verbose
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.result_file = result_file
        self.device = device

        # Allocate resources
        self._metrics: dict[str, dict] = []
        self._results = {}

        self._init_metrics(metrics)

    @abstractmethod
    def _init_metrics(self, metrics: list[str]):
        """Initialize ``self._metrics`` attributes."""
        pass

    @abstractmethod
    def _init_dataloader(self) -> DataLoader:
        """Build a dataloader for the given dataset."""
        pass

    # --- Properties ---
    @property
    def input_dir(self) -> Path:
        """Return the input directory."""
        return self._input_dir

    @input_dir.setter
    def input_dir(self, input_dir: PathLike):
        """Set the input directory."""
        self._input_dir = Path(input_dir).normalize()

    @property
    def target_dir(self) -> Path:
        """Return the target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, target_dir: PathLike | None):
        """Set the target directory."""
        target_dir = Path(target_dir).normalize() if target_dir else None
        if target_dir:
            self._target_dir = target_dir
        else:
            self._target_dir = self.input_dir.replace_part("/image/", "/target/")

    @property
    def has_target(self) -> bool:
        """Check if the dataset has target data."""
        return self.target_dir is not None and self.target_dir.is_dir()

    @property
    def metrics(self) -> dict[str, dict]:
        """Return the dictionary of metrics to evaluate."""
        return self._metrics

    @property
    def device(self) -> torch.device:
        """Return the device to use."""
        return self._device

    @device.setter
    def device(self, device: DeviceLike):
        """Set the device to use."""
        self._device = sys_ctx.get_torch_device(device)

    @property
    def results(self) -> dict:
        """Return the dictionary of measured results."""
        return self._results

    # --- Measure ---
    @abstractmethod
    def measure(self):
        """Run the evaluation."""
        pass

    # --- Logging ---
    @abstractmethod
    def log_results(self):
        """Log the evaluation results."""
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
