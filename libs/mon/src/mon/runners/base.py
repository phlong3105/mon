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
from torch import nn

from mon.core import Config, Path, Size, sys_ctx
from mon.dataset import DataLoader
from mon.nn import Model

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
        # We will initialize these attributes later to avoid a long initialization time
        self._model: Model = None

    @abstractmethod
    def _setup(self):
        """Setup the runner ready for training or inference."""
        pass

    @abstractmethod
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        pass

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

    # --- Logging ---
    @abstractmethod
    def _log_summary(self):
        """Log a summary of the current run."""
        pass

    # --- Benchmark ---
    def benchmark(self, imgsz: Size | None = None):
        """Run the benchmark for the model.

        Args:
            imgsz (Size | None, optional): The input image size for benchmarking.
                Defaults to None, which means using the default size.
        """
        if self.model is None:
            raise RuntimeError(
                "Model has not been initialized. Please call '_init_model()' "
                "before benchmarking."
            )

        imgsz = Size.from_value(imgsz or self.config.imgsz)
        self.model.benchmark(imgsz=imgsz, verbose=self.verbose)


class Evaluator(ABC):
    """Base class for all evaluators."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        input_dir: Path,
        target_dir: Path | None,
        result_file: Path | None,
        metrics: list[str],
        device: torch.device | str | int,
        verbose: bool = True,
    ):
        """Initialize a new instance.

        Args:
            input_dir (Path): The directory containing the input data.
            target_dir (Path | None): The directory containing the target data.
                If None, it will be inferred from the input directory.
            result_file (Path | None): The file to save the evaluation results.
                If None, results will not be saved.
            metrics (list[str]): The list of metrics to evaluate.
            device (torch.device | str | int): The device to use for evaluation.
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

    # --- Properties ---
    @property
    def input_dir(self) -> Path:
        """Return the input directory."""
        return self._input_dir

    @input_dir.setter
    def input_dir(self, input_dir: Path):
        """Set the input directory."""
        self._input_dir = Path(input_dir).normalize()

    @property
    def target_dir(self) -> Path:
        """Return the target directory."""
        return self._target_dir

    @target_dir.setter
    def target_dir(self, target_dir: Path | None):
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
    def device(self, device: torch.device | str | int):
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

    @abstractmethod
    def _build_dataloader(self) -> DataLoader:
        """Build a dataloader for the given dataset."""
        pass

    # --- Logging ---
    @abstractmethod
    def log_summary(self):
        """Log a summary of the current run."""
        pass

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
