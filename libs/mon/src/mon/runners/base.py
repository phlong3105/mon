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
from torch import nn

from mon.core import Config, Path, Size
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
    def _benchmark(self, imgsz: Size | None = None):
        """Run the benchmark for the model.

        Args:
            imgsz (Size | None, optional): The input image size for benchmarking.
                Defaults to None, which means using the default size.
        """
        if self.model is None:
            raise RuntimeError("model has not been initialized, please call "
                               "'self._init_model()' before benchmarking.")

        imgsz = Size.from_any(imgsz or self.config.imgsz)
        self.model.benchmark(imgsz=imgsz, verbose=self.verbose)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
