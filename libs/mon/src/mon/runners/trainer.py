#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes.
"""

from __future__ import annotations

__all__ = [
    "Trainer",
]

from abc import ABC, abstractmethod

import numpy as np
import torch
from rich.progress import Progress
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mon.core import (
    Config,
    ConfigContext,
    create_progress_bar,
    K,
    pascalize,
    Path,
    RunMode,
    Size,
    sys_ctx,
)
from mon.dataset import DataLoader
from mon.metrics import benchmark
from mon.ops import draw_info, to_image_array, write_image

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Trainer(ABC):
    """Base class for all trainers."""

    # --- Lifecycle & Initialization ---
    def __init__(self, config: Config):
        """Initialize a new instance.

        Args:
            config (Config): The configuration object containing all necessary
                parameters for training.
        """
        # Assign attributes
        self._config = config

        # Allocate resources
        # We will initialize these attributes later to avoid a long
        # initialization time
        self._model: nn.Module | None = None
        self._optimizer: Optimizer | None = None
        self._scheduler: LRScheduler | None = None
        self._train_dataloader: DataLoader | None = None
        self._val_dataloader: DataLoader | None = None
        self._best: dict[str, float] = {
            "loss": float("inf"),
        }

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
    def benchmark(self) -> bool:
        """Return the benchmark flag."""
        return self.config.benchmark

    @property
    def verbose(self) -> bool:
        """Return the verbose flag."""
        return self.config.verbose

    @property
    def model(self) -> nn.Module:
        """Return the model object."""
        return self._model

    @abstractmethod
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        pass

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer object."""
        return self._optimizer

    @property
    def scheduler(self) -> LRScheduler | None:
        """Return the scheduler object."""
        return self._scheduler

    @abstractmethod
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        pass

    @property
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return self._train_dataloader

    @property
    def val_dataloader(self) -> DataLoader | None:
        """Return the validation dataloader."""
        return self._val_dataloader

    def _init_dataloaders(self):
        """Initialize ``self._train_dataloader`` and ``self._val_dataloader``
        attributes.
        """
        train_dataloader = self.config.train_dataloader
        self._train_dataloader = DataLoader.from_config(train_dataloader)

        val_dataloader = self.config.val_dataloader
        if val_dataloader is not None:
            self._val_dataloader = DataLoader.from_config(val_dataloader)
        else:
            self._val_dataloader = None

    @property
    def best(self) -> dict[str, float]:
        """Return the best dictionary."""
        return self._best

    # --- Creation ---
    @classmethod
    def from_cli(cls, *args, **kwargs) -> "Trainer":
        """Create an instance of Trainer from command-line arguments."""
        config_ctx = ConfigContext.from_cli(*args, **kwargs)
        config = config_ctx.config_for(RunMode.TRAIN)
        return cls(config)

    # --- Control ---
    def train(self):
        """Train the model."""
        config = self.config

        # 1. Summarize the current run
        if config.verbose:
            config.log_summary()

        # 2. Setup environment
        sys_ctx.set_random_seed(config.seed)
        epochs = config.epochs

        # 3. Define model
        self._init_model()
        if self.model is None:
            raise RuntimeError(f"'model' is not initialized.")

        # 4. Define optimizer & scheduler
        self._init_optimizer()
        if self.optimizer is None:
            raise RuntimeError(f"'optimizer' is not initialized.")

        # 5. Define data
        self._init_dataloaders()
        if self.train_dataloader is None:
            raise RuntimeError(f"'train_dataloader' is not initialized.")

        # 6. Run benchmark
        self._benchmark()

        # 7. Main loop
        config.output_dir.mkdir(exist_ok=True, parents=True)
        with create_progress_bar() as pbar:
            for epoch in pbar.track(
                sequence=range(epochs),
                total=epochs,
                description=f"[bright_yellow]Training"
            ):
                # 7.1. Train epoch
                train_outputs = self._train_epoch(epoch=epoch, pbar=pbar)
                if "loss" not in train_outputs:
                    raise ValueError(
                        f"Expected 'loss' from 'self._train_epoch()', "
                        f"but got {train_outputs.keys()}."
                    )

                # 7.2. Val epoch
                val_outputs = {}
                if self.val_dataloader is not None:
                    val_outputs = self._val_epoch(epoch=epoch, pbar=pbar)

                # 7.3. Log
                self._log(
                    epoch=epoch,
                    train_outputs=train_outputs,
                    val_outputs=val_outputs,
                )

                # 7.4. Save
                self._save(
                    epoch=epoch,
                    train_outputs=train_outputs,
                    val_outputs=val_outputs,
                )

                # 7.5. Save debug
                if config.save_debug:
                    self._save_debug(
                        epoch=epoch,
                        train_outputs=train_outputs,
                        val_outputs=val_outputs,
                    )

    # --- Training ---
    @abstractmethod
    def _train_epoch(self, epoch: int, pbar: Progress) -> dict:
        """Train an epoch.

        Args:
            epoch (int): The current epoch number.
            pbar (Progress): The progress bar object.

        Returns:
            dict: A dictionary containing the training loss and other results
                for the epoch.
        """
        pass

    # --- Validation ---
    @abstractmethod
    def _val_epoch(self, epoch: int, pbar: Progress) -> dict:
        """Validate an epoch.

        Args:
            epoch (int): The current epoch number.
            pbar (Progress): The progress bar object.

        Returns:
            dict: A dictionary containing the validation metrics and other
                results for the epoch.
        """
        pass

    # --- Utilities ---
    def _benchmark(self):
        """Run the benchmark for the model."""
        config = self.config
        imgsz = Size.from_value(config.eval_imgsz)

        if config.benchmark:
            benchmark(self.model, imgsz=imgsz)

    @abstractmethod
    def _log(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Log the training and validation results for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        pass

    @abstractmethod
    def _save(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Save the model checkpoint for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        pass

    @abstractmethod
    def _save_debug(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Save debugging results for visualization.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        pass

    def _save_best_weights(
        self,
        key: str,
        value: float,
        lower_is_better: bool = False
    ):
        """Save the model checkpoint if the new value is better than the best
        value for the given key.

        Args:
            key (str): The key to compare in the best dictionary.
            value (float): The new value to compare against the best value.
            lower_is_better (bool, optional): Whether a lower value is better
                than a higher value. Defaults to False.
        """
        # If the key is not in the best dictionary, save the new value
        if key not in self.best:
            self.best[key] = value
            return

        # If the new value is not better than the best value, skip saving
        if lower_is_better and value >= self.best[key]:
            return
        elif not lower_is_better and value <= self.best[key]:
            return

        # Otherwise, update the best value and save the model checkpoint
        self.best[key] = value
        torch.save(
            self.model.state_dict(),
            self.config.output_dir / f"best_{key}{K.WEIGHTS_EXT}"
        )

    def _save_image(
        self,
        epoch: int,
        outputs: dict[str, Tensor],
        stem: str = "debug",
        show_info: bool = True
    ):
        """Save a debug image for visualization.

        Args:
            epoch (int): The current epoch number.
            outputs (dict): A dictionary containing the outputs from the model.
            stem (str, optional): The stem of the output file name.
                Defaults to "debug".
            show_info (bool, optional): Whether to draw the keys of the outputs
                as labels on the image. Defaults to True.
        """
        config = self.config

        # Create a debug image by concatenating the output tensors and
        # drawing the keys as labels
        images = []
        for k, v in outputs.items():
            image = to_image_array(torch.cat(list(v), dim=2).unsqueeze(0))
            if show_info:
                image = draw_info(image, [f"{pascalize(k)}"])
            images.append(image)
        images = np.vstack(images)

        # Save the image
        save_path = config.output_dir / "debug" / f"{stem}_epoch_{epoch+1:03}{K.IMAGE_EXT}"
        write_image(images, save_path)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
