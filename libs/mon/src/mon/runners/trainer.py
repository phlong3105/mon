#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides several metric evaluators.
"""

from __future__ import annotations

__all__ = [
    "Trainer",
]

from abc import ABC, abstractmethod

import numpy as np
import torch
from rich.progress import Progress
from torch import nn
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
        self.config = config

        # Extract commonly used attributes for convenience
        self.device = config.device
        self.benchmark = config.benchmark
        self.verbose = config.verbose

        # Allocate resources
        self.pbar = create_progress_bar()
        # We will initialize these attributes later to avoid a long
        # initialization time
        self.model: nn.Module | None = None
        self.optimizer: Optimizer | None = None
        self.scheduler: LRScheduler | None = None
        self.train_dataloader: DataLoader | None = None
        self.val_dataloader: DataLoader | None = None
        self.best: dict[str, float] = {
            "loss": float("inf"),
        }

    # --- Properties ---
    @abstractmethod
    def init_model(self):
        """Initialize ``self.model`` attribute."""
        pass

    @abstractmethod
    def init_optimizer(self):
        """Initialize ``self.optimizer`` and ``self.scheduler`` attributes."""
        pass

    def init_dataloaders(self):
        """Initialize ``self.train_dataloader`` and ``self.val_dataloader`` attributes."""
        train_dataloader = self.config.train_dataloader
        self.train_dataloader = DataLoader.from_config(train_dataloader)

        val_dataloader = self.config.val_dataloader
        if val_dataloader is not None:
            self.val_dataloader = DataLoader.from_config(val_dataloader)
        else:
            self.val_dataloader = None

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
        imgsz = Size.from_value(config.eval_imgsz)

        # 3. Define model
        self.init_model()
        if self.model is None:
            raise ValueError(f"'model' is not initialized.")

        # 4. Define optimizer & scheduler
        self.init_optimizer()
        if self.optimizer is None:
            raise ValueError(f"'optimizer' is not initialized.")

        # 5. Run benchmark
        if self.benchmark:
            benchmark(self.model, imgsz=imgsz)

        # 6. Define data
        self.init_dataloaders()
        if self.train_dataloader is None:
            raise ValueError(f"'train_dataloader' is not initialized.")

        # 7. Main loop
        config.output_dir.mkdir(exist_ok=True, parents=True)
        with create_progress_bar() as pbar:
            for epoch in pbar.track(
                sequence=range(epochs),
                total=epochs,
                description=f"[bright_yellow]Training"
            ):
                # 7.1. Train epoch
                train_outputs = self.train_epoch(epoch=epoch, pbar=pbar)
                if "loss" not in train_outputs:
                    raise ValueError(
                        f"Expected 'loss' from 'self.train_epoch()', "
                        f"but got {train_outputs.keys()}."
                    )

                # 7.2. Val epoch
                val_outputs = {}
                if self.val_dataloader is not None:
                    val_outputs |= self.val_epoch(epoch=epoch, pbar=pbar)

                # 7.3. Log
                self.log(epoch=epoch,train_outputs=train_outputs, val_outputs=val_outputs)

                # 7.4. Save
                self.save(epoch=epoch, train_outputs=train_outputs, val_outputs=val_outputs)

    # --- Training ---
    @abstractmethod
    def train_epoch(self, epoch: int, pbar: Progress) -> dict:
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
    def val_epoch(self, epoch: int, pbar: Progress) -> dict:
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
    @abstractmethod
    def log(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Log the training and validation results for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        pass

    @abstractmethod
    def save(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Save the model checkpoint for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        pass

    def save_best_weights(self, key: str, value: float, lower_is_better: bool = False):
        """Save the model checkpoint if the new value is better than the best
        value for the given key.

        Args:
            key (str): The key to compare in the best dictionary.
            value (float): The new value to compare against the best value.
            lower_is_better (bool, optional): Whether a lower value is better
                than a higher value. Defaults to False.
        """
        if key in self.best:
            best = self.best[key]
            if lower_is_better and value >= best:
                return
            elif not lower_is_better and value <= best:
                return

            self.best[key] = value
            torch.save(
                self.model.state_dict(),
                self.config.output_dir / f"best_{key}{K.WEIGHTS_EXT}"
            )
        else:
            self.best[key] = value

    def save_debug_image(self, epoch: int, outputs: dict[str, torch.Tensor]):
        """Save a debug image for visualization.

        Args:
            epoch (int): The current epoch number.
            outputs (dict[str, torch.Tensor]): A dictionary containing the
                outputs from the model.
        """
        config = self.config

        if config.save_debug:
            debug = []
            for k, v in outputs.items():
                image = to_image_array(torch.cat(list(v), dim=2).unsqueeze(0))
                image = draw_info(image, [f"{pascalize(k)}"])
                debug.append(image)
            debug = np.vstack(debug)
            save_path = config.output_dir / "debug" / f"debug_epoch_{epoch+1:03}.jpg"
            write_image(debug, save_path)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
