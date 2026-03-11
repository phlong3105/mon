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
from typing import Any

import numpy as np
import torch
from rich.progress import Progress
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from mon.core import (
    Config,
    ConfigContext,
    create_progress_bar,
    is_scalar,
    K,
    log,
    pascalize,
    Path,
    RunMode,
    Split,
    sys_ctx,
    TensorOrArray,
)
from mon.dataset import build_dataloader, DataLoader
from mon.ops import draw_info, to_image_array, write_image
from .base import Runner

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class Trainer(Runner, ABC):
    """Base class for all trainers."""

    # --- Lifecycle & Initialization ---
    def __init__(self, config: Config):
        """Initialize a new instance.

        Args:
            config (Config): The configuration object containing all necessary
                parameters for training.
        """
        super().__init__(config=config)
        # Allocate resources
        # We will initialize these attributes later to avoid a long
        # initialization time
        self._optimizer: Optimizer | None = None
        self._scheduler: LRScheduler | None = None
        self._train_dataloader: DataLoader | None = None
        self._val_dataloader: DataLoader | None = None
        self._tb_logger: SummaryWriter | None = None
        self._best: dict[str, float] = {
            "loss": float("inf"),
        }

    @abstractmethod
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        pass

    def _init_train_dataloader(self):
        """Initialize ``self._train_dataloader`` attribute."""
        config = self.config
        dataloader = self.config.train_dataloader

        self._train_dataloader = build_dataloader(
            src=dataloader,
            dataset_dir=config.data_dir,
            split=Split.TRAIN,
        )[1]

    def _init_val_dataloader(self):
        """Initialize ``self._val_dataloader`` attribute."""
        config = self.config
        dataloader = self.config.val_dataloader

        self._val_dataloader = build_dataloader(
            src=dataloader,
            dataset_dir=config.data_dir,
            split=Split.VAL,
        )[1]

    def _init_loggers(self):
        """Initialize external loggers for tracking training progress and metrics.
        """
        config = self.config

        # 1. Initialize Tensorboard logger
        if config.tensorboard_logger:
            log_dir = self.config.output_dir / "logs"
            log_dir.mkdir(exist_ok=True, parents=True)
            self._tb_logger = SummaryWriter(log_dir=str(log_dir))

        # 2. Initialize other loggers here

    # --- Properties ---
    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer object."""
        return self._optimizer

    @property
    def scheduler(self) -> LRScheduler | None:
        """Return the scheduler object."""
        return self._scheduler

    @property
    def lr(self) -> float:
        """Return the current learning rate from the optimizer."""
        if self.optimizer is None:
            return 0.0
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]
        return 0.0

    @property
    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return self._train_dataloader

    @property
    def val_dataloader(self) -> DataLoader | None:
        """Return the validation dataloader."""
        return self._val_dataloader

    @property
    def best(self) -> dict[str, float]:
        """Return the best dictionary."""
        return self._best

    # --- Creation ---
    @classmethod
    def from_cli(cls, prompt: bool = False, *args, **kwargs) -> "Trainer":
        """Create an instance of Trainer from command-line arguments.

        Args:
            prompt (bool, optional): Whether to prompt the user for input if
                necessary. Defaults to False.
        """
        config_ctx = ConfigContext.from_cli(*args, **kwargs)
        config = config_ctx.config_for(RunMode.TRAIN, prompt=prompt)
        return cls(config)

    # --- Control ---
    def train(self):
        """Train the model."""
        config = self.config

        # 1. Summarize the current run
        if config.verbose:
            config.log_summary()

        # 2. Setup environment
        config.output_dir.mkdir(exist_ok=True, parents=True)
        config.config_file.copy_to(config.output_dir / config.config_file.name)

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
        self._init_train_dataloader()
        self._init_val_dataloader()
        if self.train_dataloader is None:
            raise RuntimeError(f"'train_dataloader' is not initialized.")

        # 6. Define loggers
        self._init_loggers()

        # 7. Run benchmark
        self.benchmark()

        # 8. Main loop
        with create_progress_bar() as pbar:
            for epoch in pbar.track(
                sequence=range(epochs),
                total=epochs,
                description=f"[bright_yellow]Training"
            ):
                # 8.1. Train epoch
                self.model.train()
                train_outputs = self._train_epoch(epoch=epoch, pbar=pbar)
                if "loss" not in train_outputs:
                    raise ValueError(
                        f"Expected 'loss' from 'self._train_epoch()', "
                        f"but got {train_outputs.keys()}."
                    )

                # 8.2. Val epoch
                val_outputs = {}
                if self.val_dataloader is not None:
                    self.model.eval()
                    val_outputs = self._val_epoch(epoch=epoch, pbar=pbar)

                # 8.3. Scheduler Step
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        # If it's a Plateau scheduler, it needs a metric (usually Val Loss)
                        # Fallback to train loss if val loss isn't available
                        metric = val_outputs.get("loss", train_outputs.get("loss"))
                        self.scheduler.step(metric)
                    else:
                        # For all other standard schedulers (StepLR, CosineAnnealing, etc.)
                        self.scheduler.step()

                # 8.4. Log
                if self.verbose:
                    self._log(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

                # 8.5. Save
                if self.save:
                    self._save(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

                # 8.6. Save debug
                if self.save_debug:
                    self._save_debug(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

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

    # --- Logging ---
    def _log(
        self,
        epoch: int,
        train_outputs: dict[str, Any],
        val_outputs: dict[str, Any]
    ):
        """Log the training and validation results for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        # 1. Collect all scalars
        log_dict = {
            "train/lr": self.lr,
        }
        # Add scalars from train_outputs
        for k, v in train_outputs.items():
            if is_scalar(v):
                log_dict[f"train/{k}"] = v.item() if isinstance(v, Tensor) else v
        # Add scalars from val_outputs
        for k, v in val_outputs.items():
            if is_scalar(v):
                log_dict[f"val/{k}"] = v.item() if isinstance(v, Tensor) else v

        # 2. Log to console
        message = f"Epoch: {(epoch + 1):03}"
        for k, v in log_dict.items():
            message += f" | {k}: {v:>08.6f}"
        log(message)

        # 3. Log to external loggers
        if self._tb_logger:
            for k, v in log_dict.items():
                self._tb_logger.add_scalar(k.capitalize(), v, epoch)
            self._tb_logger.flush()

    # --- Output ---
    def _save(
        self,
        epoch: int,
        train_outputs: dict[str, Any],
        val_outputs: dict[str, Any]
    ):
        """Save the model checkpoint for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        config = self.config

        # Save last.pt
        save_dir = config.output_dir
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), save_dir / "last.pt")

        # Save best weights based on metrics
        self._save_best_weights("loss", train_outputs["loss"], lower_is_better=True)

        for k, v in val_outputs.items():
            if is_scalar(v):
                v = v.item() if isinstance(v, Tensor) else v
                self._save_best_weights(k, v)

    @abstractmethod
    def _save_debug(
        self,
        epoch: int,
        train_outputs: dict[str, Any],
        val_outputs: dict[str, Any]
    ):
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

        save_dir = self.config.output_dir
        save_dir.mkdir(exist_ok=True, parents=True)
        save_path = save_dir / f"best_{key}{K.WEIGHTS_EXT}"
        torch.save(self.model.state_dict(), save_path)

    def _save_image(
        self,
        epoch: int,
        outputs: dict[str, TensorOrArray],
        dirname: str = K.PRED_DIR,
        stem: str = "debug",
        column_first: bool = False,
        show_info: bool = True,
    ):
        """Save a debug image for visualization.

        Args:
            epoch (int): The current epoch number.
            outputs (dict): A dictionary containing the outputs from the model.
            stem (str, optional): The stem of the output file name.
                Defaults to "debug".
            dirname (str, optional): The directory name for the output file.
                Defaults to K.PRED_DIR.
            column_first (bool, optional): Whether to save the image in column
                format (i.e., stack outputs vertically). Defaults to True.
            show_info (bool, optional): Whether to draw the keys of the outputs
                as labels on the image. Defaults to True.
        """
        config = self.config

        # Create a debug image by concatenating the output tensors and
        # drawing the keys as labels
        images = []
        for k, v in outputs.items():
            if column_first:
                # Stack the tensors vertically (i.e., concatenate along the
                # width dimension)
                image = to_image_array(torch.cat(list(v), dim=1).unsqueeze(0))
            else:
                # Stack the tensors horizontally (i.e., concatenate along the
                # height dimension)
                image = to_image_array(torch.cat(list(v), dim=2).unsqueeze(0))

            if image.shape[2] == 1:
                # If the image is grayscale, repeat it to make it RGB
                image = np.repeat(image, 3, axis=2)
            if show_info:
                # Draw the key as a label on the image
                image = draw_info(image, [f"{pascalize(k)}"])
            images.append(image)

        if column_first:
            images = np.hstack(images)
        else:
            images = np.vstack(images)

        # Save the image
        save_path = config.output_dir / dirname / f"{stem}_epoch_{epoch+1:03}{K.IMAGE_EXT}"
        write_image(images, save_path)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
