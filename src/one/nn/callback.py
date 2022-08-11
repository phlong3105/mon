#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import copy
import os
import time
from _weakref import proxy
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import *
from pytorch_lightning.callbacks.progress import rich_progress
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from pytorch_lightning.utilities.model_summary import get_human_readable_count
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from torch import Tensor

from one.constants import CALLBACKS
from one.core import console
from one.core import create_dirs
from one.core import error_console
from one.core import get_next_version
from one.core import GPUMemoryUsageColumn
from one.core import is_torch_saved_file
from one.core import Path_

if _RICH_AVAILABLE:
    from rich.table import Table


@CALLBACKS.register(name="checkpoint_callback")
class CheckpointCallback(Callback):
    """
    Checkpointing is a mechanism to store the state of a computation so that
    it can be retrieved at a later point in time and continued. Process of
    writing the computation's state is referred to as Checkpointing, the data
    written as the Checkpoint, and the continuation of the application as
    Restart or Recovery.

    Attributes:
        checkpoint_dir (Path_ | None): Directory to save the checkpoints.
            Checkpoints will be save to `../<checkpoint_dir>/weights/`.
        filename (str | None): Checkpoint filename. Can contain named
            formatting options to be autofilled. If `None`, it will be set to
            `epoch={epoch}.ckpt`.
        auto_insert_metric_name (bool): When True, the checkpoints filenames
            will contain the metric name.
        monitor (str | None): Quantity to monitor. If None, will monitor loss.
        mode (str): If `save_top_k != 0`, the decision to overwrite the current
            save file is made based on either the maximization or the
            minimization of the monitored quantity. One of: [`min`, `max`].
            For `acc`, this should be `max`, for `loss` this should be `min`,
            etc.
        verbose (bool): Verbosity mode. Defaults to False.
        save_weights_only (bool): If True, then only the model’s weights will
            be saved `model.save_weights(filepath)`, else the full model is
            saved `model.save(filepath)`.
        every_n_train_steps (int | None): Number of training steps between
            checkpoints. This value must be `None` or non-negative. This must
            be mutually exclusive with `train_time_interval` and `every_n_epochs`.
            - If `every_n_train_steps == None or every_n_train_steps == 0`, we
              skip saving during training.
            - To disable, set `every_n_train_steps = 0`.
        every_n_epochs (int | None): Number of epochs between checkpoints.
            This value must be `None` or non-negative.
            - If `every_n_epochs == None` or `every_n_epochs == 0`, we skip
              saving when the epoch ends.
            - To disable, set `every_n_epochs = 0`.
        train_time_interval (timedelta | None): Checkpoints are monitored at
            the specified time interval. For all practical purposes, this cannot
            be smaller than the amount of time it takes to process a single
            training batch. This is not guaranteed to execute at the exact time
            specified, but should be close. This must be mutually exclusive
            with `every_n_train_steps` and `every_n_epochs`.
        save_on_train_epoch_end (bool): Whether to run checkpointing at the end
            of the training epoch. If this is False, then the check runs at the
            end of the validation. Defaults to None and skip saving.
    """
    
    MODE_DICT = {
        "min":  torch.tensor(np.Inf),
        "max": -torch.tensor(np.Inf)
    }
    
    def __init__(
        self,
        model_dir              : str | None       = None,
        version                : int | str | None = None,
        filename               : str | None       = None,
        auto_insert_metric_name: bool             = True,
        monitor                : str | None       = None,
        mode                   : str              = "min",
        verbose                : bool             = False,
        save_weights_only      : bool             = False,
        every_n_train_steps    : int | None       = None,
        every_n_epochs         : str | None       = 1,
        train_time_interval    : timedelta | None = None,
        save_on_train_epoch_end: bool             = False,
    ):
        super().__init__()
        self.filename                = filename
        self.auto_insert_metric_name = auto_insert_metric_name
        self.monitor                 = monitor
        self.mode                    = mode
        self.verbose                 = verbose
        self.save_weights_only       = save_weights_only
        self.every_n_train_steps     = every_n_train_steps
        self.every_n_epochs          = every_n_epochs
        self.train_time_interval     = train_time_interval
        self.save_on_train_epoch_end = save_on_train_epoch_end
        self.last_global_step_saved  = -1
        self.last_time_checked       = None
        self.best_score              = self.MODE_DICT[self.mode]
        self.current_score           = self.MODE_DICT[self.mode]
        self.best_checkpoint_path    = ""
        self.best_checkpoint_relpath = ""
        self.last_checkpoint_path    = ""
        self.last_checkpoint_relpath = ""
        
        self.init_checkpoint_dir(model_dir, version)
        self.init_monitor_mode(monitor, mode)
        self.init_triggers(
            every_n_train_steps = every_n_train_steps,
            every_n_epochs      = every_n_epochs,
            train_time_interval = train_time_interval
        )
        self.validate_init_configuration()
        
    def init_checkpoint_dir(self, model_dir: str, version: int | str | None):
        """
        Initialize the checkpoint directory.
        
        Args:
            model_dir (str): Model's dir. Checkpoints will be save to
                `../<model_dir>/<version>/weights/`.
            version (int | str | None): Experiment version. If version is not
                specified the logger inspects the save directory for existing
                versions, then automatically assigns the next available version.
                If it is a string then it is used as the run-specific
                subdirectory name, otherwise `version_${version}` is used.
        """
        if version is None:
            version = get_next_version(root_dir=model_dir)
        if isinstance(version, int):
            version = f"version_{version}"
        version = version.lower()
        
        self.checkpoint_dir = os.path.join(model_dir, version, "weights")
        console.log(f"Checkpoint directory at: {self.checkpoint_dir}.")

    def init_monitor_mode(self, monitor: str | None, mode: str):
        """
        Initialize monitor and mode.
        
        Args:
            monitor (str | None): Quantity to monitor. Defaults to None which
                will monitor loss.
            mode (str): One of: [`min`, `max`].
                - If `save_top_k != 0`, the decision to overwrite the current
                  save file is made based on either the maximization or the
                  minimization of the monitored quantity.
                - For `val_acc`, this	should be `max`, for `val_loss` this
                  should be `min`, etc.
        """
        # Check monitor key
        if monitor is None:
            monitor = "loss"
        if "train" not in monitor and "val" not in monitor:
            if (
                self.every_n_train_steps
                and (self.every_n_train_steps >= 1 or self.save_on_train_epoch_end)
            ):
                monitor = f"train_{monitor}"
            else:
                monitor = f"val_{monitor}"
        self.monitor = monitor
        
        # Recheck the monitor mode. If it is `loss`, then mode should be change
        # to "min".
        if mode not in ["min", "max"]:
            raise ValueError(f"`mode` can be `min` or `max`. But got: {mode}.")
        self.mode = "min" if "loss" in self.monitor else mode
        
    def init_triggers(
        self,
        every_n_train_steps: int | None       = None,
        every_n_epochs     : int | None       = None,
        train_time_interval: timedelta | None = None,
    ):
        """
        Initialize save checkpoint trigger.
        
        Args:
            every_n_train_steps (int | None): Number of training steps between
                checkpoints. This value must be `None` or non-negative. This
                must be mutually exclusive with `train_time_interval` and
                `every_n_epochs`.
                - If `every_n_train_steps == None` or `every_n_train_steps == 0`,
                  we skip saving during training.
                - To disable, set `every_n_train_steps = 0`.
                 Defaults to None.
            every_n_epochs (int | None): Number of epochs between checkpoints.
                This value must be None or non-negative.
                - If `every_n_epochs == None` or `every_n_epochs == 0`, we skip
                  saving when the epoch ends.
                - To disable, set `every_n_epochs = 0`.
                Defaults to None.
            train_time_interval (timedelta | None): Checkpoints are monitored
                at the specified time interval. For all practical purposes, this
                cannot be smaller than the amount of time it takes to process a
                single training batch. This is not guaranteed to execute at the
                exact time specified, but should be close. This must be mutually
                exclusive with `every_n_train_steps` and `every_n_epochs`.
                Defaults to None.
        """
        # Default to running once after each validation epoch if neither
        # `every_n_train_steps` nor `every_n_epochs` is set
        if (
            every_n_train_steps is None
            and every_n_epochs is None
            and train_time_interval is None
        ):
            every_n_epochs      = 1
            every_n_train_steps = 0
            error_console.log(
                "Both `every_n_train_steps` and `every_n_epochs` are not set. "
                "Setting `every_n_epochs=1`"
            )
        else:
            every_n_epochs      = every_n_epochs      or 0
            every_n_train_steps = every_n_train_steps or 0
        
        self.train_time_interval = train_time_interval
        self.every_n_epochs      = every_n_epochs
        self.every_n_train_steps = every_n_train_steps
    
    def validate_init_configuration(self):
        """
        Validate all attributes' values during `__init__()`.
        """
        if self.every_n_train_steps < 0:
            raise ValueError(
                f"`every_n_train_steps` must >= 0. "
                f"But got: {self.every_n_train_steps}`."
            )
        if self.every_n_epochs < 0:
            raise ValueError(
                f"`every_n_epochs` must >= 0. "
                f"But got: {self.every_n_epochs}`."
            )
        
        every_n_train_steps_triggered = self.every_n_train_steps >= 1
        every_n_epochs_triggered      = self.every_n_epochs >= 1
        train_time_interval_triggered = self.train_time_interval is not None
        combined_trigger = (every_n_train_steps_triggered
                            + every_n_epochs_triggered
                            + train_time_interval_triggered)
        if combined_trigger > 1:
            raise ValueError(
                f"Combination of parameters "
                f"`every_n_train_steps={self.every_n_train_steps}`, "
                f"`every_n_epochs={self.every_n_epochs}`, and "
                f"`train_time_interval={self.train_time_interval}` "
                f"should be mutually exclusive."
            )
        
    def on_pretrain_routine_start(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ):
        """
        When pretrain routine starts we build the `checkpoint_dir` on the fly.

        Args:
            trainer (pl.Trainer): Trainer object.
            pl_module (pl.LightningModule): LightningModule object.
        """
        self._resolve_checkpoint_dir(trainer=trainer)
        if not trainer.fast_dev_run and trainer.should_rank_save_checkpoint:
            create_dirs(paths=[self.checkpoint_dir])
            
        # If the user runs validation multiple times per training epoch,
        # we try to save checkpoint after validation instead of on train
        # epoch end
        if self.save_on_train_epoch_end is None:
            self.save_on_train_epoch_end = trainer.val_check_interval == 1.0
    
    def on_train_start(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ):
        """
        Called when the train begins.

        Args:
            trainer (pl.Trainer): Trainer object.
            pl_module (pl.LightningModule): LightningModule object.
        """
        self.last_time_checked = time.monotonic()
    
    def on_train_batch_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs  : Tensor | dict[str, Any],
        batch    : Any,
        batch_idx: int,
        unused	 : int | None = 0,
    ):
        """
        Save checkpoint on train batch end if we meet the criteria for
        `every_n_train_steps`.

        Args:
            trainer (pl.Trainer): Trainer object.
            pl_module (pl.LightningModule): LightningModule object.
            outputs (Tensor | dict): Outputs from the model.
            batch (Any): Input batch.
            batch_idx (int): Batch's index.
        """
        if self._should_skip_saving_checkpoint(trainer=trainer):
            return  # Short circuit
        
        step       = trainer.global_step
        skip_batch = (self.every_n_train_steps < 1
                      or ((step + 1) % self.every_n_train_steps != 0))
        
        train_time_interval = self.train_time_interval
        skip_time           = True
        now                 = time.monotonic()
        if train_time_interval:
            prev_time_check = self.last_time_checked
            skip_time	    = (
                prev_time_check is None
                or (now - prev_time_check) < train_time_interval.total_seconds()
            )
            # In case we have time differences across ranks broadcast the
            # decision on whether to checkpoint from rank 0 to avoid possible
            # hangs
            skip_time = trainer.training_type_plugin.broadcast(skip_time)
        if skip_batch and skip_time:
            return
        if not skip_time:
            self.last_time_checked = now
        
        self._save_checkpoint(trainer=trainer)
    
    def on_train_epoch_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule",
        unused   : int | None = None
    ):
        """
        Save a checkpoint at the end of the training epoch.

        Args:
            trainer (pl.Trainer): Trainer object.
            pl_module (pl.LightningModule): LightningModule object.
            unused (int | None):
        """
        # As we advance one step at end of training, we use `global_step - 1`
        # to avoid saving duplicates
        trainer.fit_loop.global_step -= 1
        if (
            not self._should_skip_saving_checkpoint(trainer=trainer)
            and self.save_on_train_epoch_end
            and self.every_n_epochs > 0
            and (trainer.current_epoch + 1) % self.every_n_epochs == 0
        ):
            self._save_checkpoint(trainer=trainer)
        trainer.fit_loop.global_step += 1
    
    def on_validation_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ):
        """
        Save a checkpoint at the end of the validation stage.

        Args:
            trainer (pl.Trainer): Trainer object.
            pl_module (pl.LightningModule): LightningModule object.
        """
        if (
            self._should_skip_saving_checkpoint(trainer=trainer)
            or self.save_on_train_epoch_end
            or self.every_n_epochs < 1
            or (trainer.current_epoch + 1) % self.every_n_epochs != 0
        ):
            return
        self._save_checkpoint(trainer=trainer)
    
    def on_train_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ):
        """
        Save a checkpoint when training stops.

        This will only save a checkpoint if `save_last` is also enabled as the
        monitor metrics logged during training/validation steps or end of
        epochs are not guaranteed to be available at this stage.

        Args:
            trainer (pl.Trainer): Trainer object.
            pl_module (pl.LightningModule): LightningModule object.
        """
        if self._should_skip_saving_checkpoint(trainer=trainer):
            return
        if self.verbose:
            if trainer.is_global_zero:
                console.log("Saving latest checkpoint...")
        
        # As we advance one step at end of training, we use `global_step - 1`
        # to avoid saving duplicates
        trainer.train_loop.global_step -= 1
        self._save_checkpoint(trainer=trainer)
        trainer.train_loop.global_step += 1
    
    def on_save_checkpoint(
        self,
        trainer   : "pl.Trainer",
        pl_module : "pl.LightningModule",
        checkpoint: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Called when saving a model checkpoint, use to persist state.

        Args:
            trainer (pl.Trainer): Trainer object.
            pl_module (pl.LightningModule): LightningModule object.
            checkpoint (dict): Checkpoint dictionary that will be saved.

        Returns:
            Callback state.
        """
        checkpoint["checkpoint_dir"]          = self.checkpoint_dir
        checkpoint["epoch"]                   = trainer.current_epoch + 1
        checkpoint["global_step"]             = trainer.global_step + 1
        checkpoint["monitor"]                 = self.monitor
        checkpoint["best_score"]              = self.best_score
        checkpoint["current_score"]           = self.current_score
        
        # Change from absolute path -> relative path
        checkpoint["best_checkpoint_path"]    = self.best_checkpoint_path
        checkpoint["best_checkpoint_relpath"] = self.best_checkpoint_relpath
        checkpoint["last_checkpoint_path"]    = self.last_checkpoint_path
        checkpoint["last_checkpoint_relpath"] = self.last_checkpoint_relpath
        return checkpoint
    
    def on_load_checkpoint(
        self,
        trainer       : "pl.Trainer",
        pl_module     : "pl.LightningModule",
        callback_state: dict[str, Any]
    ):
        """
        Called when loading a model checkpoint, use to reload state.

        Args:
            trainer (pl.Trainer): Trainer object.
            pl_module (pl.LightningModule): LightningModule object.
            callback_state (dict): Callback state returned by
                `on_save_checkpoint`.
        """
        trainer.fit_loop.current_epoch = callback_state.get("epoch",         0)
        trainer.fit_loop.global_step   = callback_state.get("global_step",   0)
        self.best_score                = callback_state.get("best_score",    self.MODE_DICT[self.mode])
        self.current_score             = callback_state.get("current_score", self.MODE_DICT[self.mode])
        self.best_checkpoint_path      = callback_state.get("best_checkpoint_path",    "")
        self.best_checkpoint_relpath   = callback_state.get("best_checkpoint_relpath", "")
        self.last_checkpoint_path      = callback_state.get("last_checkpoint_path",    "")
        self.last_checkpoint_relpath   = callback_state.get("last_checkpoint_relpath", "")
        
        # if self.best_checkpoint_path == "" and self.best_checkpoint_relpath != "":
        #	self.best_checkpoint_path = os.path.join(one_dir, self.best_checkpoint_relpath)
        # elif self.best_checkpoint_relpath == "" and self.best_checkpoint_path != "":
        #	self.best_checkpoint_relpath = os.path.relpath(self.best_checkpoint_path, root_dir)
        
        # if self.last_checkpoint_path == "" and self.last_checkpoint_relpath != "":
        # 	self.last_checkpoint_path = os.path.join(one_dir, self.last_checkpoint_relpath)
        # elif self.last_checkpoint_relpath == "" and self.last_checkpoint_path != "":
        #	self.last_checkpoint_relpath = os.path.relpath(self.last_checkpoint_path, root_dir)
        
        if self.filename and (self.filename not in self.best_checkpoint_path):
            self.best_score           = self.MODE_DICT[self.mode]
            self.best_checkpoint_path = None
        if self.filename and (self.filename not in self.last_checkpoint_path):
            self.current_score        = self.MODE_DICT[self.mode]
            self.last_checkpoint_path = None
        
        console.log(f"Best: {self.best_score}. \n"
                    f"Checkpoint path: {self.best_checkpoint_path}.")
        console.log(f"Current: {self.current_score}. \n"
                    f"Checkpoint path: {self.last_checkpoint_path}")
    
    def manual_load_checkpoint(self, checkpoint: Path_):
        """
        Manually load checkpoints.
        
        Args:
            checkpoint (Path_): Checkpoint filepath.
        """
        checkpoint = str(checkpoint) \
            if isinstance(checkpoint, Path) else checkpoint
        
        if is_torch_saved_file(path=checkpoint):
            checkpoint                   = torch.load(checkpoint)
            self.best_score              = checkpoint.get("best_score",    self.MODE_DICT[self.mode])
            self.current_score           = checkpoint.get("current_score", self.MODE_DICT[self.mode])
            self.best_checkpoint_path    = checkpoint.get("best_checkpoint_path",    "")
            self.best_checkpoint_relpath = checkpoint.get("best_checkpoint_relpath", "")
            self.last_checkpoint_path    = checkpoint.get("last_checkpoint_path",    "")
            self.last_checkpoint_relpath = checkpoint.get("last_checkpoint_relpath", "")
        
        # if self.best_checkpoint_path == "" and self.best_checkpoint_relpath != "":
        # 	self.best_checkpoint_path = os.path.join(one_dir, self.best_checkpoint_relpath)
        # elif self.best_checkpoint_relpath == "" and self.best_checkpoint_path != "":
        #	self.best_checkpoint_relpath = os.path.relpath(self.best_checkpoint_path, root_dir)
        
        # if self.last_checkpoint_path == "" and self.last_checkpoint_relpath != "":
        # 	self.last_checkpoint_path = os.path.join(one_dir, self.last_checkpoint_relpath)
        # elif self.last_checkpoint_relpath == "" and self.last_checkpoint_path != "":
        #	self.last_checkpoint_relpath = os.path.relpath(self.last_checkpoint_path, root_dir)
        
        if self.filename and (self.filename not in self.best_checkpoint_path):
            self.best_score           = self.MODE_DICT[self.mode]
            self.best_checkpoint_path = None
        if self.filename and (self.filename not in self.last_checkpoint_path):
            self.current_score        = self.MODE_DICT[self.mode]
            self.last_checkpoint_path = None
            
        console.log(f"Best: {self.best_score}. \n"
                    f"Checkpoint path: {self.best_checkpoint_path}.")
        console.log(f"Current: {self.current_score}. \n"
                    f"Checkpoint path: {self.last_checkpoint_path}")
                
    def _save_checkpoint(self, trainer: "pl.Trainer"):
        """
        Performs the main logic around saving a checkpoint. This method runs
        on all ranks. It is the responsibility of `trainer.save_checkpoint`
        to correctly handle the behaviour in distributed training, i.e., saving
        only on rank 0 for data parallel use cases.
        
        Args:
            trainer (pl.Trainer): Trainer object.
        """
        self._validate_monitor_key(trainer=trainer)
        
        # Track epoch when ckpt was last checked
        self.last_global_step_saved = trainer.global_step
        
        # What can be monitored
        monitor_candidates = self._monitor_candidates(
            trainer = trainer,
            epoch   = trainer.current_epoch,
            step    = trainer.global_step
        )
        
        # Callback supports multiple simultaneous modes, here we call each mode
        # sequentially
        # Mode 1: Save best checkpoint
        self._save_best_checkpoint(trainer=trainer, monitor_candidates=monitor_candidates)
        # Mode 2: Save last checkpoint
        self._save_last_checkpoint(trainer=trainer, monitor_candidates=monitor_candidates)
        
        # NOTE: Notify loggers
        if trainer.is_global_zero and trainer.logger:
            trainer.logger.after_save_checkpoint(proxy(self))
    
    def _save_best_checkpoint(
        self,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, int | float | Tensor]
    ):
        """
        Save the best checkpoint.

        Args:
            trainer (pl.Trainer): Trainer object.
            monitor_candidates (dict): Dictionary of all monitored metrics.
        """
        # Get filepath and current score
        filepath = self._format_checkpoint_path(
            filename = self.filename,
            metrics  = monitor_candidates,
            postfix  = "best",
            epoch    = trainer.current_epoch
        )
        
        current = monitor_candidates.get(self.monitor, None)
        if (
            (current is None)
            or (isinstance(current, Tensor) and torch.isnan(current))
        ):
            current = float("inf" if self.mode == "min" else "-inf")
            current = torch.tensor(current)
        current = current.cpu()
        
        # Update best
        if self.best_checkpoint_path == "":
            ckpt_dict = { filepath: current }
        else:
            ckpt_dict = {
                self.best_checkpoint_path: self.best_score.cpu(),
                filepath				 : current
            }
        reverse     = True if self.mode == "max" else False
        sorted_dict = dict(sorted(ckpt_dict.items(), key=lambda x: x[1], reverse=reverse))
        best_path, best_score = list(sorted_dict.items())[0]
        
        # Save
        if (
            self.best_checkpoint_path != ""
            and self.best_checkpoint_path != best_path
            and trainer.should_rank_save_checkpoint
        ):
            self._del_model(trainer=trainer, filepath=self.best_checkpoint_path)
            
        if best_path != self.best_checkpoint_path:
            previous_score               = self.best_score
            self.best_score              = best_score
            self.best_checkpoint_path    = best_path
            # self.best_checkpoint_relpath = os.path.relpath(self.best_checkpoint_path, one_dir)
            
            self._save_model(trainer=trainer, filepath=best_path)
            
            if self.verbose:
                epoch = monitor_candidates.get("epoch")
                step  = monitor_candidates.get("step")
                if trainer.is_global_zero:
                    key  = self.monitor.replace("checkpoint/", "")
                    key  = key.replace("/", "_")
                    best_key     = "Best"
                    previous_key = "previous"
                    console.log(
                        f"[bold][Epoch {epoch:04d}, Step {step:08d}] "
                        f"[red]{best_key.ljust(7)}[/red] {key}: {self.best_score:10.6f}, "
                        f"{previous_key.ljust(8)}: {previous_score:10.6f}.\n"
                        f"Save checkpoint: {self.best_checkpoint_path}"
                    )
    
    def _save_last_checkpoint(
        self,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, int | float | Tensor]
    ):
        """
        Save the last checkpoint when training end.
        
        Args:
            trainer (pl.Trainer): Trainer object.
            monitor_candidates (dict): Dictionary of all monitored metrics.
        """
        # Get filepath
        filepath = self._format_checkpoint_path(
            filename = self.filename,
            metrics  = monitor_candidates,
            postfix  = "last",
            epoch    = trainer.current_epoch
        )
        
        # Save
        if self.last_checkpoint_path != "" and self.last_checkpoint_path != filepath:
            self._del_model(trainer=trainer, filepath=self.last_checkpoint_path)
        
        self.last_checkpoint_path    = filepath
        # self.last_checkpoint_relpath = os.path.relpath(self.last_checkpoint_path, one_dir)
        self._save_model(trainer=trainer, filepath=filepath)
        
        if self.verbose:
            current = monitor_candidates.get(self.monitor, None)
            if current is None or (isinstance(current, Tensor) and torch.isnan(current)):
                current = float("inf" if self.mode == "min" else "-inf")
                current = torch.tensor(current)
            current = current.cpu()
            
            if current != self.best_score:
                epoch = monitor_candidates.get("epoch")
                step  = monitor_candidates.get("step")
                if trainer.is_global_zero:
                    key         = self.monitor.replace("checkpoint/", "")
                    key         = key.replace("/", "_")
                    current_key = "Current"
                    best_key    = "best"
                    console.log(
                        f"[Epoch {epoch:04d}, Step {step:08d}] "
                        f"{current_key.ljust(7)} {key}: {current:10.6f}, "
                        f"{best_key.ljust(8)}: {self.best_score:10.6f}.\n"
                        f"Save checkpoint: {self.last_checkpoint_path}"
                    )
        
    def _save_model(self, trainer: "pl.Trainer", filepath: str):
        """
        Save the model's checkpoint.
        
        Args:
            trainer (pl.Trainer): Trainer object.
            filepath (str): Saved path.
        """
        # In debugging, track when we save checkpoints
        if hasattr(trainer, "dev_debugger"):
            trainer.dev_debugger.track_checkpointing_history(filepath=filepath)
        
        # Delegate the saving to the trainer
        trainer.save_checkpoint(
            filepath     = filepath,
            weights_only = self.save_weights_only
        )
    
    # noinspection PyMethodMayBeStatic
    def _del_model(self, trainer: "pl.Trainer", filepath: str):
        """
        Delete model's checkpoint.
        
        Args:
            trainer (pl.Trainer): Trainer object.
            filepath (str): Checkpoint path to delete.
        """
        if trainer.should_rank_save_checkpoint and os.path.exists(path=filepath):
            os.remove(path=filepath)
            # if self.verbose:
                # console.log(f"[Epoch {trainer.current_epoch}] Removed checkpoint: {filepath}")
    
    def _should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        """
        Check the trainer if saving checkpoint is possible.

        Args:
            trainer (pl.Trainer): Trainer object.

        Returns:
            Returns True will skip saving the current checkpoint. Else, False.
        """
        from pytorch_lightning.trainer.states import TrainerFn
        
        return (
            trainer.fast_dev_run                                   # Disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING               # Don't save anything during non-fit
            or trainer.sanity_checking                             # Don't save anything during sanity check
            or self.last_global_step_saved == trainer.global_step  # Already saved at the last step
        )
    
    # noinspection PyMethodMayBeStatic
    def _monitor_candidates(
        self,
        trainer: "pl.Trainer",
        epoch  : int,
        step   : int
    ) -> dict[str, int | float | Tensor]:
        """
        Get the monitored candidates.
        
        Args:
            trainer (pl.Trainer): Trainer object.
            epoch (int): Current training epoch.
            step (int): Current training step.

        Returns:
            Dictionary of all monitored metrics.
        """
        monitor_candidates = copy.deepcopy(trainer.callback_metrics)
        monitor_candidates.update(epoch=epoch, step=step)
        return monitor_candidates
    
    def _validate_monitor_key(self, trainer: "pl.Trainer"):
        """
        Run simple validate on the monitored key.
        
        Args:
            trainer (pl.Trainer): Trainer object.
        """
        metrics = trainer.callback_metrics
        
        # Validate metrics
        if self.monitor is not None and not self._is_valid_monitor_key(metrics):
            m = (
                f"ModelCheckpoint(monitor='{self.monitor}') "
                f"not found in the returned metrics:"
                f" {list(metrics.keys())}. "
                f"HINT: Did you call self.log('{self.monitor}', value) "
                f"in the LightningModule?"
            )
            if not trainer.fit_loop.epoch_loop.val_loop._has_run:
                console.log(m)
            else:
                raise ValueError(m)
    
    def _is_valid_monitor_key(
        self, metrics: dict[str, int | float | Tensor]
    ) -> bool:
        """
        Check if model's metrics has `monitor` key.
        
        Args:
            metrics (dict): Metrics defined in the model.
        """
        return self.monitor in metrics or len(metrics) == 0
        
    def _resolve_checkpoint_dir(self, trainer: "pl.Trainer"):
        """
        Determines model checkpoint save directory at runtime. Reference
        attributes from the trainer's logger to determine where to save
        checkpoints.
        Base path for saving weights is set in this priority:
            1. Checkpoint callback's path (if passed in)
            2. Default_root_dir from trainer if trainer has no logger
            3. Weights_save_path from trainer, if user provides it
            4. User provided weights_saved_path
        Base path gets extended with logger name and version (if these are
        available) and subdir "checkpoints".
        
        Args:
            trainer (pl.Trainer): Trainer object.
        """
        if self.checkpoint_dir is not None:
            return  # Short circuit
        
        if trainer.logger is not None:
            if trainer.weights_save_path != trainer.default_root_dir:
                # Fuser has changed `weights_save_path`, it overrides anything
                checkpoint_dir = trainer.weights_save_path
            else:
                checkpoint_dir = (trainer.logger.save_dir or trainer.default_root_dir)
        else:
            checkpoint_dir = trainer.weights_save_path
        
        checkpoint_dir      = trainer.training_type_plugin.broadcast(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir

    def _format_checkpoint_path(
        self,
        filename: str | None,
        metrics : dict[str, int | float | Tensor],
        postfix : str | None = None,
        epoch   : int | None = None
    ) -> str:
        """
        Format and return the checkpoint path to save.

        Args:
            filename (str | None): Checkpoint filename. Can contain named
                formatting options to be autofilled. If None, it will be set to
                `epoch={epoch}.ckpt`. Defaults to None.
            metrics (dict): Metrics defined in the model.
            postfix (str | None): Name postfix. One of: [`best`, `last`].
                Defaults to None.
            epoch (int | None): Current training epoch. Defaults to None.
        """
        if filename is None:
            filename = ""
        
        # Postfix
        if postfix:
            if filename == "":
                filename = f"{postfix}"
            else:
                filename = f"{filename}_{postfix}"
        
        # Include Epoch #
        if epoch:
            if filename == "":
                filename = f"[epoch={epoch:03d}]"
            else:
                filename = f"{filename}_[epoch={epoch:03d}]"
        
        # Include Metric
        if self.auto_insert_metric_name and metrics:
            key, value = self.monitor, metrics.get(self.monitor, None)
            key 	   = key.replace("checkpoint/", "")
            key 	   = key.replace("/", "_")
            if value is not None:
                filename = f"{filename}_[{key}={value:0.4f}]"
        
        filename += ".ckpt"
        return os.path.join(self.checkpoint_dir, filename)

    # noinspection PyMethodMayBeStatic
    def _file_exists(self, filepath: Path_, trainer: "pl.Trainer") -> bool:
        """
        Checks if a file exists on rank 0 and broadcasts the result to all
        other ranks, preventing the internal state to diverge between ranks.
        
        Args:
            filepath (Path_): Filepath.
            trainer (pl.Trainer): Trainer object.
        """
        exists = os.path.exists(path=filepath)
        return trainer.training_type_plugin.broadcast(exists)


@CALLBACKS.register(name="rich_model_summary")
class RichModelSummary(ModelSummary):
    """Generates a summary of all layers in a
    :class:`~pytorch_lightning.core.lightning.LightningModule` with `rich text
    formatting <https://github.com/willmcgugan/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichModelSummary

        trainer = Trainer(callbacks=RichModelSummary())

    You could also enable `RichModelSummary` using the
    :class:`~pytorch_lightning.callbacks.RichProgressBar`

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichProgressBar

        trainer = Trainer(callbacks=RichProgressBar())

    Args:
        max_depth (int): The maximum depth of layer nesting that the summary
            will include. A value of 0 turns the layer summary off.

    Raises:
        ModuleNotFoundError:
            If required `rich` package is not installed on the device.
    """

    def __init__(self, max_depth: int = 1):
        if not _RICH_AVAILABLE:
            raise ModuleNotFoundError(
                "`RichProgressBar` requires `rich` to be installed. "
                "Install it by running `pip install -U rich`."
            )
        super().__init__(max_depth)

    @staticmethod
    def summarize(
        summary_data        : list[tuple[str, list[str]]],
        total_parameters    : int,
        trainable_parameters: int,
        model_size          : float,
    ):
        table = Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")

        column_names = list(zip(*summary_data))[0]

        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        console.log(table)

        parameters = []
        for param in [trainable_parameters,
                      total_parameters - trainable_parameters,
                      total_parameters, model_size]:
            parameters.append("{:<{}}".format(
                get_human_readable_count(int(param)), 10)
            )
        
        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]}")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]}")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]}")
        grid.add_row(f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}")

        console.log(grid)


@CALLBACKS.register(name="rich_progress_bar")
class RichProgressBar(RichProgressBar):
    """
    Override `pytorch_lightning.callbacks.progress.rich_progress` to add some
    customizations.
    """
    
    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            self._console = console
            # self._console: Console = Console(**self._console_kwargs)
            self._console.clear_live()
            self._metric_component = rich_progress.MetricsTextColumn(
                trainer = trainer,
                style   = self.theme.metrics
            )
            self.progress = rich_progress.CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh = False,
                disable      = self.is_disabled,
                console      = self._console,
            )
            self.progress.start()
            # Progress has started
            self._progress_stopped = False
            
    def configure_columns(self, trainer) -> list:
        return [
            TextColumn(
                console.get_datetime().strftime("[%x %H:%M:%S:%f]"),
                justify = "left",
                style   = "log.time"
            ),
            TextColumn("[progress.description][{task.description}]"),
            rich_progress.CustomBarColumn(
                complete_style = self.theme.progress_bar,
                finished_style = self.theme.progress_bar_finished,
                pulse_style    = self.theme.progress_bar_pulse,
            ),
            rich_progress.BatchesProcessedColumn(style="progress.download"),
            "•",
            GPUMemoryUsageColumn(),
            "•",
            rich_progress.ProcessingSpeedColumn(style="progress.data.speed"),
            "•",
            TimeRemainingColumn(),
            ">",
            TimeElapsedColumn(),
            SpinnerColumn(),
        ]


CALLBACKS.register(name="backbone_finetuning",             module=BackboneFinetuning)
CALLBACKS.register(name="device_stats_monitor",            module=DeviceStatsMonitor)
CALLBACKS.register(name="early_stopping",                  module=EarlyStopping)
CALLBACKS.register(name="gradient_accumulation_scheduler", module=GradientAccumulationScheduler)
CALLBACKS.register(name="learning_rate_monitor",           module=LearningRateMonitor)
CALLBACKS.register(name="model_checkpoint",                module=ModelCheckpoint)
CALLBACKS.register(name="model_pruning",                   module=ModelPruning)
CALLBACKS.register(name="model_summary",                   module=ModelSummary)
CALLBACKS.register(name="quantization_aware_training",     module=QuantizationAwareTraining)
CALLBACKS.register(name="stochastic_weight_averaging",     module=StochasticWeightAveraging)
