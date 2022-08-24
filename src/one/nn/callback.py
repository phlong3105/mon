#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import copy
import re
from _weakref import proxy
from datetime import timedelta

import pytorch_lightning as pl
import yaml
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import *
from pytorch_lightning.callbacks.progress import rich_progress
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.imports import _RICH_AVAILABLE
from pytorch_lightning.utilities.logger import _name
from pytorch_lightning.utilities.logger import _version
from pytorch_lightning.utilities.model_summary import get_human_readable_count
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

from one.constants import *
from one.core import *

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


# noinspection PyMethodMayBeStatic,PyProtectedMember
@CALLBACKS.register(name="model_checkpoint")
class ModelCheckpoint(Checkpoint):
    r"""
    Save the model periodically by monitoring a quantity. Every metric logged
    with :meth:`~pytorch_lightning.core.module.log` or
    :meth:`~pytorch_lightning.core.module.log_dict` in LightningModule is a
    candidate for the monitor key.

    After training finishes, use :attr:`best_model_path` to retrieve the path
    to the best checkpoint file and :attr:`best_model_score` to retrieve its
    score.

    Args:
        root (Path_): Root directory to save checkpoint files. By default, root
            is RUNS_DIR and will be set at runtime to the location specified by
            :class:`~pytorch_lightning.trainer.trainer.Trainer`'s
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.default_root_dir`
            or :paramref:`~pytorch_lightning.trainer.trainer.Trainer.weights_save_path`
            arguments and if the Trainer uses a logger, the path will also
            contain logger name and version.
        filename (str | None): Checkpoint filename. Can contain named
            formatting options to be autofilled. Defaults to None and will be
            set to {epoch}-{step}.
        monitor (str | None): Quantity to monitor. Defaults to None which saves
            a checkpoint only for the last epoch.
        save_last (bool): When True, saves an exact copy of the checkpoint to a
            file `last.ckpt` whenever a checkpoint file gets saved. This allows
            accessing the latest checkpoint in a deterministic manner.
            Defaults to None.
        save_top_k (int):
            - If `save_top_k == k`, the best k models according to the quantity
              monitored will be saved.
            - If `save_top_k == 0`, no models are saved.
            - If `save_top_k == -1`, all models are saved. Please note that the
              monitors are checked every `every_n_epochs` epochs.
            - If `save_top_k >= 2` and the callback is called multiple times
              inside an epoch, the name of the saved file will be appended with
              a version count starting with `v1`.
        save_on_train_epoch_end (bool): Whether to run checkpointing at the end
            of the training epoch. If this is False, then the check runs at the
            end of the validation. Defaults to False.
        save_weights_only (bool): If True, then only the model's weights will
            be saved. Otherwise, the optimizer states, lr-scheduler states,
            etc. are added in the checkpoint too. Defaults to False.
        mode (str): One of {min, max}. If `save_top_k != 0`, the decision to
            overwrite the current save file is made based on either the
            maximization or the minimization of the monitored quantity.
            For `val_acc`, this should be `max`, for `val_loss` this should be
            `min`, etc.
        every_n_train_steps (int | None): Number of training steps between
            checkpoints.
            - If `every_n_train_steps == None` or `every_n_train_steps == 0`,
              we skip saving during training.
            - To disable, set `every_n_train_steps = 0`. This value must be
              None or non-negative.
            - This must be mutually exclusive with `train_time_interval` and
              `every_n_epochs`.
        train_time_interval (timedelta | None): Checkpoints are monitored at
            the specified time interval. For all practical purposes, this
            cannot be smaller than the amount of time it takes to process a
            single training batch. This is not guaranteed to execute at the
            exact time specified, but should be close. This must be mutually
            exclusive with `every_n_train_steps` and `every_n_epochs`.
        every_n_epochs (int | None): Number of epochs between checkpoints.
            This value must be None or non-negative.
            - To disable saving top-k checkpoints, set `every_n_epochs = 0`.
            - This argument does not impact the saving of `save_last=True`
              checkpoints.
            - If all of `every_n_epochs`, `every_n_train_steps` and
              `train_time_interval` are None, we save a checkpoint at the end
              of every epoch (equivalent to `every_n_epochs = 1`).
            - If `every_n_epochs == None` and either
              `every_n_train_steps != None` or `train_time_interval != None`,
              saving at the end of each epoch is disabled (equivalent to
              `every_n_epochs = 0`).
            - This must be mutually exclusive with `every_n_train_steps` and
              `train_time_interval`.
            - Setting both `ModelCheckpoint(..., every_n_epochs=V, save_on_train_epoch_end=False)`
              and `Trainer(max_epochs=N, check_val_every_n_epoch=M)` will only
              save checkpoints at epochs 0 < E <= N where both values for
              `every_n_epochs` and `check_val_every_n_epoch` evenly divide E.
        auto_insert_metric_name (bool):
            - When True, the checkpoints filenames will contain the metric name.
              For example, `filename='checkpoint_{epoch:02d}-{acc:02.0f}` with
              epoch `1` and acc `1.12` will resolve to `checkpoint_epoch=01-acc=01.ckpt`.
            - Is useful to set it to False when metric names contain `/` as this
              will result in extra folders. For example,
              `filename='epoch={epoch}-step={step}-val_acc={val/acc:.2f}', auto_insert_metric_name=False`
        verbose (bool): Verbosity mode. Defaults to False.
    """

    checkpoint_join_char = "-"
    checkpoint_name_last = "last"
    file_extension       = ".ckpt"
    starting_version     = 1
    
    mode_dict = {
        "min":  torch.tensor(np.Inf),
        "max": -torch.tensor(np.Inf)
    }
    
    def __init__(
        self,
        root                   : Path_            = RUNS_DIR,
        filename               : str   | None     = None,
        monitor                : str   | None     = None,
        save_last              : bool             = None,
        save_top_k             : int              = 1,
        save_on_train_epoch_end: bool             = False,
        save_weights_only      : bool             = False,
        mode                   : str              = "min",
        every_n_train_steps    : int | None       = None,
        train_time_interval    : timedelta | None = None,
        every_n_epochs         : int | None       = None,
        auto_insert_metric_name: bool             = True,
        verbose                : bool             = False,
    ):
        super().__init__()
        self.monitor                 = monitor
        self.save_last               = save_last
        self.save_top_k              = save_top_k
        self.save_on_train_epoch_end = save_on_train_epoch_end
        self.save_weights_only       = save_weights_only
        self.auto_insert_metric_name = auto_insert_metric_name
        self.verbose                 = verbose
        
        self.last_epoch_saved        = 0
        self.last_global_step_saved  = 0  # no need to save when no steps were taken
        self.last_time_checked: float  | None = None
        self.current_score    : Tensor | None = None
        self.best_model_score : Tensor | None = None
        self.best_k_models    : dict[str, Tensor] = {}
        self.kth_best_model_path = None
        self.best_model_path     = None
        self.last_model_path     = None
        
        if mode not in self.mode_dict:
            raise ValueError(
                f"`mode` must be be {', '.join(self.mode_dict.keys())}. "
                f"But got: {mode}"
            )
        self.mode      = mode
        self.kth_value = self.mode_dict[mode]
        
        self.fs = get_filesystem(root if root else "")
        if root and self.fs.protocol == "file":
            root = os.path.realpath(root)
        self.root     = Path(root)
        self.ckpt_dir = self.root / "weights"
        self.filename = filename
        
        self.init_triggers(
            every_n_train_steps = every_n_train_steps,
            every_n_epochs      = every_n_epochs,
            train_time_interval = train_time_interval
        )
        self.validate_init_configuration()

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor                 = self.monitor,
            last_epoch_saved        = self.last_epoch_saved,
            last_global_step_saved  = self.last_global_step_saved,
            mode                    = self.mode,
            every_n_train_steps     = self.every_n_train_steps,
            every_n_epochs          = self.every_n_epochs,
            train_time_interval     = self.train_time_interval,
            save_on_train_epoch_end = self.save_on_train_epoch_end,
        )
       
    def init_triggers(
        self,
        every_n_train_steps: int       | None,
        every_n_epochs     : int       | None,
        train_time_interval: timedelta | None,
    ):
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None \
            and every_n_epochs is None \
            and train_time_interval is None:
            every_n_epochs      = 1
            every_n_train_steps = 0
            console.log(
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
        assert_larger_or_equal_than(self.save_top_k, -1)
        assert_larger_or_equal_than(self.every_n_train_steps, 0)
        assert_larger_or_equal_than(self.every_n_epochs, 0)
        
        every_n_train_steps_triggered = self.every_n_train_steps >= 1
        every_n_epochs_triggered      = self.every_n_epochs >= 1
        train_time_interval_triggered = self.train_time_interval is not None
        if every_n_train_steps_triggered \
            + every_n_epochs_triggered \
            + train_time_interval_triggered > 1:
            raise error_console.log(
                f"Combination of parameters "
                f"`every_n_train_steps={self.every_n_train_steps}`, "
                f"`every_n_epochs={self.every_n_epochs}` and "
                f"`train_time_interval={self.train_time_interval}` "
                "should be mutually exclusive."
            )

        if self.monitor is None:
            # -1: save all epochs, 0: nothing is saved, 1: save last epoch
            if self.save_top_k not in (-1, 0, 1):
                raise error_console.log(
                    f"ModelCheckpoint(save_top_k={self.save_top_k}, monitor=None) "
                    f"is not a valid configuration. No quantity for `top_k` to "
                    f"track."
                )
            if self.save_top_k == -1 and self.save_last:
                console.log(
                    "ModelCheckpoint(save_last=True, save_top_k=-1, monitor=None)"
                    " will duplicate the last checkpoint saved."
                )
                
    def setup(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage    : str | None = None
    ):
        self.resolve_ckpt_dir(trainer)
        assert self.root is not None
        if trainer.is_global_zero and stage == "fit":
            self.warn_if_dir_not_empty(self.root)

        # Setting these attributes needs to happen as early as possible BEFORE
        # reloading callback states, because the attributes are part of the
        # state_key which needs to be fully defined before reloading.
        if self.save_on_train_epoch_end is None:
            # If the user runs validation multiple times per training epoch or
            # multiple training epochs without validation, then we run after
            # validation instead of on train epoch end
            self.save_on_train_epoch_end = \
                trainer.val_check_interval == 1.0 \
                and trainer.check_val_every_n_epoch == 1
    
    def resolve_ckpt_dir(self, trainer: "pl.Trainer"):
        """
        Determines model checkpoint save directory at runtime. Reference
        attributes from the trainer's logger to determine where to save
        checkpoints. The path for saving weights is set in this priority:

        1.  The ModelCheckpoint's root if passed in
        2.  The Trainer's weights_saved_path if passed in (deprecated)
        3.  The Logger's log_dir if the trainer has loggers
        4.  The Trainer's default_root_dir if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".
        """
        if self.root is not None:
            # Short circuit if dirpath was passed to ModelCheckpoint
            return

        # TODO: Remove weights_save_path logic here in v1.8
        if trainer._weights_save_path_internal != trainer.default_root_dir:
            # the user has changed weights_save_path
            ckpt_path = os.path.join(trainer._weights_save_path_internal, "checkpoints")
        elif trainer.loggers:
            if len(trainer.loggers) == 1:
                assert trainer.logger is not None
                save_dir = trainer.logger.save_dir or trainer.default_root_dir
            else:
                save_dir = trainer.default_root_dir

            name    = _name(trainer.loggers)
            version = _version(trainer.loggers)
            version = version if isinstance(version, str) else f"version_{version}"

            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        ckpt_path = trainer.strategy.broadcast(ckpt_path)
        self.root = Path(ckpt_path)
    
    def on_train_start(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ):
        self.last_time_checked = time.monotonic()

    def on_train_batch_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs  : STEP_OUTPUT,
        batch    : Any,
        batch_idx: int,
    ):
        """
        Save checkpoint on train batch end if we meet the criteria for
        `every_n_train_steps`
        """
        if self.should_skip_saving_checkpoint(trainer=trainer):
            return
        skip_batch = self.every_n_train_steps < 1 \
                     or (trainer.global_step % self.every_n_train_steps != 0)

        train_time_interval = self.train_time_interval
        skip_time           = True
        now                 = time.monotonic()
        if train_time_interval:
            prev_time_check = self.last_time_checked
            skip_time = \
                prev_time_check is None \
                or (now - prev_time_check) < train_time_interval.total_seconds()
            # In case we have time differences across ranks broadcast the
            # decision on whether to checkpoint from rank 0 to avoid possible
            # hangs
            skip_time = trainer.strategy.broadcast(skip_time)

        if skip_batch and skip_time:
            return
        if not skip_time:
            self.last_time_checked = now

        candidates = self.monitor_candidates(trainer=trainer)
        self.save_topk_checkpoint(trainer=trainer, monitor_candidates=candidates)
        self.save_last_checkpoint(trainer=trainer, monitor_candidates=candidates)

    def on_train_epoch_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ):
        """
        Save a checkpoint at the end of the training epoch.
        """
        if not self.should_skip_saving_checkpoint(trainer=trainer) \
            and self.save_on_train_epoch_end:
            candidates = self.monitor_candidates(trainer=trainer)
            if self.every_n_epochs >= 1 \
                and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
                self.save_topk_checkpoint(
                    trainer            = trainer,
                    monitor_candidates = candidates
                )
            self.save_last_checkpoint(
                trainer            = trainer,
                monitor_candidates = candidates
            )

    def on_validation_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ):
        """
        Save a checkpoint at the end of the validation stage.
        """
        if not self.should_skip_saving_checkpoint(trainer=trainer) \
            and not self.save_on_train_epoch_end:
            candidates = self.monitor_candidates(trainer=trainer)
            if self.every_n_epochs >= 1 \
                and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
                self.save_topk_checkpoint(
                    trainer            = trainer,
                    monitor_candidates = candidates
                )
            self.save_last_checkpoint(
                trainer            = trainer,
                monitor_candidates = candidates
            )
    
    def state_dict(self) -> dict[str, Any]:
        return {
            "root"                  : self.root,
            "ckpt_dir"              : self.ckpt_dir,
            "last_epoch_saved"      : self.last_epoch_saved,
            "last_global_step_saved": self.last_global_step_saved,
            "monitor"               : self.monitor,
            "best_model_score"      : self.best_model_score,
            "best_model_path"       : self.best_model_path,
            "current_score"         : self.current_score,
            "best_k_models"         : self.best_k_models,
            "kth_best_model_path"   : self.kth_best_model_path,
            "kth_value"             : self.kth_value,
            "last_model_path"       : self.last_model_path,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        root_from_ckpt = state_dict.get("root", self.root)

        if self.root == root_from_ckpt:
            self.best_model_score    = state_dict["best_model_score"]
            self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
            self.kth_value           = state_dict.get("kth_value",           self.kth_value)
            self.best_k_models       = state_dict.get("best_k_models",       self.best_k_models)
            self.last_model_path     = state_dict.get("last_model_path",     self.last_model_path)
            self.kth_best_model_path = Path(self.kth_best_model_path)
            self.last_model_path     = Path(self.last_model_path)
        else:
            error_console.log(
                f"The `root` has changed from {root_from_ckpt!r} to "
                f"{self.root!r}, therefore `best_model_score`, "
                f"`kth_best_model_path`, `kth_value`, `last_model_path` and"
                f" `best_k_models` won't be reloaded. Only `best_model_path` "
                f"will be reloaded."
            )

        self.best_model_path        = Path(state_dict["best_model_path"])
        self.last_epoch_saved       = state_dict.get("last_epoch_saved",       self.last_epoch_saved)
        self.last_global_step_saved = state_dict.get("last_global_step_saved", self.last_global_step_saved)
    
    def save_topk_checkpoint(
        self,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, Tensor]
    ):
        if self.save_top_k == 0:
            return

        # Validate metric
        if self.monitor is not None:
            if self.monitor not in monitor_candidates:
                m = (
                    f"`ModelCheckpoint(monitor={self.monitor!r})` could not "
                    f"find the monitored key in the returned "
                    f"metrics: {list(monitor_candidates)}. "
                    f"HINT: Did you call `log({self.monitor!r}, value)` in "
                    f"the `LightningModule`?"
                )
                if trainer.fit_loop.epoch_loop.val_loop._has_run:
                    raise RuntimeError(m)
                error_console.log(m)
            self.save_monitor_checkpoint(
                trainer            = trainer,
                monitor_candidates = monitor_candidates
            )
        else:
            self.save_none_monitor_checkpoint(
                trainer            = trainer,
                monitor_candidates = monitor_candidates
            )
    
    def save_last_checkpoint(
        self,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, Tensor]
    ):
        if not self.save_last:
            return

        filepath = self.format_checkpoint_name(
            metrics  = monitor_candidates,
            filename = self.checkpoint_name_last
        )
        version_cnt = self.starting_version
        while self.file_exists(filepath, trainer) \
            and filepath != self.last_model_path:
            filepath = self.format_checkpoint_name(
                metrics  = monitor_candidates,
                filename = self.checkpoint_name_last,
                ver      = version_cnt
            )
            version_cnt += 1

        # Set the last model path before saving because it will be part of the
        # state.
        previous             = self.last_model_path
        self.last_model_path = filepath
        self._save_checkpoint(trainer=trainer, filepath=filepath)
        if previous and previous != filepath:
            trainer.strategy.remove_checkpoint(str(previous))

    def save_monitor_checkpoint(
        self,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, Tensor]
    ):
        assert self.monitor
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self.update_best_and_save(
                current            = current,
                trainer            = trainer,
                monitor_candidates = monitor_candidates
            )
        elif self.verbose:
            epoch = monitor_candidates["epoch"]
            step  = monitor_candidates["step"]
            console.log(
                f"[Epoch {epoch:04d}, Step {step:08d}] "
                f"{self.monitor!r} was not in top {self.save_top_k}."
            )

    def save_none_monitor_checkpoint(
        self,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, Tensor]
    ):
        filepath = self.get_metric_interpolated_filepath_name(
            monitor_candidates = monitor_candidates,
            trainer            = trainer
        )
        # Set the best model path before saving because it will be part of the
        # state.
        previous             = self.best_model_path
        self.best_model_path = filepath
        self._save_checkpoint(trainer=trainer, filepath=filepath)
        if self.save_top_k == 1 and previous and previous != filepath:
            trainer.strategy.remove_checkpoint(previous)

    def update_best_and_save(
        self,
        current           : Tensor,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, Tensor]
    ):
        k = len(self.best_k_models) + 1 \
            if self.save_top_k == -1 \
            else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(str(del_filepath))

        # Do not save nan, replace with +/- inf
        if isinstance(current, Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

        filepath = self.get_metric_interpolated_filepath_name(
            monitor_candidates = monitor_candidates,
            trainer            = trainer,
            del_filepath       = del_filepath
        )

        # Save the current score
        self.current_score                = current
        self.best_k_models[str(filepath)] = current

        if len(self.best_k_models) == k:
            # Monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value           = self.best_k_models[self.kth_best_model_path]

        _op = min if self.mode == "min" else max
        self.best_model_path  = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]

        if self.verbose:
            epoch = monitor_candidates["epoch"]
            step  = monitor_candidates["step"]
            if trainer.is_global_zero:
                key          = self.monitor.replace("checkpoint/", "")
                key          = key.replace("/", "_")
                best_key     = "Best"
                previous_key = "previous"
                console.log(
                    f"[bold][Epoch {epoch:04d}, Step {step:08d}] "
                    f"{self.monitor!r} reached {current:10.6f}"
                    f" (best {self.best_model_score:10.6f}). "
                    f"Saving model to {str(filepath)!r} as top {k}"
                )
        self._save_checkpoint(trainer=trainer, filepath=filepath)

        if del_filepath is not None and filepath != del_filepath:
            trainer.strategy.remove_checkpoint(del_filepath)

    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: Path):
        trainer.save_checkpoint(
            filepath     = filepath,
            weights_only = self.save_weights_only
        )
        self.last_epoch_saved       = trainer.current_epoch
        self.last_global_step_saved = trainer.global_step

        # Notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    def should_skip_saving_checkpoint(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn
        return (
            bool(trainer.fast_dev_run)                             # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING               # don't save anything during non-fit
            or trainer.sanity_checking                             # don't save anything during sanity check
            or self.last_global_step_saved == trainer.global_step  # already saved at the last step
        )
    
    def monitor_candidates(self, trainer: "pl.Trainer") -> dict[str, Tensor]:
        candidates = deepcopy(trainer.callback_metrics)
        # Cast to int if necessary because `self.log("epoch", 123)` will
        # convert it to float. If it's not a tensor or does not exist we
        # overwrite it as it's likely an error
        epoch = candidates.get("epoch")
        step  = candidates.get("step")
        candidates["epoch"] = epoch.int() \
            if isinstance(epoch, Tensor) \
            else torch.tensor(trainer.current_epoch)
        candidates["step"] = step.int() \
            if isinstance(step, Tensor) \
            else torch.tensor(trainer.global_step)
        return candidates
    
    def check_monitor_top_k(
        self,
        trainer: "pl.Trainer",
        current: Tensor | None = None
    ) -> bool:
        if current is None:
            return False

        if self.save_top_k == -1:
            return True

        less_than_k_models = len(self.best_k_models) < self.save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": torch.lt, "max": torch.gt}[self.mode]
        should_update_best_and_save = monitor_op(current, self.best_k_models[str(self.kth_best_model_path)])

        # If using multiple devices, make sure all processes are unanimous on
        # the decision.
        should_update_best_and_save = trainer.strategy.reduce_boolean_decision(bool(should_update_best_and_save))

        return should_update_best_and_save
    
    def get_metric_interpolated_filepath_name(
        self,
        monitor_candidates: dict[str, Tensor],
        trainer           : "pl.Trainer",
        del_filepath      : str | None = None
    ) -> Path:
        filepath    = self.format_checkpoint_name(monitor_candidates)
        version_cnt = self.starting_version
        while self.file_exists(filepath, trainer) and filepath != del_filepath:
            filepath = self.format_checkpoint_name(
                metrics = monitor_candidates,
                ver     = version_cnt
            )
            version_cnt += 1
        return filepath
    
    def format_checkpoint_name(
        self,
        metrics : dict[str, Tensor],
        filename: str | None = None,
        ver     : int | None = None
    ) -> Path:
        """
        Generate a filename according to the defined template.
        """
        filename = filename or self.filename
        filename = self._format_checkpoint_name(
            filename                = filename,
            metrics                 = metrics,
            auto_insert_metric_name = self.auto_insert_metric_name
        )
        if ver is not None:
            filename = self.checkpoint_join_char.join((filename, f"v{ver}"))
        ckpt_name = f"{filename}{self.file_extension}"
        return (self.ckpt_dir / ckpt_name) if self.ckpt_dir else ckpt_name
    
    @classmethod
    def _format_checkpoint_name(
        cls,
        filename               : str | None,
        metrics                : dict[str, Tensor],
        prefix                 : str  = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            # Filename is not set, use default name
            filename = "{epoch}" + cls.checkpoint_join_char + "{step}"

        # Check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            for group in groups:
                name = group[1:]

                if auto_insert_metric_name:
                    filename = filename.replace(group, name + "={" + name)

                # Support for dots: https://stackoverflow.com/a/7934969
                filename = filename.replace(group, f"{{0[{name}]")

                if name not in metrics:
                    metrics[name] = torch.tensor(0)
            filename = filename.format(metrics)

        if prefix:
            filename = cls.checkpoint_join_char.join([prefix, filename])

        return filename
    
    def to_yaml(self, filepath: Path_ | None = None):
        """
        Saves the `best_k_models` dict containing the checkpoint paths with the
        corresponding scores to a YAML file.
        """
        best_k = {k: v.item() for k, v in self.best_k_models.items()}
        if filepath is None:
            assert self.ckpt_dir
            filepath = os.path.join(self.ckpt_dir, "best_k_models.yaml")
        with self.fs.open(filepath, "w") as fp:
            yaml.dump(best_k, fp)
    
    def file_exists(self, filepath: Path_, trainer: "pl.Trainer") -> bool:
        """
        Checks if a file exists on rank 0 and broadcasts the result to all
        other ranks, preventing the internal state to diverge between ranks.
        """
        exists = self.fs.exists(str(filepath))
        return trainer.strategy.broadcast(exists)
    
    def warn_if_dir_not_empty(self, dirpath: Path_):
        if self.save_top_k != 0 \
            and self.fs.isdir(dirpath) \
            and len(self.fs.ls(dirpath)) > 0:
            console.log(
                f"Checkpoint directory {dirpath} exists and is not empty."
            )


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
class RichProgressBar(callbacks.RichProgressBar):
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
        if torch.cuda.is_available():
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
        else:
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
CALLBACKS.register(name="model_pruning",                   module=ModelPruning)
CALLBACKS.register(name="model_summary",                   module=ModelSummary)
CALLBACKS.register(name="quantization_aware_training",     module=QuantizationAwareTraining)
CALLBACKS.register(name="stochastic_weight_averaging",     module=StochasticWeightAveraging)
