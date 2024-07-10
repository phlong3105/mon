#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements callbacks for logging training/testing progress to the
console.
"""

from __future__ import annotations

__all__ = [
    "LogTrainingProgress"
]

import collections
import math
import time
from copy import deepcopy
from datetime import timedelta
from timeit import default_timer as timer
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.utilities.types import STEP_OUTPUT

from mon import core
from mon.globals import CALLBACKS

console       = core.console
error_console = core.error_console


# noinspection PyMethodMayBeStatic
@CALLBACKS.register(name="log_training_progress")
class LogTrainingProgress(callbacks.Callback):
    """Logs training/testing progress to the console.
    
    Args:
        dirpath: Directory to save the log file.
        filename: Log filename. Default: ``"train_log.csv"``.
        every_n_epochs: Log every n epochs. This value must be ``None`` or
            non-negative. Default: ``1``.
        every_n_train_steps: Log every n training steps.
        train_time_interval: Log every n seconds.
        log_on_train_epoch_end: Log on train epoch end.
        verbose: Verbosity. Default: ``True``.
        
    See Also: :class:`lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`.
    """
    
    def __init__(
        self,
        dirpath               : core.Path,
        filename              : str              = "log.csv",
        every_n_epochs        : int       | None = 1,
        every_n_train_steps   : int       | None = None,
        train_time_interval   : timedelta | None = None,
        log_on_train_epoch_end: bool      | None = None,
        verbose               : bool             = True
    ):
        super().__init__()
        self._dirpath     = core.Path(dirpath) if dirpath is not None else None
        self._filename    = core.Path(filename).stem
        self._candidates  = collections.OrderedDict()
        self._start_epoch = 0
        self._start_time  = 0
        self._logger      = None
        self._verbose     = verbose
        
        self._train_time_interval    = None
        self._every_n_epochs         = None
        self._every_n_train_steps    = None
        self._log_on_train_epoch_end = log_on_train_epoch_end
        self._last_global_step_saved = 0
        self._last_time_checked      = None
        self._init_triggers(every_n_epochs, every_n_train_steps, train_time_interval)
    
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str):
        """Called when fit, validate, test, predict, or tune begins."""
        dirpath = self._dirpath or core.Path(trainer.default_root_dir)
        # if str(dirpath.stem) != "log":
        #     dirpath /= "log"
        dirpath = trainer.strategy.broadcast(dirpath)
        self._dirpath = core.Path(dirpath)
    
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Called when the train begins."""
        self._candidates  = self._init_candidates(trainer, pl_module)
        self._start_epoch = int(trainer.current_epoch)
        self._start_time  = timer()
        
        # Create log file
        log_file       = self._dirpath / f"{core.Path(self._filename).stem}.csv"
        log_file_exist = log_file.exists()
        self._dirpath.mkdir(parents=True, exist_ok=True)
        self._logger   = open(str(log_file), "a")
        
        if not log_file_exist:
            self._logger.write(f"{'Model'},{pl_module.name}\n")
            self._logger.write(f"{'Fullname'},{pl_module.fullname}\n")
            if hasattr(pl_module, "params"):
                self._logger.write(f"{'Parameters'},{pl_module.params}\n")
            headers = ",".join(list(self._candidates.keys()))
            self._logger.write(f"\n{headers}\n")
            self._logger.flush()
    
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Called when the train ends."""
        end_time      = timer()
        elapsed_epoch = int(trainer.current_epoch) - self._start_epoch
        elapsed_time  = end_time - self._start_time
        elapsed_hours = elapsed_time / 3600
        
        # Close log file
        self._logger.write(
            f"\nEpochs,{elapsed_epoch}"
            f"Seconds,{elapsed_time:.3f}"
            f"Hours,{elapsed_hours:.3f}\n"
        )
        self._logger.flush()
        self._logger.close()
        
        if self._verbose:
            if trainer.is_global_zero:
                console.log(
                    f"\n{elapsed_epoch} epochs completed "
                    f"in {elapsed_time :.3f} seconds "
                    f"({elapsed_hours:.3f} hours)\n"
                )
    
    def on_train_batch_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs  : STEP_OUTPUT,
        batch    : Any,
        batch_idx: int
    ):
        """Called when the train batch ends."""
        if self._should_skip_logging(trainer):
            return
        skip_batch = (
            self._every_n_train_steps < 1
            or (trainer.global_step % self._every_n_train_steps != 0)
        )
        
        train_time_interval = self._train_time_interval
        skip_time = True
        now       = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = prev_time_check is None or (now - prev_time_check) < train_time_interval.total_seconds()
            # In case we have time differences across ranks broadcast the decision
            # on whether to checkpoint from rank 0 to avoid possible hangs
            skip_time = trainer.strategy.broadcast(skip_time)
        
        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now
        
        if trainer.is_global_zero:
            monitor_candidates = self._get_monitor_candidates(trainer)
            candidates         = self._update_candidates(monitor_candidates)
            self._log(candidates)
    
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Called when the train epoch ends."""
        if not self._should_skip_logging(trainer) and self._should_log_on_train_epoch_end(trainer):
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if trainer.is_global_zero:
                    monitor_candidates = self._get_monitor_candidates(trainer)
                    candidates         = self._update_candidates(monitor_candidates)
                    self._log(candidates)
    
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        """Called when the validation loop ends."""
        if not self._should_skip_logging(trainer) and not self._should_log_on_train_epoch_end(trainer):
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if trainer.is_global_zero:
                    monitor_candidates = self._get_monitor_candidates(trainer)
                    candidates         = self._update_candidates(monitor_candidates)
                    self._log(candidates)
    
    def _init_triggers(
        self,
        every_n_epochs     : int       | None = None,
        every_n_train_steps: int       | None = None,
        train_time_interval: timedelta | None = None,
    ):
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            every_n_epochs      = 1
            every_n_train_steps = 0
            console.log("Both every_n_train_steps and every_n_epochs are not set. Setting every_n_epochs=1")
        else:
            every_n_epochs      = every_n_epochs      or 0
            every_n_train_steps = every_n_train_steps or 0
        
        self._train_time_interval = train_time_interval
        self._every_n_epochs      = every_n_epochs
        self._every_n_train_steps = every_n_train_steps
        
    def _init_candidates(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Header
        self._candidates  = collections.OrderedDict()
        self._candidates |= {"epoch": None}
        self._candidates |= {"step" : None}
        # Train metrics
        self._candidates |= {"train/loss": None}
        if pl_module.train_metrics is not None:
            for m in pl_module.train_metrics:
                self._candidates |= {f'train/{m.name}': None}
        # Val metrics
        self._candidates |= {"val/loss": None}
        if pl_module.val_metrics is not None:
            for m in pl_module.val_metrics:
                self._candidates |= {f'val/{m.name}': None}
        
        return self._candidates
    
    def _update_candidates(self, monitor_candidates: dict[str, torch.Tensor]) -> dict[str, Any]:
        candidates = deepcopy(self._candidates)
        for c, v in monitor_candidates.items():
            if c in candidates:
                candidates[c] = v
        candidates["step"] = monitor_candidates.get("global_step", monitor_candidates["step"])
        return candidates
    
    def _get_monitor_candidates(self, trainer: "pl.Trainer") -> dict[str, torch.Tensor]:
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # Cast to int if necessary because `self.log("epoch", 123)` will convert
        # it to float. if it is not a tensor or does not exist, we overwrite it
        # as it is likely an error.
        # epoch       = monitor_candidates.get("epoch")
        # step        = monitor_candidates.get("step")
        # global_step = monitor_candidates.get("global_step")
        # monitor_candidates["epoch"]       = epoch.int()       if isinstance(epoch,       torch.Tensor) else torch.tensor(trainer.current_epoch)
        # monitor_candidates["step"]        = step.int()        if isinstance(step,        torch.Tensor) else torch.tensor(trainer.global_step)
        # monitor_candidates["global_step"] = global_step.int() if isinstance(global_step, torch.Tensor) else torch.tensor(trainer.global_step)
        monitor_candidates["epoch"]       = torch.tensor(trainer.current_epoch)
        monitor_candidates["step"]        = torch.tensor(trainer.global_step)
        monitor_candidates["global_step"] = torch.tensor(trainer.global_step)
        return monitor_candidates
    
    def _should_skip_logging(self, trainer: "pl.Trainer") -> bool:
        from lightning.pytorch.trainer.states import TrainerFn
        
        return (
            bool(trainer.fast_dev_run)                              # disable logging with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING                # don't log anything during non-fit
            or trainer.sanity_checking                              # don't log anything during sanity check
            or self._last_global_step_saved == trainer.global_step  # already log at the last step
        )
    
    def _should_log_on_train_epoch_end(self, trainer: "pl.Trainer") -> bool:
        if self._log_on_train_epoch_end is not None:
            return self._log_on_train_epoch_end
        
        # If `check_val_every_n_epoch != 1`, we can't say when the validation dataloader will be loaded
        # so let's not enforce saving at every training epoch end
        if trainer.check_val_every_n_epoch != 1:
            return False
        
        # No validation means log on train epoch end
        num_val_batches = (sum(trainer.num_val_batches) if isinstance(trainer.num_val_batches, list) else trainer.num_val_batches)
        if num_val_batches == 0:
            return True
        
        # If the user runs validation multiple times per training epoch, then we run after validation
        # instead of on train epoch end
        return trainer.val_check_interval == 1.0
    
    def _log(self, candidates: dict[str, torch.Tensor]):
        # Logger
        row = f""
        for i, (k, v) in enumerate(candidates.items()):
            if i == len(candidates) - 1:
                row += f""  if v is None else f"{v}"
            else:
                row += f"," if v is None else f"{v},"
        self._logger.write(f"{row}\n")
        self._logger.flush()
        
        # Console
        if self._verbose:
            row  = f""
            row1 = f""
            row2 = f""
            for i, (c, v) in enumerate(candidates.items()):
                # New line every 6 columns
                if i > 0 and i % 6 == 0:
                    row  += f"{row1}\n{row2}\n"
                    row1  = f""
                    row2  = f""
                # Add to row
                row1 += f"{c:>12} "
                if v is None:
                    row2 += f"{'':>12} "
                elif math.isnan(v):
                    row2 += f"{'NaN':>12} "
                elif int(v) == v and int(v) != 0:
                    row2 += f"{int(v):>12d} "
                else:
                    row2 += f"{v:>12.6f} "
            # Final row
            if row1 != "" and row2 != "":
                row += f"{row1}\n{row2}\n"
            print()
            console.log(f"{row}")
