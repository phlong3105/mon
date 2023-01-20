#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the checkpoint callback to save model automatically
during training.
"""

from __future__ import annotations

__all__ = [
    "ModelCheckpoint",
]

import _weakref
import collections
import copy
import os
import re
import time
from datetime import datetime, timedelta
from timeit import default_timer as timer
from typing import Any

import lightning
import numpy as np
import torch
from lightning.pytorch.utilities import cloud_io, logger

from mon.coreml import constant
from mon.coreml.callback import base
from mon.coreml.typing import PathType
from mon.foundation import console, error_console, pathlib, RUNS_DIR, yaml


# region Model Checkpoint Callback

def _name(loggers: list[Any], separator: str = "_") -> str:
    if len(loggers) == 1:
        return loggers[0].name
    else:
        # Concatenate names together, removing duplicates and preserving order
        return separator.join(dict.fromkeys(str(logger.name) for logger in loggers))
    

@constant.CALLBACK.register(name="model_checkpoint")
class ModelCheckpoint(base.Checkpoint):
    """Save the model periodically by monitoring a quantity. Every metric
    logged with :meth:`lightning.pytorch.core.module.log` or
    :meth:`lightning.pytorch.core.module.log_dict` in
    :class:`lightning.LightningModule` is a candidate for the monitor key.

    After training finishes, use :param:`best_model_path` to retrieve the path to
    the best checkpoint file and :param:`best_model_score` to retrieve its score.

    Args:
        root: Root directory to save checkpoint files. By default, root is
            :data:`~mon.foundation.constant.RUNS_DIR` and will be set at runtime
            to the location specified by
            :class:`~pytorch_lightning.trainer.trainer.Trainer`'s
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.default_root_dir`
            or
            :paramref:`~pytorch_lightning.trainer.trainer.Trainer.weights_save_path`
            arguments and if the Trainer uses a logger, the path will also
            contain logger name and version.
        filename: Checkpoint filename. Can contain named formatting options to
            be autofilled. Defaults to None and will be set to {epoch}-{step}.
        monitor: Quantity to monitor. Defaults to None which saves a checkpoint
            only for the last epoch.
        save_last: When True, saves an exact copy of the checkpoint to a file
            `last.ckpt` whenever a checkpoint file gets saved. This allows
            accessing the latest checkpoint in a deterministic manner. Defaults
            to None.
        save_top_k:
            - If `save_top_k == k`, the best k models according to the quantity
              monitored will be saved.
            - If `save_top_k == 0`, no models are saved.
            - If `save_top_k == -1`, all models are saved. Please note that the
              monitors are checked every `every_n_epochs` epochs.
            - If `save_top_k >= 2` and the callback is called multiple times
              inside an epoch, the name of the saved file will be appended with
              a version count starting with `v1`.
        save_on_train_epoch_end: Whether to run checkpointing at the end of the
            training epoch. If this is False, then the check runs at the end of
            the validation. Defaults to False.
        save_weights_only: If True, then only the model's weights will be saved.
            Otherwise, the optimizer states, lr-scheduler states, etc. are added
            in the checkpoint too. Defaults to False.
        mode: One of {min, max}. If `save_top_k != 0`, the decision to overwrite
            the current save file is made based on either the maximization or
            the minimization of the monitored quantity. For `val_acc`, this
            should be `max`, for `val_loss` this should be `min`, etc.
        every_n_train_steps: Number of training steps between checkpoints.
            - If `every_n_train_steps == None` or `every_n_train_steps == 0`,
              we skip saving during training.
            - To disable, set `every_n_train_steps = 0`. This value must be
              None or non-negative.
            - This must be mutually exclusive with `train_time_interval` and
              `every_n_epochs`.
        train_time_interval: Checkpoints are monitored at the specified time
            interval. For all practical purposes, this cannot be smaller than
            the amount of time it takes to process a single training batch. This
            is not guaranteed to execute at the exact time specified, but should
            be close. This must be mutually exclusive with `every_n_train_steps`
            and `every_n_epochs`.
        every_n_epochs: Number of epochs between checkpoints. This value must be
            None or non-negative.
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
        auto_insert_metric_name:
            - When True, the checkpoints filenames will contain the metric name.
              For example, `filename='checkpoint_{epoch:02d}-{acc:02.0f}` with
              epoch `1` and acc `1.12` will resolve to `checkpoint_epoch=01-acc=01.ckpt`.
            - Is useful to set it to False when metric names contain `/` as this
              will result in extra folders. For example,
              `filename='epoch={epoch}-step={step}-val_acc={val/acc:.2f}', auto_insert_metric_name=False`
        verbose: Verbosity mode. Defaults to False.
    """

    checkpoint_join_char = "-"
    checkpoint_name_last = "last"
    checkpoint_name_best = "best"
    file_extension       = ".ckpt"
    starting_version     = 1
    
    mode_dict = {
        "min":  torch.tensor(np.Inf),
        "max": -torch.tensor(np.Inf)
    }
    
    def __init__(
        self,
        root                   : PathType         = RUNS_DIR,
        filename               : str       | None = None,
        monitor                : str       | None = None,
        save_last              : bool             = None,
        save_top_k             : int              = 1,
        save_on_train_epoch_end: bool             = False,
        save_weights_only      : bool             = False,
        mode                   : str              = "min",
        every_n_train_steps    : int       | None = None,
        train_time_interval    : timedelta | None = None,
        every_n_epochs         : int       | None = None,
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
        self.logger                  = None
        
        self.start_epoch                                = 0
        self.start_time                                 = 0
        self.last_epoch_saved                           = 0
        self.last_global_step_saved                     = 0  # no need to save when no steps were taken
        self.last_time_checked: float        | None     = None
        self.current_score    : torch.Tensor | None     = None
        self.best_model_score : torch.Tensor | None     = None
        self.best_k_models    : dict[str, torch.Tensor] = {}
        self.kth_best_model_path                        = None
        self.best_model_path                            = None
        self.last_model_path                            = None
        self.keys                                       = {}
        
        if mode not in self.mode_dict:
            raise ValueError(
                f"`mode` must be be {', '.join(self.mode_dict.keys())}. "
                f"But got: {mode}"
            )
        self.mode      = mode
        self.kth_value = self.mode_dict[mode]
        """
        if root and self.fs.protocol == "file":
            root = os.path.realpath(root)
        self.root = Path(root)
        if project is not None and project != "":
            self.root = self.root / project
        self.ckpt_dir = self.root / "weights"
        """
        self.fs       = lightning.pytorch.utilities.cloud_io.get_filesystem(root if root else "")
        self.root     = root
        self.ckpt_dir = root / "weights" if (root is not None) else None
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
        assert self.save_top_k >= -1
        assert self.every_n_train_steps >= 0
        assert self.every_n_epochs >= 0
        
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
        trainer  : lightning.Trainer,
        pl_module: lightning.LightningModule,
        stage    : str | None = None
    ):
        if self.root is None \
            or (hasattr(pl_module, "root") and self.root != pl_module.root):
            self.root     = pl_module.root
            self.ckpt_dir = self.root / "weights"
            
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
    
    def resolve_ckpt_dir(self, trainer: lightning.Trainer):
        """Determines model checkpoint save directory at runtime. Reference
        attributes from the trainer's logger to determine where to save
        checkpoints. The path for saving weights is set in this priority:

        1. The ModelCheckpoint's root if passed in
        2. The Trainer's weights_saved_path if passed in (deprecated)
        3. The Logger's log_dir if the trainer has loggers
        4. The Trainer's default_root_dir if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".
        """
        if self.root is not None:
            # Short circuit if dirpath was passed to ModelCheckpoint
            return

        # Remove weights_save_path logic here in v1.8
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
            version = lightning.pytorch.utilities.logger._version(trainer.loggers)
            version = version if isinstance(version, str) else f"version_{version}"

            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        ckpt_path = trainer.strategy.broadcast(ckpt_path)
        self.root = pathlib.Path(ckpt_path)
    
    def on_train_start(
        self,
        trainer  : lightning.Trainer,
        pl_module: lightning.LightningModule,
    ):
        self.start_epoch       = trainer.current_epoch
        self.start_time        = timer()
        self.last_time_checked = time.monotonic()
        self.logger            = open(self.root / "log.txt", "a", encoding="utf-8")
        self.logger.write(f"\n================================================================================\n")
        self.logger.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n\n")
        
        # Print model's info
        self.logger.write(f"{'Model':<10}: {pl_module.name}\n")
        self.logger.write(f"{'Data':<10}: {pl_module.fullname}\n")
        if hasattr(pl_module, "params"):
            self.logger.write(f"{'Parameters':<10}: {pl_module.params}\n")
        
        # Print header
        monitor   = self.monitor.replace("checkpoint/", "")
        monitor   = f"monitor_" + monitor.split("/")[0]
        headers   = f"\n{'Epoch':>10} {'step':>10} {monitor:>16} {'train_loss':>12} "
        self.keys = collections.OrderedDict(train_loss=0)
        if pl_module.train_metrics is not None:
            for m in pl_module.train_metrics:
                headers   += f"{f'train_{m.name}':>12} "
                self.keys |= {f'train_{m.name}': 0}
        headers   += f"{'val_loss':>12} "
        self.keys |= {"val_loss": 0}
        if pl_module.val_metrics is not None:
            for m in pl_module.val_metrics:
                headers   += f"{f'val_{m.name}':>12} "
                self.keys |= {f'val_{m.name}': 0}
        headers += f" {'reach':<16}"
    
        console.log(f"[bold]{headers}")
        self.logger.write(f"\nTraining:")
        self.logger.write(headers + "\n")
        self.logger.flush()
    
    def on_train_end(
        self,
        trainer  : lightning.Trainer,
        pl_module: lightning.LightningModule,
    ):
        end_time = timer()
        console.log(
            f"\n{trainer.current_epoch - self.start_epoch} epochs completed "
            f"in {(end_time - self.start_time):.3f} seconds "
            f"({((end_time - self.start_time) / 3600):.3f} hours)"
        )
        self.logger.write(
            f"\n{trainer.current_epoch - self.start_epoch} epochs completed "
            f"in {(end_time - self.start_time):.3f} seconds "
            f"({((end_time - self.start_time) / 3600):.3f} hours)"
        )
        self.logger.flush()
        self.logger.close()
    
    def on_train_batch_end(
        self,
        trainer  : lightning.Trainer,
        pl_module: lightning.LightningModule,
        outputs  : Any,
        batch    : Any,
        batch_idx: int,
    ):
        """Save checkpoint on train batch end if we meet the criteria for
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
        trainer  : lightning.Trainer,
        pl_module: lightning.LightningModule
    ):
        """Save a checkpoint at the end of the training epoch."""
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
        trainer  : lightning.Trainer,
        pl_module: lightning.LightningModule
    ):
        """Save a checkpoint at the end of the validation stage."""
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
            self.kth_best_model_path = pathlib.Path(self.kth_best_model_path)
            self.last_model_path     = pathlib.Path(self.last_model_path)
        else:
            error_console.log(
                f"The `root` has changed from {root_from_ckpt!r} to "
                f"{self.root!r}, therefore `best_model_score`, "
                f"`kth_best_model_path`, `kth_value`, `last_model_path` and"
                f" `best_k_models` won't be reloaded. Only `best_model_path` "
                f"will be reloaded."
            )

        self.best_model_path        = pathlib.Path(state_dict["best_model_path"])
        self.last_epoch_saved       = state_dict.get("last_epoch_saved",       self.last_epoch_saved)
        self.last_global_step_saved = state_dict.get("last_global_step_saved", self.last_global_step_saved)
    
    def save_topk_checkpoint(
        self,
        trainer           : lightning.Trainer,
        monitor_candidates: dict[str, torch.Tensor]
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
        trainer           : lightning.Trainer,
        monitor_candidates: dict[str, torch.Tensor]
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
        trainer           : lightning.Trainer,
        monitor_candidates: dict[str, torch.Tensor]
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
            epoch = monitor_candidates.pop("epoch")
            step  = monitor_candidates.pop("step")
            for c, v in monitor_candidates.items():
                if "epoch" not in c:
                    continue
                c = c.replace("checkpoint/", "")
                m = c.split("/")[0]
                c = f"train_{m}" if "train" in c else f"val_{m}"
                self.keys[c] = v
            row1 = f"{epoch:>10d} {step:>10d} {current:>16.6f} "
            row2 = f"{epoch:>10d} {step:>10d} {current:>16.6f} "
            for _, v in self.keys.items():
                row1 += f"{v:>12.6f} "
                row2 += f"{v:>12.6f} "
            row1 += f" {f'not in top {self.save_top_k}':<16.6s}"
            row2 += f" {f'not in top {self.save_top_k}':<16.6s}\n"
            
            console.log(row1)
            self.logger.write(row2)
            self.logger.flush()

    def save_none_monitor_checkpoint(
        self,
        trainer           : lightning.Trainer,
        monitor_candidates: dict[str, torch.Tensor]
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
        current           : torch.Tensor,
        trainer           : lightning.Trainer,
        monitor_candidates: dict[str, torch.Tensor]
    ):
        k = len(self.best_k_models) + 1 \
            if self.save_top_k == -1 \
            else self.save_top_k

        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(str(del_filepath))

        # Do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
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
        
        is_new_best = str(filepath) == str(self.best_model_path)
        epoch       = monitor_candidates.pop("epoch")
        step        = monitor_candidates.pop("step")
        for c, v in monitor_candidates.items():
            if "epoch" not in c:
                continue
            c = c.replace("checkpoint/", "")
            m = c.split("/")[0]
            c = f"train_{m}" if "train" in c else f"val_{m}"
            self.keys[c] = v
        if is_new_best:
            if self.verbose and trainer.is_global_zero:
                row1 = f"{epoch:>10d} {step:>10d} [red]{current:>16.6f}[default] "
                row2 = f"{epoch:>10d} {step:>10d} {current:>16.6f} "
                for _, v in self.keys.items():
                    row1 += f"{v:>12.6f} "
                    row2 += f"{v:>12.6f} "
                row1 += f" [red]{'best':<16.6s}[default]"
                row2 += f" {'best':<16.6s}\n"
                console.log(row1)
                self.logger.write(row2)
            self._save_best_checkpoint(trainer=trainer, filepath=filepath)
        else:
            if self.verbose and trainer.is_global_zero:
                row1 = f"{epoch:>10d} {step:>10d} [orange1]{current:>16.6f}[default] "
                row2 = f"{epoch:>10d} {step:>10d} {current:>16.6f} "
                for _, v in self.keys.items():
                    row1 += f"{v:>12.6f} "
                    row2 += f"{v:>12.6f} "
                row1 += f" [orange1]{f'top {k}':<16.6s}[default]"
                row2 += f" {f'top {k}':<16.6s}\n"
                console.log(row1)
                self.logger.write(row2)
        
        self.logger.flush()
        self._save_checkpoint(trainer=trainer, filepath=filepath)
        self._save_last_checkpoint(trainer=trainer, filepath=filepath)
        
        if del_filepath is not None and filepath != del_filepath:
            trainer.strategy.remove_checkpoint(del_filepath)

    def _save_checkpoint(
        self,
        trainer : lightning.Trainer,
        filepath: PathType,
    ):
        filepath = pathlib.Path(filepath)
        trainer.save_checkpoint(
            filepath     = filepath,
            weights_only = self.save_weights_only
        )
        self.last_epoch_saved       = trainer.current_epoch
        self.last_global_step_saved = trainer.global_step
        
        # Notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(_weakref.proxy(self))
    
    def _save_best_checkpoint(
        self,
        trainer : lightning.Trainer,
        filepath: PathType,
    ):
        filepath = pathlib.Path(filepath)
        filepath = filepath.parent / f"{self.checkpoint_name_best}.pt"
        trainer.save_checkpoint(
            filepath     = filepath,
            weights_only = True
        )
    
    def _save_last_checkpoint(
        self,
        trainer : lightning.Trainer,
        filepath: PathType,
    ):
        filepath = pathlib.Path(filepath)
        filepath = filepath.parent / f"{self.checkpoint_name_last}.pt"
        trainer.save_checkpoint(
            filepath     = filepath,
            weights_only = True
        )
    
    def should_skip_saving_checkpoint(self, trainer: lightning.Trainer) -> bool:
        from lightning.pytorch.trainer.states import TrainerFn
        return (
            bool(trainer.fast_dev_run)                             # disable checkpointing with fast_dev_run
            or trainer.state.fn != TrainerFn.FITTING               # don't save anything during non-fit
            or trainer.sanity_checking                             # don't save anything during sanity check
            or self.last_global_step_saved == trainer.global_step  # already saved at the last step
        )
    
    def monitor_candidates(
        self,
        trainer: lightning.Trainer
    ) -> dict[str, torch.Tensor]:
        candidates = copy.deepcopy(trainer.callback_metrics)
        # Cast to int if necessary because `self.log("epoch", 123)` will
        # convert it to float. If it's not a tensor or does not exist we
        # overwrite it as it's likely an error
        epoch = candidates.get("epoch")
        step  = candidates.get("step")
        candidates["epoch"] = epoch.int() \
            if isinstance(epoch, torch.Tensor) \
            else torch.tensor(trainer.current_epoch)
        candidates["step"] = step.int() \
            if isinstance(step, torch.Tensor) \
            else torch.tensor(trainer.global_step)
        return candidates
    
    def check_monitor_top_k(
        self,
        trainer: lightning.Trainer,
        current: torch.Tensor | None = None
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
        monitor_candidates: dict[str, torch.Tensor],
        trainer           : lightning.Trainer,
        del_filepath      : PathType | None = None
    ) -> PathType:
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
        metrics : dict[str, torch.Tensor],
        filename: str | None = None,
        ver     : int | None = None
    ) -> PathType:
        """Generates a filename according to the defined template."""
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
        metrics                : dict[str, torch.Tensor],
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
    
    def reset_keys(self):
        for k, v in self.keys.items():
            self.keys[k] = 0
    
    def to_yaml(self, filepath: PathType | None = None):
        """Saves the param:`best_k_models` dict containing the checkpoint paths
        with the corresponding scores to a YAML file.
        """
        best_k = {k: v.item() for k, v in self.best_k_models.items()}
        if filepath is None:
            assert self.ckpt_dir
            filepath = os.path.join(self.ckpt_dir, "best_k_models.yaml")
        with self.fs.open(filepath, "w") as fp:
            yaml.dump(best_k, fp)
    
    def file_exists(
        self,
        filepath: PathType,
        trainer : lightning.Trainer
    ) -> bool:
        """Checks if a file exists on rank 0 and broadcasts the result to all
        other ranks, preventing the internal state to diverge between ranks.
        """
        exists = self.fs.exists(str(filepath))
        return trainer.strategy.broadcast(exists)
    
    def warn_if_dir_not_empty(self, dirpath: PathType):
        if self.save_top_k != 0 \
            and self.fs.isdir(dirpath) \
            and len(self.fs.ls(dirpath)) > 0:
            console.log(
                f"Checkpoint directory {dirpath} exists and is not empty."
            )

# endregion
