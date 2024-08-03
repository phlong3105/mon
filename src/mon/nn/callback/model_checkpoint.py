#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the checkpoint callback to save model automatically
during training.
"""

from __future__ import annotations

__all__ = [
    "ModelCheckpoint",
]

import os
import time
from datetime import timedelta
from typing import Any
from weakref import proxy

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks

from mon import core
from mon.globals import CALLBACKS

console      = core.console
error_console = core.error_console


# region Model Checkpoint

@CALLBACKS.register(name="model_checkpoint")
class ModelCheckpoint(callbacks.ModelCheckpoint):
    """Save the model periodically by monitoring a quantity. We modified the
    original code to save only the best and last checkpoints by always setting
    :math:`save_top_k = 1`.
    
    See Also: :class:`lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`.
    """
    
    CHECKPOINT_JOIN_CHAR = "_"
    CHECKPOINT_NAME_LAST = "last"
    CHECKPOINT_NAME_BEST = "best"
    FILE_EXTENSION       = ".ckpt"
    STARTING_VERSION     = 1
    
    def __init__(
        self,
        dirpath                : str       | None = None,
        filename               : str       | None = None,
        monitor                : str       | None = None,
        mode                   : str              = "min",
        save_last              : bool             = False,
        save_top_k             : int              = 1,
        save_weights_only      : bool             = False,
        every_n_epochs         : int       | None = 1,
        every_n_train_steps    : int       | None = None,
        train_time_interval    : timedelta | None = None,
        save_on_train_epoch_end: bool             = False,
        enable_version_counter : bool             = False,
        auto_insert_metric_name: bool             = True,
        verbose                : bool             = False,
    ):
        if dirpath is not None:
            dirpath = core.Path(dirpath)
            dirpath = dirpath / "weights" if dirpath.name != "weights" else dirpath
            dirpath = str(dirpath)
            
        super().__init__(
            dirpath                 = dirpath,
            filename                = filename,
            monitor                 = monitor,
            mode                    = mode,
            save_last               = save_last,
            save_top_k              = 1,
            save_weights_only       = save_weights_only,
            every_n_epochs          = every_n_epochs,
            every_n_train_steps     = every_n_train_steps,
            train_time_interval     = train_time_interval,
            save_on_train_epoch_end = save_on_train_epoch_end,
            enable_version_counter  = enable_version_counter,
            auto_insert_metric_name = auto_insert_metric_name,
            verbose                 = verbose,
        )
        self._last_epoch_saved = 0
    
    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        # Make sure that we have the correct filename (1)
        if (
            hasattr(pl_module, "fullname")
            and pl_module.fullname is not None
            and self.filename is None
        ):
            self.filename = pl_module.fullname
    
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._last_time_checked = time.monotonic()
        # Make sure that we have the correct filename (2)
        if (
            hasattr(pl_module, "fullname")
            and pl_module.fullname is not None
            and self.filename is None
        ):
            self.filename = pl_module.fullname
        
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor                 = self.monitor,
            mode                    = self.mode,
            last_epoch_saved        = self._last_epoch_saved,
            last_global_step_saved  = self._last_global_step_saved,
            every_n_epochs          = self._every_n_epochs,
            every_n_train_steps     = self._every_n_train_steps,
            train_time_interval     = self._train_time_interval,
            save_on_train_epoch_end = self._save_on_train_epoch_end,
        )
    
    def state_dict(self) -> dict[str, Any]:
        return {
            "dirpath"               : self.dirpath,
            "monitor"               : self.monitor,
            "last_epoch_saved"      : int(self._last_epoch_saved),
            "last_global_step_saved": int(self._last_global_step_saved),
            "best_model_score"      : self.best_model_score,
            "best_model_path"       : self.best_model_path,
            "current_score"         : self.current_score,
            "best_k_models"         : self.best_k_models,
            "kth_best_model_path"   : self.kth_best_model_path,
            "kth_value"             : self.kth_value,
            "last_model_path"       : self.last_model_path,
        }
    
    def load_state_dict(self, state_dict: dict[str, Any]):
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)
        
        if self.dirpath == dirpath_from_ckpt:
            self.best_model_score    = state_dict["best_model_score"]
            self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
            self.kth_value           = state_dict.get("kth_value",           self.kth_value)
            self.best_k_models       = state_dict.get("best_k_models",       self.best_k_models)
            self.last_model_path     = state_dict.get("last_model_path",     self.last_model_path)
        else:
            error_console.log(
                f"The dirpath has changed from {dirpath_from_ckpt!r} to "
                f"{self.dirpath!r}, therefore `best_model_score`, "
                f"`kth_best_model_path`, `kth_value`, `last_model_path` and "
                f"`best_k_models` won't be reloaded. Only `best_model_path` "
                f"will be reloaded."
            )
        
        self.best_model_path         = state_dict["best_model_path"]
        # self.kth_best_model_path     = core.Path(self.kth_best_model_path)
        # self.last_model_path         = core.Path(self.last_model_path)
        self._last_epoch_saved       = int(state_dict.get("last_epoch_saved",       self._last_epoch_saved))
        self._last_global_step_saved = int(state_dict.get("last_global_step_saved", self._last_global_step_saved))
    
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str):
        filepath_pt = str(filepath).replace(self.FILE_EXTENSION, ".pt")
        trainer.save_checkpoint(filepath_pt, True)
        trainer.save_checkpoint(filepath, self.save_weights_only)
        
        self._last_epoch_saved       = int(trainer.current_epoch)
        self._last_global_step_saved = int(trainer.global_step)
        self._last_checkpoint_saved  = filepath
        
        # Notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
    
    def __init_triggers(
        self,
        every_n_train_steps: int       | None,
        every_n_epochs     : int       | None,
        train_time_interval: timedelta | None,
    ):
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            every_n_epochs      = 1
            every_n_train_steps = 0
            console.log(
                "Both every_n_train_steps and every_n_epochs are not set. "
                "Setting every_n_epochs=1"
            )
        else:
            every_n_epochs      = every_n_epochs      or 0
            every_n_train_steps = every_n_train_steps or 0
        
        self._train_time_interval = train_time_interval
        self._every_n_epochs      = every_n_epochs
        self._every_n_train_steps = every_n_train_steps
    
    def _format_checkpoint_name(
        self,
        filename               : str | None,
        metrics                : dict[str, torch.Tensor],
        prefix                 : str  = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        if "last" in filename:
            filename = self.CHECKPOINT_NAME_LAST
        else:
            filename = self.CHECKPOINT_NAME_BEST
            if auto_insert_metric_name:
                metric_name = self._parse_metric_name()
                if metric_name:
                    filename += f"_{metric_name}"
        if self.filename not in [None, "None", ""]:
            filename = f"{self.filename}_{filename}"
        if prefix:
            filename = self.CHECKPOINT_JOIN_CHAR.join([prefix, filename])
        return filename
    
    def __resolve_ckpt_dir(self, trainer: "pl.Trainer") -> str:
        """Determines model checkpoint save directory at runtime. Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

        1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
        2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
        3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

        The path gets extended with subdirectory "checkpoints".

        """
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath
        else:
            # if no loggers, use default_root_dir
            # dirpath = os.path.join(trainer.default_root_dir, "weights")
            dirpath = trainer.default_root_dir
            core.Path(dirpath).mkdir(parents=True, exist_ok=True)
            return dirpath
    
    def __warn_if_dir_not_empty(self, dirpath: core.Path | str) -> None:
        if (self.save_top_k != 0
            and self._fs.isdir(dirpath)
            and len(self._fs.ls(dirpath)) > 0
        ):
            console.log(f"Checkpoint directory {dirpath} exists and is not empty.")
    
    def _save_last_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: str[str, torch.Tensor]):
        if not self.save_last:
            return
        
        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST)
        
        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath     = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_LAST, ver=version_cnt)
                version_cnt += 1
        
        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        # if self.save_last == "link" and self._last_checkpoint_saved and self.save_top_k != 0:
        #     self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        # else:
        #     self._save_checkpoint(trainer, filepath)
        self._save_checkpoint(trainer, filepath)
        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            self._remove_checkpoint(trainer, previous)
            
    def _save_monitor_checkpoint(self, trainer: "pl.Trainer", monitor_candidates: dict[str, torch.Tensor]):
        assert self.monitor
        
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)
        elif self.verbose:
            if trainer.is_global_zero:
                epoch = monitor_candidates["epoch"]
                step  = monitor_candidates["step"]
                console.log(f"{f'{self.monitor}':>25} was not in top {self.save_top_k}")
    
    def _update_best_and_save(
        self,
        current           : torch.Tensor,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, torch.Tensor]
    ):
        k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k
        
        del_filepath = None
        if len(self.best_k_models) == k and k > 0:
            del_filepath = self.kth_best_model_path
            self.best_k_models.pop(del_filepath)
        
        # Do not save nan, replace with +/- inf
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)
        
        filepath = self.format_checkpoint_name(monitor_candidates, self.CHECKPOINT_NAME_BEST)
        
        # Save the current score
        self.current_score           = current
        self.best_k_models[filepath] = current
        
        if len(self.best_k_models) == k:
            # Monitor dict has reached k elements
            _op = max if self.mode == "min" else min
            self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.kth_value           = self.best_k_models[self.kth_best_model_path]
        
        _op = min if self.mode == "min" else max
        self.best_model_path  = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
        self.best_model_score = self.best_k_models[self.best_model_path]
        
        if self.verbose:
            if trainer.is_global_zero:
                epoch = monitor_candidates["epoch"]
                step  = monitor_candidates["step"]
                console.log(f"{f'{self.monitor}':>25} reached {current:>12.6f}, [red]saving as top {k}")
        self._save_checkpoint(trainer, filepath)
        
        if del_filepath and self._should_remove_checkpoint(trainer, del_filepath, filepath):
            self._remove_checkpoint(trainer, del_filepath)
    
    def _parse_metric_name(self) -> str | None:
        if self.monitor is None:
            return None
        
        metric_name = self.monitor
        for old in ["train", "val", "test", "/"]:
            metric_name = metric_name.replace(old, "")
        return metric_name

# endregion
