#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the checkpoint callback to save model automatically
during training.
"""

from __future__ import annotations

__all__ = [
    "ModelCheckpoint",
]

import collections
import os
import time
from datetime import datetime, timedelta
from timeit import default_timer as timer
from typing import Any
from weakref import proxy

import lightning.pytorch as pl
import torch
from lightning.pytorch import callbacks

from mon.foundation import console, error_console, pathlib
from mon.globals import CALLBACKS


# region Model Checkpoint Callback

@CALLBACKS.register(name="model_checkpoint")
class ModelCheckpoint(callbacks.ModelCheckpoint):
    """Save the model periodically by monitoring a quantity.
    
    See Also: :class:`lightning.pytorch.callbacks.model_checkpoint.ModelCheckpoint`.
    """
    
    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_NAME_LAST = "last"
    CHECKPOINT_NAME_BEST = "last"
    FILE_EXTENSION       = ".ckpt"
    STARTING_VERSION     = 1
    
    def __init__(
        self,
        dirpath                : pathlib.Path,
        filename               : str | None       = None,
        monitor                : str | None       = None,
        verbose                : bool             = False,
        save_last              : bool             = None,
        save_top_k             : int              = 1,
        save_weights_only      : bool             = False,
        mode                   : str              = "min",
        auto_insert_metric_name: bool             = True,
        every_n_train_steps    : int | None       = None,
        train_time_interval    : timedelta | None = None,
        every_n_epochs         : int | None       = None,
        save_on_train_epoch_end: bool             = False,
    ):
        self.start_epoch      = 0
        self.start_time       = 0
        self.last_epoch_saved = 0
        self.best_model_score: torch.Tensor | None = None
        self.keys     = {}
        self.ckpt_dir = dirpath / "weights" if (dirpath is not None) else None
        self.logger   = None
        
        super().__init__(
            dirpath                 = dirpath,
            filename                = filename,
            monitor                 = monitor,
            save_last               = save_last,
            save_top_k              = save_top_k,
            save_on_train_epoch_end = save_on_train_epoch_end,
            save_weights_only       = save_weights_only,
            mode                    = mode,
            every_n_train_steps     = every_n_train_steps,
            train_time_interval     = train_time_interval,
            every_n_epochs          = every_n_epochs,
            auto_insert_metric_name = auto_insert_metric_name,
            verbose                 = verbose,
        )
        
    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor                 = self.monitor,
            last_epoch_saved        = self.last_epoch_saved,
            last_global_step_saved  = self._last_global_step_saved,
            mode                    = self.mode,
            every_n_train_steps     = self._every_n_train_steps,
            every_n_epochs          = self._every_n_epochs,
            train_time_interval     = self._train_time_interval,
            save_on_train_epoch_end = self._save_on_train_epoch_end,
        )

    def setup(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage    : str
    ) -> None:
        dirpath       = pathlib.Path(self.__resolve_ckpt_dir(trainer))
        dirpath       = trainer.strategy.broadcast(dirpath)
        self.dirpath  = dirpath
        self.ckpt_dir = self.dirpath / "weights"
        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)
    
    def on_train_start(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
    ) -> None:
        self._last_time_checked = time.monotonic()
        
        # Our extension
        self.start_epoch = trainer.current_epoch
        self.start_time  = timer()
        self.logger      = open(self.dirpath / "log.txt", "a", encoding = "utf-8")
        self.logger.write(
            f"\n================================================================================\n"
        )
        self.logger.write(f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}\n\n")
        
        # Print model's info
        self.logger.write(f"{'Model':<10}: {pl_module.name}\n")
        self.logger.write(f"{'Data':<10}: {pl_module.fullname}\n")
        if hasattr(pl_module, "params"):
            self.logger.write(f"{'Parameters':<10}: {pl_module.params}\n")
        
        # Print header
        monitor = self.monitor.replace("checkpoint/", "")
        monitor = f"monitor_" + monitor.split("/")[0]
        headers = f"\n{'Epoch':>10} {'step':>10} {monitor:>16} " \
                  f"{'train_loss':>12} "
        self.keys = collections.OrderedDict(train_loss=0)
        if pl_module.train_metrics is not None:
            for m in pl_module.train_metrics:
                headers += f"{f'train_{m.name}':>12} "
                self.keys |= {f'train_{m.name}': 0}
        headers += f"{'val_loss':>12} "
        self.keys |= {"val_loss": 0}
        if pl_module.val_metrics is not None:
            for m in pl_module.val_metrics:
                headers += f"{f'val_{m.name}':>12} "
                self.keys |= {f'val_{m.name}': 0}
        headers += f" {'reach':<16}"
        
        console.log(f"[bold]{headers}")
        self.logger.write(f"\nTraining:")
        self.logger.write(headers + "\n")
        self.logger.flush()
    
    def on_train_end(
        self,
        trainer  : "pl.Trainer",
        pl_module: "pl.LightningModule"
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
    
    def state_dict(self) -> dict[str, Any]:
        return {
            "dirpath"               : self.dirpath,
            "ckpt_dir"              : self.ckpt_dir,
            "last_epoch_saved"      : self.last_epoch_saved,
            "last_global_step_saved": self._last_global_step_saved,
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
        dirpath_from_ckpt = state_dict.get("dirpath", self.dirpath)
        
        if self.dirpath_from_ckpt == dirpath_from_ckpt:
            self.best_model_score    = state_dict["best_model_score"]
            self.kth_best_model_path = state_dict.get("kth_best_model_path", self.kth_best_model_path)
            self.kth_value           = state_dict.get("kth_value"          , self.kth_value)
            self.best_k_models       = state_dict.get("best_k_models"      , self.best_k_models)
            self.last_model_path     = state_dict.get("last_model_path"    , self.last_model_path)
            
            # Our extension
            self.kth_best_model_path = pathlib.Path(self.kth_best_model_path)
            self.last_model_path     = pathlib.Path(self.last_model_path)
        else:
            error_console.log(
                f"The dirpath has changed from {dirpath_from_ckpt!r} to "
                f"{self.dirpath!r}, therefore `best_model_score`, "
                f"`kth_best_model_path`, `kth_value`, `last_model_path` and "
                f"`best_k_models` won't be reloaded. Only `best_model_path` "
                f"will be reloaded."
            )
        
        self.best_model_path         = pathlib.Path(state_dict["best_model_path"])
        self.last_epoch_saved        = state_dict.get("last_epoch_saved",       self.last_epoch_saved)
        self._last_global_step_saved = state_dict.get("last_global_step_saved", self.last_global_step_saved)
    
    def _save_checkpoint(
        self,
        trainer : "pl.Trainer",
        filepath: pathlib.Path | str
    ):
        trainer.save_checkpoint(pathlib.Path(filepath), self.save_weights_only)
        
        self.last_epoch_saved        = trainer.current_epoch
        self._last_global_step_saved = trainer.global_step
        
        # Notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))

    def __init_triggers(
        self,
        every_n_train_steps: int | None,
        every_n_epochs     : int | None,
        train_time_interval: timedelta | None,
    ) -> None:
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
            every_n_epochs      = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0
        
        self._train_time_interval = train_time_interval
        self._every_n_epochs      = every_n_epochs
        self._every_n_train_steps = every_n_train_steps
    
    def format_checkpoint_name(
        self,
        metrics : dict[str, torch.Tensor],
        filename: str | None = None,
        ver     : int | None = None
    ) -> pathlib.Path | str:
        """Generates a file_name according to the defined template."""
        filename = filename or self.filename
        filename = self._format_checkpoint_name(
            filename                = filename,
            metrics                 = metrics,
            auto_insert_metric_name = self.auto_insert_metric_name
        )
        if ver is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{ver}"))
        
        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        # Our extension
        return (self.ckpt_dir / ckpt_name) if self.ckpt_dir else ckpt_name
    
    def _find_last_checkpoints(self, trainer: "pl.Trainer") -> set[str]:
        # Find all checkpoints in the folder
        ckpt_path = self.__resolve_ckpt_dir(trainer) / "weights"
        if self._fs.exists(ckpt_path):
            return {
                os.path.normpath(p)
                for p in self._fs.ls(ckpt_path, detail=False)
                if self.CHECKPOINT_NAME_LAST in os.path.split(p)[1]
            }
        return set()
    
    def __warn_if_dir_not_empty(self, dirpath: pathlib.Path | str) -> None:
        if self.save_top_k != 0 \
            and self._fs.isdir(dirpath) \
            and len(self._fs.ls(dirpath)) > 0:
            console.log(f"Checkpoint directory {dirpath} exists and is not empty.")
            
    def _save_monitor_checkpoint(
        self,
        trainer           : "pl.Trainer",
        monitor_candidates: dict[str, torch.Tensor]
    ):
        assert self.monitor
        current = monitor_candidates.get(self.monitor)
        if self.check_monitor_top_k(trainer, current):
            assert current is not None
            self._update_best_and_save(
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
    
    def update_best_and_save(
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
            current = torch.tensor(
                float("inf" if self.mode == "min" else "-inf"),
                device=current.device
            )
        
        filepath = self._get_metric_interpolated_filepath_name(
            monitor_candidates = monitor_candidates,
            trainer            = trainer,
            del_filepath       = del_filepath
        )
        
        # Save the current score
        self.current_score = current
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
        epoch = monitor_candidates.pop("epoch")
        step  = monitor_candidates.pop("step")
        for c, v in monitor_candidates.items():
            if "epoch" not in c:
                continue
            c = c.replace("checkpoint/", "")
            m = c.split("/")[0]
            c = f"train_{m}" if "train" in c else f"val_{m}"
            self.keys[c] = v
        if is_new_best:
            if self.verbose and trainer.is_global_zero:
                row1 = f"{epoch:>10d} {step:>10d} [red]{current:>16.6f}[" \
                       f"default] "
                row2 = f"{epoch:>10d} {step:>10d} {current:>16.6f} "
                for _, v in self.keys.items():
                    row1 += f"{v:>12.6f} "
                    row2 += f"{v:>12.6f} "
                row1 += f" [red]{'best':<16.6s}[default]"
                row2 += f" {'best':<16.6s}\n"
                console.log(row1)
                self.logger.write(row2)
            
            # Our extension
            trainer.save_checkpoint(
                filepath     = pathlib.Path(filepath).parent / f"{self.CHECKPOINT_NAME_BEST}.pt",
                weights_only = True
            )
        else:
            if self.verbose and trainer.is_global_zero:
                row1 = f"{epoch:>10d} {step:>10d} [orange1]{current:>16.6f}[" \
                       f"default] "
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
        
        # Our extension
        trainer.save_checkpoint(
            filepath     = pathlib.Path(filepath).parent / f"{self.CHECKPOINT_NAME_LAST}.pt",
            weights_only = True
        )
        
        if del_filepath is not None and filepath != del_filepath:
            self._remove_checkpoint(trainer, del_filepath)
    
# endregion
