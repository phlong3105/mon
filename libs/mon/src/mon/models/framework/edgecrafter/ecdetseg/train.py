#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for EdgeCrafter models.
"""

from __future__ import annotations

__all__ = [
    "ECDetSeg_Trainer",
]

import argparse

import torch
from rich.progress import Progress
from tensordict import TensorDict
from typing_extensions import override

from mon.core import ConfigContext, Path, RunMode, TRAINERS, Weights
from mon.runners import Trainer
from .engine.core import YAMLConfig
from .engine.misc import dist_utils
from .engine.solver import TASKS

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

@TRAINERS.register(name="ecdet")
@TRAINERS.register(name="ecseg")
class ECDetSeg_Trainer(Trainer):
    """Trainer for EdgeCrafter models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute.

        For this method, we only need to update the configuration inside
        and the solver will handle the rest.
        """
        config = self.config

        args = config.extra
        args.cfg_path = config.config_file.parent / args.cfg_paths

        # Resolve model's weights
        if config.get("resume"):
            args.resume = Weights(path=Path(config.resume)).rectify_path(root=config.root)
            args.tuning = None
        elif config.get("tuning"):
            args.resume = None
            args.tuning = Weights(path=Path(config.tuning)).rectify_path(root=config.root)
        else:
            args.resume = config.weights
            args.tuning = None

        self.config.extra = args

    @override
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes.

        For this method, we only need to update the configuration inside
        and the solver will handle the rest.
        """
        pass

    # --- Creation ---
    @classmethod
    def from_cli(cls, prompt: bool = False, **kwargs) -> "Trainer":
        """Create an instance of Trainer from command-line arguments.

        Args:
            prompt (bool, optional): Whether to prompt the user for input if
                necessary. Defaults to False.
        """
        # Additional model's arguments for training
        parser = argparse.ArgumentParser(description="edgecrafter")
        parser.add_argument("-r", "--resume", type=str, help="Resume from checkpoint")
        parser.add_argument("-t", "--tuning", type=str, help="Tuning from checkpoint")
        parser.add_argument("--use-amp",      action="store_true", help="Auto mixed precision training")
        parser.add_argument("--test-only",    action="store_true", default=False)
        parser.add_argument("--print-method", type=str, default="builtin", help="Print method")
        parser.add_argument("--print-rank",   type=int, default=0, help="Print rank id")
        parser.add_argument("--local-rank",   type=int, help="Local rank id")
        args = vars(parser.parse_args())
        kwargs |= args

        config_ctx = ConfigContext.from_cli(**kwargs)
        config = config_ctx.config_for(RunMode.TRAIN, prompt=prompt)
        return cls(config)

    # --- Control ---
    @override
    def train(self):
        """Train the model."""
        config = self.config

        # 1. Setup
        self._setup()

        dist_utils.setup_distributed(
            print_rank=config.print_rank,
            print_method=config.print_method,
            seed=config.seed,
        )

        # 2. Create solver
        args = config.extra
        cfg = YAMLConfig(**args)

        if args.resume or args.tuning:
            if "ViTAdapter" in cfg.yaml_cfg:
                cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

        solver = TASKS[cfg.yaml_cfg["task"]](cfg)

        # 3. Main loop
        if config.test_only:
            solver.val()
        else:
            solver.fit()

        dist_utils.cleanup()

    # --- Training ---
    @override
    def _train_epoch(self, epoch: int, pbar: Progress) -> TensorDict:
        """Train an epoch.

        Args:
            epoch (int): The current epoch number.
            pbar (Progress): The progress bar object.

        Returns:
            TensorDict: A dictionary containing the training loss and other
                results for the epoch.
        """
        pass

    # --- Validation ---
    @override
    @torch.no_grad()
    def _val_epoch(self, epoch: int, pbar: Progress) -> TensorDict:
        """Validate an epoch.

        Args:
            epoch (int): The current epoch number.
            pbar (Progress): The progress bar object.

        Returns:
            TensorDict: A dictionary containing the validation metrics and
                other results for the epoch.
        """
        pass

    # --- Output ---
    @override
    def _save_debug(self, epoch: int, train_outputs: TensorDict, val_outputs: TensorDict):
        """Save debugging results for visualization.

        Args:
            epoch (int): The current epoch number.
            train_outputs (TensorDict): The outputs from the training epoch.
            val_outputs (TensorDict): The outputs from the validation epoch.
        """
        pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
