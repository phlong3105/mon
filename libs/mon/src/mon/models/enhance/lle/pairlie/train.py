#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for PairLIE models.
"""

from __future__ import annotations

__all__ = [
    "PairLIE_Trainer",
]

from typing import Any

import pyiqa
import torch
from rich.progress import Progress
from tensordict import TensorDict
from typing_extensions import override

from mon.core import K, OPTIMIZERS, Path, SCHEDULERS, TRAINERS
from mon.runners import Trainer
from . import loss as L
from .model import pairlie

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

@TRAINERS.register(name="pairlie")
class PairLIE_Trainer(Trainer):
    """Trainer for PairLIE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = pairlie(**config.model | {"weights": weights})
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        config = self.config
        lr_scheduler = config.lr_scheduler
        decay = lr_scheduler.pop("decay", 100)

        milestones = []
        for i in range(1, config.epochs + 1):
            if i % decay == 0:
                milestones.append(i)
        lr_scheduler["milestones"] = milestones

        self._optimizer = OPTIMIZERS.build(params=self.model.parameters(), **config.optimizer)
        self._scheduler = SCHEDULERS.build(optimizer=self._optimizer, **config.lr_scheduler)

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
        config = self.config
        device = self.device

        # 1. Define losses
        L_C = L.L_C().to(device)
        L_R = L.L_R().to(device)
        L_P = L.L_P().to(device)
        # Loss weights
        W_C = config.loss.W_C
        W_R = config.loss.W_R
        W_P = config.loss.W_P

        # 2. Train loop
        train_outputs = {}
        losses = []

        task = pbar.add_task(
            f"[bright_yellow]Train Epoch {epoch+1:03}",
            total=len(self._train_dataloader)
        )
        for i, datapoint in enumerate(self._train_dataloader):
            # 2.1. Prepare inputs
            datapoint = datapoint.to(device)
            image = datapoint["image"]
            target = datapoint["target"]

            # 2.2. Forward pass
            outputs1 = self.model(data={"image": image}, save_debug=True)
            outputs2 = self.model(data={"image": target}, save_debug=True)

            # 2.3. Extract outputs
            L1, R1, X1 = outputs1["L"], outputs1["R"], outputs1["X"]
            L2, R2, X2 = outputs2["L"], outputs2["R"], outputs2["X"]

            # 2.4. Calculate loss
            # Enhance loss
            l_C = W_C * L_C(R1, R2)
            l_R = W_R * L_R(L1, R1, image, X1)
            l_P = W_P * L_P(image, X1)
            # Total loss
            loss = l_C + l_R + l_P

            # 2.5. Backward pass
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())

            pbar.update(task, advance=1)
        pbar.remove_task(task)

        # 3. Output
        train_outputs |= {
            "loss": sum(losses) / len(losses),
        }
        return TensorDict(train_outputs, batch_size=[]).cpu()

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
        config = self.config
        device = self.device

        # 1. Define metrics
        psnr_metric = pyiqa.create_metric("psnr", device=device)
        ssim_metric = pyiqa.create_metric("ssim", device=device)
        ssimc_metric = pyiqa.create_metric("ssimc", device=device)

        # 2. Val loop
        val_outputs = {}
        psnrs = []
        ssims = []
        ssimcs = []

        task = pbar.add_task(
            f"[bright_cyan]Val Epoch {epoch+1:03}",
            total=len(self._val_dataloader)
        )
        for i, datapoint in enumerate(self._val_dataloader):
            # 2.1. Prepare inputs
            datapoint = datapoint.to(device)
            image = datapoint["image"]
            target = datapoint["target"]

            # 2.2. Forward pass
            outputs = self.model(data=datapoint, save_debug=True)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            L = outputs["L"]
            R = outputs["R"]
            X = outputs["X"]
            D = outputs["D"]

            # 2.4. Calculate metrics
            psnrs.append(psnr_metric(enhanced, target).detach())
            ssims.append(ssim_metric(enhanced, target).detach())
            ssimcs.append(ssimc_metric(enhanced, target).detach())

            # 2.5. Debug outputs
            if i == 0:
                val_outputs |= {
                    "image": image,
                    "target": target,
                    "enhanced": enhanced,
                    "L": L,
                    "R": R,
                    "X": X,
                    "D": D,
                }

            pbar.update(task, advance=1)
        pbar.remove_task(task)

        # 3. Output
        val_outputs |= {
            "psnr": torch.cat(psnrs).mean().item(),
            "ssim": torch.cat(ssims).mean().item(),
            "ssimc": torch.cat(ssimcs).mean().item(),
        }
        return TensorDict(val_outputs, batch_size=[]).cpu()

    # --- Output ---
    @override
    def _save_debug(self, epoch: int, train_outputs: TensorDict, val_outputs: TensorDict):
        """Save debugging results for visualization.

        Args:
            epoch (int): The current epoch number.
            train_outputs (TensorDict): The outputs from the training epoch.
            val_outputs (TensorDict): The outputs from the validation epoch.
        """
        debug_image = {
            "image": val_outputs["image"],
            "target": val_outputs["target"],
            "enhanced": val_outputs["enhanced"],
            "L": val_outputs["L"],
            "R": val_outputs["R"],
            "X": val_outputs["X"],
            "D": val_outputs["D"],
        }
        self._save_image(
            epoch=epoch,
            outputs=debug_image,
            dirname=K.PRED_DIR,
            stem="debug",
            column_first=True,
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
