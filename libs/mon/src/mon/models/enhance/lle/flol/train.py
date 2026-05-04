#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for FLOL models.
"""

from __future__ import annotations

__all__ = [
    "FLOL_Trainer",
]

import pyiqa
import torch
from rich.progress import Progress
from tensordict import TensorDict
from torch import nn
from typing_extensions import override

from mon.core import K, OPTIMIZERS, Path, SCHEDULERS, TRAINERS
from mon.runners import Trainer
from .model import flol

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

@TRAINERS.register(name="flol")
class FLOL_Trainer(Trainer):
    """Trainer for FLOL models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = flol(**config.model | {"weights": weights})
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        config = self.config

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
        L_l1 = nn.L1Loss().to(device)
        L_lpips = pyiqa.create_metric("lpips", net="vgg", as_loss=True, device=device)
        # Loss weights
        W_lpips = config.loss.W_lpips

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
            outputs = self.model(data=datapoint, save_debug=True)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            amplitude = outputs["amplitude"]

            # 2.4. Calculate loss
            l_l1 = L_l1(enhanced, target) + L_l1(amplitude, target)
            l_lpips = W_lpips * L_lpips(enhanced, target)
            # Total loss
            loss = l_l1 + l_lpips

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
            outputs = self.model(data=datapoint, save_debug=self.save_debug)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            enhanced = torch.clamp(enhanced, 0.0, 1.0)

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
        }
        self._save_image(
            epoch=epoch,
            outputs=debug_image,
            dirname=K.PRED_DIR,
            stem="debug",
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
