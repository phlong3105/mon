#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for Zero-DCE and Zero-DCE++ models.
"""

from __future__ import annotations

__all__ = [
    "ZeroDCE_Trainer",
]

import pyiqa
import torch
from rich.progress import Progress
from typing_extensions import override

from mon.core import K, OPTIMIZERS, Path
from mon.runners import Trainer
from . import loss as L
from .model import zero_dce

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

class ZeroDCE_Trainer(Trainer):
    """Trainer for Zero-DCE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = zero_dce(**config.model | { "weights": weights})
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        config = self.config

        self._optimizer = OPTIMIZERS.build(params=self.model.parameters(), **config.optimizer)
        self._scheduler = None

    # --- Training ---
    @override
    def _train_epoch(self, epoch: int, pbar: Progress) -> dict:
        """Train an epoch.

        Args:
            epoch (int): The current epoch number.
            pbar (Progress): The progress bar object.

        Returns:
            dict: A dictionary containing the training loss and other results
                for the epoch.
        """
        config = self.config
        device = self.device

        # 1. Define losses
        L_tv = L.L_tv().to(device)
        L_spa = L.L_spa().to(device)
        L_col = L.L_col().to(device)
        L_exp = L.L_exp(16, config.loss.L_exp_mean).to(device)
        # Loss weights
        W_tv = config.loss.W_tv
        W_spa = config.loss.W_spa
        W_col = config.loss.W_col
        W_exp = config.loss.W_exp

        # 2. Train loop
        grad_clip_norm = config.grad_clip_norm
        train_outputs = {}
        losses = []

        task = pbar.add_task(
            f"[bright_yellow]Train Epoch {epoch+1:03}",
            total=len(self.train_dataloader)
        )
        for i, datapoint in enumerate(self.train_dataloader):
            # 2.1. Prepare inputs
            image = datapoint["image"]
            image = image.to(device)

            # 2.2. Forward pass
            outputs = self.model(image)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            r = outputs["r"]

            # 2.4. Calculate loss
            # Enhance loss
            l_tv = W_tv * L_tv(r)
            l_spa = W_spa * torch.mean(L_spa(enhanced, image))
            l_col = W_col * torch.mean(L_col(enhanced))
            l_exp = W_exp * torch.mean(L_exp(enhanced))
            # Total loss
            loss = l_tv + l_spa + l_col + l_exp

            # 2.5. Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            self.optimizer.step()
            losses.append(loss.item())

            pbar.update(task, advance=1)
        pbar.remove_task(task)

        # 3. Output
        train_outputs |= {
            "loss": sum(losses) / len(losses),
        }
        return train_outputs

    # --- Validation ---
    @override
    @torch.no_grad()
    def _val_epoch(self, epoch: int, pbar: Progress) -> dict:
        """Validate an epoch.

        Args:
            epoch (int): The current epoch number.
            pbar (Progress): The progress bar object.

        Returns:
            dict: A dictionary containing the validation metrics and other
                results for the epoch.
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
            total=len(self.val_dataloader)
        )
        for i, datapoint in enumerate(self.val_dataloader):
            # 2.1. Prepare inputs
            image = datapoint["image"]
            image = image.to(device)
            target = datapoint["target"]
            target = target.to(device)

            # 2.2. Forward pass
            outputs = self.model(image)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]

            # 2.4. Calculate metrics
            psnrs.append(psnr_metric(enhanced, target).detach().cpu())
            ssims.append(ssim_metric(enhanced, target).detach().cpu())
            ssimcs.append(ssimc_metric(enhanced, target).detach().cpu())

            # 2.5. Debug outputs
            if i == 0:
                val_outputs |= {
                    "image": image.cpu(),
                    "target": target.cpu(),
                    "enhanced": enhanced.cpu(),
                }

            pbar.update(task, advance=1)
        pbar.remove_task(task)

        # 3. Output
        val_outputs |= {
            "psnr": torch.cat(psnrs).mean().item(),
            "ssim": torch.cat(ssims).mean().item(),
            "ssimc": torch.cat(ssimcs).mean().item(),
        }
        return val_outputs

    # --- Output ---
    @override
    def _save_debug(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Save debugging results for visualization.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        debug_image = {
            "image": val_outputs["image"],
            "target": val_outputs["target"],
            "enhanced": val_outputs["enhanced"],
        }
        self._save_image(epoch, debug_image, dirname=K.PRED_DIR, stem="debug")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
