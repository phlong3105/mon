#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for CLODE models.
"""

from __future__ import annotations

__all__ = [
    "CLODE_Trainer",
]

import pyiqa
import torch
from rich.progress import Progress
from typing_extensions import override

from mon.core import log, OPTIMIZERS, Path
from mon.models.enhance.clode import loss as L
from mon.runners import Trainer
from .model import clode

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

class CLODE_Trainer(Trainer):
    """Trainer for CLODE models."""

    # --- Properties ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = clode(**config.model | { "weights": weights})
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
        L_spa = L.L_spa().to(device)
        L_col = L.L_col().to(device)
        L_exp = L.L_exp(16, config.loss.L_exp_mean).to(device)
        # Loss weights
        L_tv_w = config.loss.L_tv_w
        L_spa_w = config.loss.L_spa_w
        L_col_w = config.loss.L_col_w
        L_exp_w = config.loss.L_exp_w

        # 2. Train loop
        self.model.train()
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
            eval_time = torch.tensor([0, 3]).float().to(device)

            # 2.2. Forward pass
            outputs = self.model(image, eval_time)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            A_map = outputs["curve_map"]
            noise_map = outputs["noise_map"]

            # 2.4. Calculate loss
            # Enhance loss
            l_param = L_tv_w  * torch.mean(A_map)
            l_col = L_col_w * L_col(enhanced)
            l_spa = L_spa_w * L_spa(enhanced, image)
            l_exp = L_exp_w * L_exp(enhanced)
            l_noise = torch.mean(noise_map)
            # Total loss
            loss = l_spa + l_col + l_exp + l_param + l_noise

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
            "loss": torch.cat(losses).mean().item(),
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
        self.model.eval()
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
            target = datapoint["target"]
            image = image.to(device)
            target = target.to(device)
            eval_time = torch.tensor([0, 3]).float().to(device)

            # 2.2. Forward pass
            outputs = self.model(image, eval_time, inference=True)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]

            # 2.4. Calculate metrics
            psnrs.append(psnr_metric(enhanced, target).detach().cpu())
            ssims.append(ssim_metric(enhanced, target).detach().cpu())
            ssimcs.append(ssimc_metric(enhanced, target).detach().cpu())

            # 2.5. Debug outputs
            if i == 0:
                val_outputs |= {
                    "image": image.detach().cpu(),
                    "target": target.detach().cpu(),
                    "enhanced": enhanced.detach().cpu(),
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

    # --- Utilities ---
    @override
    def _log(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Log the training and validation results for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        config = self.config

        if config.verbose:
            loss = train_outputs.get("loss", float("nan"))
            psnr = val_outputs.get("psnr", float("nan"))
            ssim = val_outputs.get("ssim", float("nan"))
            ssimc = val_outputs.get("ssimc", float("nan"))
            log(
                f"Epoch: {(epoch + 1):03} | "
                f"Loss: {loss:08.6f} | "
                f"PSNR: {psnr:08.6f} | "
                f"SSIM: {ssim:08.6f} | "
                f"SSIM-C: {ssimc:08.6f}",
            )

    @override
    def _save(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Save the model checkpoint for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        config = self.config

        torch.save(self.model.state_dict(), config.output_dir / "last.pt")
        self._save_best_weights("loss", train_outputs["loss"], lower_is_better=True)
        self._save_best_weights("psnr", val_outputs["psnr"])
        self._save_best_weights("ssim", val_outputs["ssim"])
        self._save_best_weights("ssimc", val_outputs["ssimc"])

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
        self._save_image(epoch, debug_image)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
