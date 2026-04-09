#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for SLICE models.
"""

from __future__ import annotations

__all__ = [
    "SLICE_Trainer",
]

from typing import Any

import pyiqa
import torch
from rich.progress import Progress
from typing_extensions import override

from mon.core import (
    OPTIMIZERS,
    Path,
    resolve_project_root,
    RunMode,
    Task,
    TRAINERS,
)
from mon.ops import normalize_minmax
from mon.runners import Trainer
from . import loss as L
from .model import slice

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

@TRAINERS.register(name="slice")
class SLICE_Trainer(Trainer):
    """Trainer for SLICE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = slice(**config.model | {"weights": weights})
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        config = self.config
        epochs = config.epochs

        self._optimizer = OPTIMIZERS.build(params=self.model.parameters(), **config.optimizer)
        # self._scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

    # --- Training ---
    @override
    def _train_epoch(self, epoch: int, pbar: Progress) -> dict[str, Any]:
        """Train an epoch.

        Args:
            epoch (int): The current epoch number.
            pbar (Progress): The progress bar object.

        Returns:
            dict[str, Any]: A dictionary containing the training loss and other
                results for the epoch.
        """
        config = self.config
        device = self.device
        use_depth = config.model.use_depth

        # 1. Define losses
        L_tv_A = L.L_tv().to(device)
        L_spa = L.L_spa().to(device)
        L_col = L.L_col().to(device)
        L_col_pre = L.L_col_pre().to(device)
        L_exp = L.L_exp(16, config.loss.E).to(device)
        # L_exp = L.L_exp_asym(16, config.loss.E).to(device)
        # Loss weights
        W_tv_A = config.loss.W_tv_A
        W_spa = config.loss.W_spa
        W_col = config.loss.W_col
        W_col_pre = config.loss.W_col_pre
        W_exp = config.loss.W_exp
        W_enhance = config.loss.W_enhance
        W_denoise = config.loss.W_denoise
        W_equi_A = config.loss.W_equi_A

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
            depth = datapoint.get("depth", None)
            depth = depth.to(device) if use_depth else None
            t = torch.tensor([0.0, config.T]).float().to(device)

            # 2.2. Forward pass
            outputs = self.model(image=image, depth=depth, t=t, save_debug=self.save_debug)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            curve_map = outputs["curve_map"]

            # 2.4. Calculate loss
            # Enhance loss
            l_tv_A = W_tv_A * torch.mean(L_tv_A(curve_map, depth))
            l_spa = W_spa * L_spa(image, enhanced, depth)
            l_col = W_col * L_col(enhanced)
            l_col_pre = W_col_pre * L_col_pre(image, enhanced)
            l_exp = W_exp * L_exp(enhanced)
            l_enhance = W_enhance * (l_tv_A + l_spa + l_col + l_col_pre + l_exp)
            # Denoise loss
            l_denoise = W_denoise * torch.mean(outputs["l_denoise"])
            # Equivariance loss
            if W_equi_A not in [0, None]:
                l_equi_A = W_equi_A * self.model.loss_equi_A(image, depth)
            else:
                l_equi_A = 0  # torch.tensor(0.0, device=device)
            # Total loss
            loss = l_enhance + l_denoise + l_equi_A

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
    def _val_epoch(self, epoch: int, pbar: Progress) -> dict[str, Any]:
        """Validate an epoch.

        Args:
            epoch (int): The current epoch number.
            pbar (Progress): The progress bar object.

        Returns:
            dict[str, Any]: A dictionary containing the validation metrics and
                other results for the epoch.
        """
        config = self.config
        device = self.device
        use_depth = config.model.use_depth

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
            depth = datapoint.get("depth", None)
            depth = depth.to(device) if use_depth else None
            target = datapoint["target"]
            target = target.to(device)
            t = torch.tensor([0.0, config.T]).float().to(device)

            # 2.2. Forward pass
            outputs = self.model(image=image, depth=depth, t=t, save_debug=True)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            curve_map = outputs["curve_map"]
            noise_map = outputs["noise_map"]
            denoised = outputs["denoised"]

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
                    "curve_map": curve_map.detach().cpu(),
                    "noise_map": noise_map.detach().cpu(),
                    "denoised": denoised.detach().cpu(),
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
    def _save_debug(
        self,
        epoch: int,
        train_outputs: dict[str, Any],
        val_outputs: dict[str, Any]
    ):
        """Save debugging results for visualization.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict[str, Any]): The outputs from the training epoch.
            val_outputs (dict[str, Any]): The outputs from the validation epoch.
        """
        debug_image = {
            "image": val_outputs["image"],
            "target": val_outputs["target"],
            "enhanced": val_outputs["enhanced"],
            "curve_map": normalize_minmax(val_outputs["curve_map"]),
            "noise_map": normalize_minmax(val_outputs["noise_map"]),
            "denoised": val_outputs["denoised"],
        }

        self._save_image(epoch, debug_image, column_first=True)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

def main():
    """Unit test for SLICE_Trainer."""
    trainer = SLICE_Trainer.from_cli(
        root=resolve_project_root(current_dir),
        config_file="slice_sice_me.yaml",
        task=Task.LLE,
        mode=RunMode.TRAIN,
        arch="slice",
        model="slice",
        device="auto",
        save=True,
        save_debug=True,
        exist_ok=False,
        verbose=True,
    )
    trainer.train()


if __name__ == "__main__":
    pass

# endregion
