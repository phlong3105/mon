#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for IZ-DCE and IZ-DCE-ODE models.
"""

from __future__ import annotations

__all__ = [
    "IZDCE_ODE_Trainer",
    "IZDCE_Trainer",
]

import pyiqa
import torch
from rich.progress import Progress
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing_extensions import override

from mon.core import OPTIMIZERS, Path, resolve_project_root, RunMode, Task
from mon.runners import Trainer
from . import loss as L
from .model import iz_dce, iz_dce_ode

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

class IZDCE_Trainer(Trainer):
    """Trainer for IZ-DCE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = iz_dce(**config.model | { "weights": weights})
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
        L_tv_A = L.L_tv().to(device)
        L_spa = L.L_spa().to(device)
        L_col = L.L_col().to(device)
        L_col_pre = L.L_col_pre().to(device)
        L_exp = L.L_exp(16, config.loss.E).to(device)
        # Loss weights
        L_tv_A_w = config.loss.L_tv_A_w
        L_spa_w = config.loss.L_spa_w
        L_col_w = config.loss.L_col_w
        L_col_pre_w = config.loss.L_col_pre_w
        L_exp_w = config.loss.L_exp_w
        L_enhance_w = config.loss.L_enhance_w
        L_denoise_w = config.loss.L_denoise_w

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
            depth = depth.to(device) if depth is not None else None

            # 2.2. Forward pass
            outputs = self.model(image, depth)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            A = outputs["A"]

            # 2.4. Calculate loss
            # Enhance loss
            l_tv_A = L_tv_A_w * torch.mean(L_tv_A(A, depth))
            l_spa = L_spa_w * L_spa(image, enhanced, depth)
            l_col = L_col_w * L_col(enhanced)
            l_col_pre = L_col_pre_w * L_col_pre(image, enhanced)
            l_exp = L_exp_w * L_exp(enhanced)
            l_enhance = l_tv_A + l_spa + l_col + l_col_pre + l_exp
            # Denoise loss
            l_denoise = torch.mean( outputs["l_denoise"])
            # Total loss
            loss = (L_enhance_w * l_enhance) + (L_denoise_w * l_denoise)

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
            depth = datapoint.get("depth", None)
            depth = depth.to(device) if depth is not None else None
            target = datapoint["target"]
            target = target.to(device)

            # 2.2. Forward pass
            outputs = self.model(image, depth)

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
        self._save_image(epoch, debug_image)


class IZDCE_ODE_Trainer(IZDCE_Trainer):
    """Trainer for IZ-DCE-ODE models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = iz_dce_ode(**config.model | { "weights": weights})
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
        L_tv_A = L.L_tv().to(device)
        L_spa = L.L_spa().to(device)
        L_col = L.L_col().to(device)
        L_col_pre = L.L_col_pre().to(device)
        L_exp = L.L_exp(16, config.loss.E).to(device)
        # Loss weights
        L_tv_A_w = config.loss.L_tv_A_w
        L_spa_w = config.loss.L_spa_w
        L_col_w = config.loss.L_col_w
        L_col_pre_w = config.loss.L_col_pre_w
        L_exp_w = config.loss.L_exp_w
        L_enhance_w = config.loss.L_enhance_w
        L_denoise_w = config.loss.L_denoise_w

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
            depth = depth.to(device) if depth is not None else None
            eval_time = torch.tensor([0, 3]).float().to(device)

            # 2.2. Forward pass
            outputs = self.model(image, depth, eval_time)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            A = outputs["A"]

            # 2.4. Calculate loss
            # Enhance loss
            l_tv_A = L_tv_A_w * torch.mean(L_tv_A(A, depth))
            l_spa = L_spa_w * L_spa(image, enhanced, depth)
            l_col = L_col_w * L_col(enhanced)
            l_col_pre = L_col_pre_w * L_col_pre(image, enhanced)
            l_exp = L_exp_w * L_exp(enhanced)
            l_enhance = l_tv_A + l_spa + l_col + l_col_pre + l_exp
            # Denoise loss
            l_denoise = torch.mean(outputs["l_denoise"])
            # Total loss
            loss = (L_enhance_w * l_enhance) + (L_denoise_w * l_denoise)

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
            depth = datapoint.get("depth", None)
            depth = depth.to(device) if depth is not None else None
            target = datapoint["target"]
            target = target.to(device)
            eval_time = torch.tensor([0, 3]).float().to(device)

            # 2.2. Forward pass
            outputs = self.model(image, depth, eval_time=eval_time)

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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

def main1():
    """Unit test for IZDCE_Trainer."""
    trainer = IZDCE_Trainer.from_cli(
        root=resolve_project_root(current_dir),
        config_file="iz_dce_sice_me.yaml",
        task=Task.ENHANCE,
        mode=RunMode.TRAIN,
        arch="iz_dce",
        model="iz_dce",
        save=True,
        save_debug=True,
        exist_ok=False,
        verbose=True,
    )
    trainer.train()


def main2():
    trainer = IZDCE_ODE_Trainer.from_cli(
        root=resolve_project_root(current_dir),
        config_file="iz_dce_ode_sice_me.yaml",
        task=Task.ENHANCE,
        mode=RunMode.TRAIN,
        arch="iz_dce",
        model="iz_dce_ode",
        save=True,
        save_debug=True,
        exist_ok=False,
        verbose=True,
    )
    trainer.train()


if __name__ == "__main__":
    pass

# endregion
