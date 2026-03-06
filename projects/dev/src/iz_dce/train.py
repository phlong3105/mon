#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides several metric evaluators.
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

from mon.core import log, OPTIMIZERS, Path
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

    # --- Properties ---
    @override
    def init_model(self):
        """Initialize ``self.model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = iz_dce(**config.model | { "weights": weights})
        model = model.to(device)
        model.train()
        self.model = model

    @override
    def init_optimizer(self):
        """Initialize ``self.optimizer`` and ``self.scheduler`` attributes."""
        config = self.config
        epochs = config.epochs

        self.optimizer = OPTIMIZERS.build(params=self.model.parameters(), **config.optimizer)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

    # --- Training ---
    @override
    def train_epoch(self, epoch: int, pbar: Progress) -> dict:
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
        self.model.train()
        grad_clip_norm = config.grad_clip_norm
        outputs = {}
        losses = []

        task = pbar.add_task(
            f"[bright_yellow]Train Epoch {epoch+1:03}",
            total=len(self.train_dataloader)
        )
        for j, datapoint in enumerate(self.train_dataloader):
            # 2.1. Prepare inputs
            image = datapoint["image"]
            depth = datapoint.get("depth", None)
            image = image.to(device)
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

        # 3. Step the scheduler at the end of the epoch
        if self.scheduler is not None:
            self.scheduler.step()

        # 4. Output
        outputs |= {
            "loss": sum(losses) / len(losses),
        }
        return outputs

    # --- Validation ---
    @override
    @torch.no_grad()
    def val_epoch(self, epoch: int, pbar: Progress) -> dict:
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
        outputs = {}
        psnrs = []
        ssims = []
        ssimcs = []

        task = pbar.add_task(
            f"[bright_cyan]Val Epoch {epoch+1:03}",
            total=len(self.val_dataloader)
        )
        for j, datapoint in enumerate(self.val_dataloader):
            # 2.1. Prepare inputs
            image = datapoint["image"]
            depth = datapoint.get("depth", None)
            target = datapoint["target"]
            image = image.to(device)
            depth = depth.to(device) if depth is not None else None
            target = target.to(device)

            # 2.2. Forward pass
            outputs = self.model(image, depth)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]

            # 2.4. Calculate metrics
            psnrs.append(torch.mean(psnr_metric(enhanced, target)))
            ssims.append(torch.mean(ssim_metric(enhanced, target)))
            ssimcs.append(torch.mean(ssimc_metric(enhanced, target)))

            # 2.5. Debug outputs
            if j == 0:
                outputs |= {
                    "image": image.cpu(),
                    "depth": depth.cpu() if depth is not None else None,
                    "target": target.cpu(),
                    "enhanced": enhanced.cpu(),
                }

            pbar.update(task, advance=1)
        pbar.remove_task(task)

        # 3. Output
        outputs |= {
            "psnr": sum(psnrs) / len(psnrs),
            "ssim": sum(ssims) / len(ssims),
            "ssimc": sum(ssimcs) / len(ssimcs),
        }
        return outputs

    # --- Utilities ---
    @override
    def log(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Log the training and validation results for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        config = self.config

        if config.verbose:
            current_lr = self.scheduler.get_last_lr()[0]
            loss = train_outputs.get("loss", float("nan"))
            psnr = val_outputs.get("psnr", float("nan"))
            ssim = val_outputs.get("ssim", float("nan"))
            ssimc = val_outputs.get("ssimc", float("nan"))
            log(
                f"Epoch: {(epoch + 1):03} | "
                f"LR: {current_lr:08.6f} | "
                f"Loss: {loss:08.6f} | "
                f"PSNR: {psnr:08.6f} | "
                f"SSIM: {ssim:08.6f} | "
                f"SSIM-C: {ssimc:08.6f}",
            )

    @override
    def save(self, epoch: int, train_outputs: dict, val_outputs: dict):
        """Save the model checkpoint for the current epoch.

        Args:
            epoch (int): The current epoch number.
            train_outputs (dict): The outputs from the training epoch.
            val_outputs (dict): The outputs from the validation epoch.
        """
        config = self.config

        # Save weights
        torch.save(self.model.state_dict(), config.output_dir / "last.pt")
        self.save_best_weights("loss", train_outputs["loss"], lower_is_better=True)
        self.save_best_weights("psnr", val_outputs["psnr"])
        self.save_best_weights("ssim", val_outputs["ssim"])
        self.save_best_weights("ssimc", val_outputs["ssimc"])

        # Save debug
        debug_image = {
            "image": val_outputs["image"],
            "depth": val_outputs["depth"],
            "target": val_outputs["target"],
            "enhanced": val_outputs["enhanced"],
        }
        self.save_debug_image(epoch, debug_image)


class IZDCE_ODE_Trainer(IZDCE_Trainer):
    """Trainer for IZ-DCE-ODE models."""

    # --- Properties ---
    @override
    def init_model(self):
        """Initialize ``self.model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = iz_dce_ode(**config.model | { "weights": weights})
        model = model.to(device)
        model.train()
        self.model = model

    # --- Training ---
    @override
    def train_epoch(self, epoch: int, pbar: Progress) -> dict:
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
        self.model.train()
        grad_clip_norm = config.grad_clip_norm
        outputs = {}
        losses = []

        task = pbar.add_task(
            f"[bright_yellow]Train Epoch {epoch+1:03}",
            total=len(self.train_dataloader)
        )
        for j, datapoint in enumerate(self.train_dataloader):
            # 2.1. Prepare inputs
            image = datapoint["image"]
            depth = datapoint.get("depth", None)
            image = image.to(device)
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
        outputs |= {
            "loss": sum(losses) / len(losses),
        }
        return outputs

    # --- Validation ---
    @override
    @torch.no_grad()
    def val_epoch(self, epoch: int, pbar: Progress) -> dict:
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
        outputs = {}
        psnrs = []
        ssims = []
        ssimcs = []

        task = pbar.add_task(
            f"[bright_cyan]Val Epoch {epoch+1:03}",
            total=len(self.val_dataloader)
        )
        for j, datapoint in enumerate(self.val_dataloader):
            # 2.1. Prepare inputs
            image = datapoint["image"]
            depth = datapoint.get("depth", None)
            target = datapoint["target"]
            image = image.to(device)
            depth = depth.to(device) if depth is not None else None
            target = target.to(device)
            eval_time = torch.tensor([0, 3]).float().to(device)

            # 2.2. Forward pass
            outputs = self.model(image, depth, eval_time=eval_time)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]

            # 2.4. Calculate metrics
            psnrs.append(torch.mean(psnr_metric(enhanced, target)))
            ssims.append(torch.mean(ssim_metric(enhanced, target)))
            ssimcs.append(torch.mean(ssimc_metric(enhanced, target)))

            # 2.5. Debug outputs
            if j == 0:
                outputs |= {
                    "image": image.cpu(),
                    "depth": depth.cpu() if depth is not None else None,
                    "target": target.cpu(),
                    "enhanced": enhanced.cpu(),
                }

            pbar.update(task, advance=1)
        pbar.remove_task(task)

        # 3. Output
        outputs |= {
            "psnr": sum(psnrs) / len(psnrs),
            "ssim": sum(ssims) / len(ssims),
            "ssimc": sum(ssimcs) / len(ssimcs),
        }
        return outputs

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
