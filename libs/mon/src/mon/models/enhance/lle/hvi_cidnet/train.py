#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for HVI-CIDNet models.
"""

from __future__ import annotations

__all__ = [
    "HVI_CIDNet_Trainer",
]

import random

import pyiqa
import torch
from rich.progress import Progress
from tensordict import TensorDict
from typing_extensions import override

from mon.core import K, OPTIMIZERS, Path, TRAINERS
from mon.nn import (
    CosineAnnealingRestartCyclicLR,
    CosineAnnealingRestartLR,
    GradualWarmupScheduler,
)
from mon.runners import Trainer
from . import loss as L
from .model import hvi_cidnet

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

@TRAINERS.register(name="hvi_cidnet")
class HVI_CIDNet_Trainer(Trainer):
    """Trainer for HVI-CIDNet models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = hvi_cidnet(**config.model | {"weights": weights})
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        config = self.config

        # Optimizer
        self._optimizer = OPTIMIZERS.build(params=self.model.parameters(), **config.optimizer)

        # Scheduler
        lr_scheduler = config.lr_scheduler
        lr_warmup_scheduler = config.lr_warmup_scheduler
        epochs = config.epochs

        if lr_scheduler.name == "CosineAnnealingRestartCyclicLR":
            if lr_warmup_scheduler is not None:
                scheduler_step = CosineAnnealingRestartCyclicLR(
                    optimizer=self._optimizer,
                    periods=[
                        (epochs // 4) - lr_warmup_scheduler.warmup_epochs,
                        (epochs * 3) // 4,
                    ],
                    restart_weights=[1, 1],
                    eta_mins=[0.0002, 0.0000001],
                )
                scheduler = GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    multiplier=1,
                    total_epoch=lr_warmup_scheduler.warmup_epochs,
                    after_scheduler=scheduler_step,
                )
            else:
                scheduler = CosineAnnealingRestartCyclicLR(
                    optimizer=self._optimizer,
                    periods=[epochs // 4, (epochs * 3) // 4],
                    restart_weights=[1, 1],
                    eta_mins=[0.0002, 0.0000001],
                )
        elif lr_scheduler.name == "CosineAnnealingRestartLR":
            if lr_warmup_scheduler is not None:
                scheduler_step = CosineAnnealingRestartLR(
                    optimizer=self._optimizer,
                    periods=(epochs - lr_warmup_scheduler.warmup_epochs,),
                    restart_weights=(1,),
                    eta_min=1e-7,
                )
                scheduler = GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    multiplier=1,
                    total_epoch=lr_warmup_scheduler.warmup_epochs,
                    after_scheduler=scheduler_step,
                )
            else:
                scheduler = CosineAnnealingRestartLR(
                    optimizer=self._optimizer,
                    periods=(epochs,),
                    restart_weights=(1,),
                    eta_min=1e-7,
                )
        else:
            raise ValueError(f"unsupported lr_scheduler {lr_scheduler.name}.")

        self._scheduler = scheduler

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
        L_l1 = L.L1Loss(loss_weight=config.loss.W_L1, reduction="mean").to(device)
        L_d = L.SSIM(weight=config.loss.W_D).to(device)
        L_e = L.EdgeLoss(loss_weight=config.loss.W_E).to(device)
        L_p = L.PerceptualLoss(
            layer_weights={
                "conv1_2": 1,
                "conv2_2": 1,
                "conv3_4": 1,
                "conv4_4": 1,
            },
            perceptual_weight=1.0,
            criterion="mse",
        ).to(device)
        # Loss weights
        W_HVI = config.loss.W_HVI
        W_P = config.loss.W_P

        # 2. Train loop
        gamma = config.gamma
        grad_clip_norm = config.grad_clip_norm
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
            target_rgb = datapoint["target"]

            # Use random gamma function (enhancement curve) to improve generalization
            if gamma:
                gamma_ = random.randint(gamma.start_gamma, gamma.end_gamma) / 100.0
            else:
                gamma_ = 1.0

            # 2.2. Forward pass
            outputs = self.model(image=image, gamma=gamma_, save_debug=True)

            # 2.3. Extract outputs
            output_rgb = outputs["enhanced"]
            output_hvi = self.model.HVIT(output_rgb)
            target_hvi = self.model.HVIT(target_rgb)

            # 2.4. Calculate loss
            l_hvi = (
                L_l1(output_hvi, target_hvi)
                + L_d(output_hvi, target_hvi)
                + L_e(output_hvi, target_hvi)
                + W_P * L_p(output_hvi, target_hvi)[0]
            )
            l_rgb = (
                L_l1(output_rgb, target_rgb)
                + L_d(output_rgb, target_rgb)
                + L_e(output_rgb, target_rgb)
                + W_P * L_p(output_rgb, target_rgb)[0]
            )
            # Total loss
            loss = l_rgb + W_HVI * l_hvi

            # 2.5. Backward pass
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm, norm_type=2)
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
