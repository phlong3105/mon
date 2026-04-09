#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for SCI and SCI++ models.
"""

from __future__ import annotations

__all__ = [
    "SCI_Finetuner",
    "SCI_PP_Finetuner",
    "SCI_PP_Trainer",
    "SCI_Trainer",
]

from typing import Any

import pyiqa
import torch
from rich.progress import Progress
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim import Adam
from typing_extensions import override

from mon.core import K, OPTIMIZERS, Path, TRAINERS
from mon.runners import Trainer
from . import loss as L
from .model import sci, sci_pp

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

# --- SCI ---

@TRAINERS.register(name="sci")
class SCI_Trainer(Trainer):
    """Trainer for SCI models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = None  # config.finetune

        model = sci(**config.model | { "weights": weights})
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

        # 1. Define losses
        criterion = L.LossFunction().to(device)

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
            # image = image.to(device)
            image = Variable(image, requires_grad=False).to(device)

            # 2.2. Forward pass
            outputs = self.model(
                data={"image": image, "inference": False},
                save_debug=True,
            )

            # 2.3. Extract outputs
            i_list = outputs["i_list"]
            x_list = outputs["x_list"]

            # 2.4. Calculate loss
            loss = torch.zeros(1, device=device)
            for j in range(len(i_list)):
                loss += criterion(x_list[j], i_list[j])

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
            outputs = self.model(
                data={"image": image, "inference": False},
                save_debug=True,
            )

            # 2.3. Extract outputs
            enhanced = outputs["r_list"][0]
            illumination = outputs["i_list"][0]
            attention = outputs["a_list"][0]

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
                    "illumination": illumination.cpu(),
                    "attention": attention.cpu(),
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
            "illumination": val_outputs["illumination"],
            "attention": val_outputs["attention"],
        }
        self._save_image(epoch, debug_image, dirname=K.PRED_DIR, stem="debug", column_first=True)


class SCI_Finetuner(Trainer):
    """Finetuner for SCI models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = sci(**config.model | { "weights": weights})
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
        epoch = 100  # Overwrite epoch for finetuning

        # 1. Define losses
        criterion = L.LossFunction().to(device)

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
            # image = Variable(image, requires_grad=False).to(device)

            # 2.2. Forward pass
            outputs = self.model(
                data={"image": image, "inference": True},
                save_debug=True,
            )

            # 2.3. Extract outputs
            illumination = outputs["illumination"]

            # 2.4. Calculate loss
            loss = criterion(image, illumination)

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
            outputs = self.model(
                data={"image": image, "inference": True},
                save_debug=True,
            )

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            illumination = outputs["illumination"]

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
                    "illumination": illumination.cpu(),
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
            "illumination": val_outputs["illumination"],
        }
        self._save_image(epoch, debug_image, dirname=K.PRED_DIR, stem="debug", column_first=True)


# --- SCI++ ---

@TRAINERS.register(name="sci++")
class SCI_PP_Trainer(Trainer):
    """Trainer for SCI++ models."""

    # --- Lifecycle & Initialization ---
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_step = 0

    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = None  # config.finetune

        model = sci_pp(**config.model | { "weights": weights})
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _init_optimizer(self):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        config = self.config

        self._optimizer = OPTIMIZERS.build(params=self.model.ha.parameters(), **config.optimizer)
        self._optimizer_b = Adam(
            list(self.model.hb.parameters()) + list(self.model.calibrate.parameters()),
            **config.optimizer
        )
        self._scheduler = None

    # --- Control ---
    @override
    def train(self):
        self.total_step = 0
        super().train()

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

        # 1. Define losses
        criterion = L.LossFunction().to(device)

        # 2. Train loop
        grad_clip_norm = config.grad_clip_norm
        train_outputs = {}
        losses = []

        task = pbar.add_task(
            f"[bright_yellow]Train Epoch {epoch+1:03}",
            total=len(self.train_dataloader)
        )
        for i, datapoint in enumerate(self.train_dataloader):
            self.total_step += 1

            # 2.1. Prepare inputs
            image = datapoint["image"]
            # image = image.to(device)
            image = Variable(image, requires_grad=False).to(device)

            # 2.2. Forward pass
            outputs = self.model(
                data={"image": image, "inference": False},
                save_debug=True,
            )

            # 2.3. Extract outputs
            i_list = outputs["i_list"]
            x_list = outputs["x_list"]

            # 2.4. Calculate loss
            loss_1 = criterion(x_list[0], i_list[0])
            loss_2 = F.l1_loss(i_list[0], i_list[1]) + 0.1 * criterion(x_list[0], i_list[1])
            loss_3 = F.l1_loss(i_list[0], i_list[2]) + 0.1 * criterion(x_list[0], i_list[2])

            # 2.5. Backward pass
            if self.total_step % 10 < 7:
                for param in self.model.ha.parameters():
                    param.requires_grad = True
                for param in self.model.hb.parameters():
                    param.requires_grad = False
                for param in self.model.calibrate.parameters():
                    param.requires_grad = False
                self._optimizer.zero_grad()
                loss = loss_1
                loss.backward()
                self._optimizer.step()
            else:
                for param in self.model.ha.parameters():
                    param.requires_grad = False
                for param in self.model.hb.parameters():
                    param.requires_grad = True
                for param in self.model.calibrate.parameters():
                    param.requires_grad = True
                self._optimizer_b.zero_grad()
                loss = (loss_2 + loss_3)
                loss.backward()
                self._optimizer_b.step()
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
            outputs = self.model(
                data={"image": image, "inference": False},
                save_debug=True,
            )

            # 2.3. Extract outputs
            enhanced = outputs["r_list"][0]
            illumination = outputs["i_list"][0]
            attention = outputs["a_list"][0]

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
                    "illumination": illumination.cpu(),
                    "attention": attention.cpu(),
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
            "illumination": val_outputs["illumination"],
            "attention": val_outputs["attention"],
        }
        self._save_image(epoch, debug_image, dirname=K.PRED_DIR, stem="debug", column_first=True)


class SCI_PP_Finetuner(SCI_Finetuner):
    """Finetuner for SCI++ models."""
    pass

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
