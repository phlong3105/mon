#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for Zero-DCE and Zero-DCE++ models.
"""

from __future__ import annotations

__all__ = [
    "RetinexNet_Trainer",
]

from typing import Any

import pyiqa
import torch
from rich.progress import Progress
from tensordict import TensorDict
from torch import nn
from typing_extensions import Literal, override

from mon.core import (
    create_progress_bar,
    K,
    OPTIMIZERS,
    Path,
    SCHEDULERS,
    Split,
    sys_ctx,
    TRAINERS,
)
from mon.dataset import build_dataloader, transform as T
from mon.ops import normalize_minmax
from mon.runners import Trainer
from .model import retinexnet
from .utils import smooth

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region TRAINER
# ==============================================================================

@TRAINERS.register(name="retinexnet")
class RetinexNet_Trainer(Trainer):
    """Trainer for RetinexNet models."""

    # --- Lifecycle & Initialization ---
    @override
    def _init_model(self):
        """Initialize ``self._model`` attribute."""
        config = self.config
        device = self.device
        weights = config.finetune

        model = retinexnet(**config.model | {"weights": weights})
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _setup(self):
        """Setup the runner ready for training."""
        config = self.config

        # Setup environment
        config.output_dir.mkdir(exist_ok=True, parents=True)
        config.config_file.copy_to(config.output_dir / config.config_file.name)
        sys_ctx.set_random_seed(config.seed)

        # Define model
        self._init_model()

        # Define optimizer & scheduler
        # self._init_optimizer()

        # Define data
        self._init_train_dataloader()
        self._init_val_dataloader()

        # Define loggers
        self._init_loggers()

    @override
    def _init_optimizer(self, phase: Literal["decom", "enhance"]):
        """Initialize ``self._optimizer`` and ``self._scheduler`` attributes."""
        config = self.config

        if phase == "decom":
            self._optimizer = OPTIMIZERS.build(params=self.model.decom_net.parameters(), **config.optimizer)
            self._scheduler = SCHEDULERS.build(optimizer=self._optimizer, **config.lr_scheduler)
        elif phase == "enhance":
            self._optimizer = OPTIMIZERS.build(params=self.model.enhance_net.parameters(), **config.optimizer)
            self._scheduler = SCHEDULERS.build(optimizer=self._optimizer, **config.lr_scheduler)
        else:
            self._optimizer = OPTIMIZERS.build(params=self.model.parameters(), **config.optimizer | {"lr": 0.0001})
            self._scheduler = None

    @override
    def _init_train_dataloader(self):
        """Initialize ``self._train_dataloader`` attribute."""
        config = self.config
        dataloader = self.config.train_dataloader

        transforms = T.Compose([
            T.RandomCrop(height=96, width=96, p=1.0),
            T.RandomOrder([
                T.Compose([]),                                                                    # mode 0: original
                T.Compose([T.VerticalFlip(p=1.0)]),                                               # mode 1: flip ud
                T.Compose([T.Transpose(p=1.0), T.HorizontalFlip(p=1.0)]),                         # mode 2: rot CCW 90
                T.Compose([T.Transpose(p=1.0), T.HorizontalFlip(p=1.0), T.VerticalFlip(p=1.0)]),  # mode 3: rot CCW 90 + flip ud
                T.Compose([T.HorizontalFlip(p=1.0), T.VerticalFlip(p=1.0)]),                      # mode 4: rot 180
                T.Compose([T.HorizontalFlip(p=1.0)]),                                             # mode 5: rot 180 + flip ud
                T.Compose([T.Transpose(p=1.0), T.VerticalFlip(p=1.0)]),                           # mode 6: rot CCW 270
                T.Compose([T.Transpose(p=1.0)]),                                                  # mode 7: rot CCW 270 + flip ud
            ], n=1, p=1.0),
            T.Normalize(normalization="min_max"),
            T.ToTensorV2(transpose_mask=True),
        ])

        self._train_dataloader = build_dataloader(
            src=dataloader,
            dataset_dir=config.data_dir,
            transforms=transforms,
            split=Split.TRAIN,
        )[1]

    # --- Control ---
    @override
    def train(self):
        """Train the model."""
        config = self.config

        # 1. Setup
        self._setup()
        # Validate that all necessary components are initialized
        if self._model is None:
            raise RuntimeError(f"model is not initialized.")
        if self._train_dataloader is None:
            raise RuntimeError(f"train_dataloader is not initialized.")

        # 2. Summarize the current run
        if config.verbose:
            self._log_summary()

        # 3. Run benchmark
        if config.benchmark:
            self._benchmark()

        # 4. Main loop (DecomNet)
        self._init_optimizer(phase="decom")
        epochs = config.epochs
        with create_progress_bar() as pbar:
            for epoch in pbar.track(
                sequence=range(epochs),
                total=epochs,
                description=f"[bright_yellow]Training DecomNet"
            ):
                # 4.1. Train epoch
                self.model.decom_net.train()
                self.model.enhance_net.eval()
                train_outputs = self._train_epoch(epoch=epoch, phase="decom", pbar=pbar)
                if "loss" not in train_outputs:
                    raise ValueError(
                        f"expected loss to be returned from 'self._train_epoch()', "
                        f"got {list(train_outputs.keys())}."
                    )

                # 4.2. Scheduler Step
                if self._scheduler is not None:
                    self._scheduler.step()

                # 4.3. Log
                if self.verbose:
                    self._log(epoch, train_outputs=train_outputs, val_outputs={})

                # 4.4. Save
                if self.save:
                    self._save(epoch, train_outputs=train_outputs, val_outputs={})

        # 5. Main loop (EnhanceNet)
        current_epoch = epochs
        end_epoch = epochs * 2
        self._init_optimizer(phase="enhance")
        with create_progress_bar() as pbar:
            for epoch in pbar.track(
                sequence=range(current_epoch, end_epoch),
                total=(end_epoch - current_epoch),
                description=f"[bright_yellow]Training EnhanceNet"
            ):
                # 5.1. Train epoch
                self.model.decom_net.eval()
                self.model.enhance_net.train()
                train_outputs = self._train_epoch(epoch=epoch, phase="enhance", pbar=pbar)
                if "loss" not in train_outputs:
                    raise ValueError(
                        f"expected loss to be returned from 'self._train_epoch()', "
                        f"got {list(train_outputs.keys())}."
                    )

                # 5.2. Val epoch
                val_outputs = {}
                if self._val_dataloader is not None:
                    self.model.eval()
                    val_outputs = self._val_epoch(epoch=epoch, pbar=pbar)

                # 5.3. Scheduler Step
                if self._scheduler is not None:
                    self._scheduler.step()

                # 5.4. Log
                if self.verbose:
                    self._log(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

                # 5.5. Save
                if self.save:
                    self._save(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

                # 5.6. Save debug
                if self.save_debug:
                    self._save_debug(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

    # --- Training ---
    @override
    def _train_epoch(
        self,
        epoch: int,
        phase: Literal["decom", "enhance"],
        pbar: Progress
    ) -> dict[str, Any]:
        """Train an epoch.

        Args:
            epoch (int): The current epoch number.
            phase (str): The current training phase. Either "decom" or "enhance".
            pbar (Progress): The progress bar object.

        Returns:
            dict[str, Any]: A dictionary containing the training loss and other
                results for the epoch.
        """
        device = self.device

        # 1. Define losses
        L = nn.L1Loss().to(device)

        # 2. Train loop (DecomNet)
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
            R_low, L_low = self.model.decom_net(image)
            R_high, L_high = self.model.decom_net(target)
            L_delta = self.model.enhance_net(R_low, L_low)

            # 2.3. Extract outputs
            L_low_3 = torch.cat((L_low, L_low, L_low), dim=1)
            L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)
            L_delta_3 = torch.cat((L_delta, L_delta, L_delta), dim=1)

            # 2.4. Calculate loss
            if phase == "decom":
                # DecomNet loss
                l_recon_low = L(R_low * L_low_3, image)
                l_recon_high = L(R_high * L_high_3, target)
                l_recon_mutal_low = L(R_high * L_low_3, image)
                l_recon_mutal_high = L(R_low * L_high_3, target)
                l_equal_R = L(R_low, R_high.detach())
                l_smooth_low = smooth(R_low, L_low)
                l_smooth_high = smooth(R_high, L_high)
                loss = (
                    l_recon_low + l_recon_high
                    + 0.001 * l_recon_mutal_low
                    + 0.001 * l_recon_mutal_high
                    + 0.1 * l_smooth_low
                    + 0.1 * l_smooth_high
                    + 0.01 * l_equal_R
                )
            else:
                # EnhanceNet loss
                l_relight = L(R_low * L_delta_3, target)
                l_smooth_delta = smooth(R_low, L_delta)
                loss = l_relight + 3 * l_smooth_delta

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
            outputs = self.model(data=datapoint, save_debug=True)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            enhanced = torch.clamp(enhanced, 0.0, 1.0)
            R = outputs["R"]
            L = outputs["L"]
            L_delta = outputs["L_delta"]

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
                    "reflectance": R,
                    "illumination": L,
                    "illumination_delta": L_delta,
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
        debug_images = {
            "image": val_outputs["image"],
            "target": val_outputs["target"],
            "enhanced": val_outputs["enhanced"],
            "reflectance": val_outputs["reflectance"],
            "illumination": normalize_minmax(val_outputs["illumination"]),
            "illumination_delta": normalize_minmax(val_outputs["illumination_delta"]),
        }
        self._save_image(
            epoch=epoch,
            outputs=debug_images,
            dirname=K.DEBUG_DIR,
            stem="",
            column_first=True,
            show_info=True,
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
