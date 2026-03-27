#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Runners.

This module provides training runner classes for Zero-DCE and Zero-DCE++ models.
"""

from __future__ import annotations

__all__ = [
    "RetinexNet_Trainer",
]

import pyiqa
import torch
from rich.progress import Progress
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

        model = retinexnet(**config.model | { "weights": weights})
        model = model.to(device)
        model.train()
        self._model = model

    @override
    def _init_optimizer(self, phase: Literal["decom", "enhance", "whole"]):
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

        # 1. Summarize the current run
        if config.verbose:
            config.log_summary()

        # 2. Setup environment
        config.output_dir.mkdir(exist_ok=True, parents=True)
        config.config_file.copy_to(config.output_dir / config.config_file.name)

        sys_ctx.set_random_seed(config.seed)
        epochs = config.epochs

        # 3. Define model
        self._init_model()
        if self.model is None:
            raise RuntimeError(f"'model' is not initialized.")

        # 4. Define optimizer & scheduler
        # self._init_optimizer()
        # if self.optimizer is None:
        #     raise RuntimeError(f"'optimizer' is not initialized.")

        # 5. Define data
        self._init_train_dataloader()
        self._init_val_dataloader()
        if self.train_dataloader is None:
            raise RuntimeError(f"'train_dataloader' is not initialized.")

        # 6. Define loggers
        self._init_loggers()

        # 7. Run benchmark
        self.benchmark()

        # 8. Main loop (DecomNet)
        self._init_optimizer(phase="decom")
        with create_progress_bar() as pbar:
            for epoch in pbar.track(
                sequence=range(epochs),
                total=epochs,
                description=f"[bright_yellow]Training DecomNet"
            ):
                # 8.1. Train epoch
                self.model.decom_net.train()
                self.model.enhance_net.eval()
                train_outputs = self._train_epoch(epoch=epoch, phase="decom", pbar=pbar)
                if "loss" not in train_outputs:
                    raise ValueError(
                        f"Expected 'loss' from 'self._train_epoch()', "
                        f"but got {train_outputs.keys()}."
                    )

                # 8.2. Scheduler Step
                if self.scheduler is not None:
                    self.scheduler.step()

                # 8.3. Log
                if self.verbose:
                    self._log(epoch, train_outputs=train_outputs, val_outputs={})

                # 8.4. Save
                if self.save:
                    self._save(epoch, train_outputs=train_outputs, val_outputs={})

        # 9. Main loop (EnhanceNet)
        current_epoch = epochs
        end_epoch = epochs * 2
        self._init_optimizer(phase="enhance")
        with create_progress_bar() as pbar:
            for epoch in pbar.track(
                sequence=range(current_epoch, end_epoch),
                total=(end_epoch - current_epoch),
                description=f"[bright_yellow]Training EnhanceNet"
            ):
                # 9.1. Train epoch
                self.model.decom_net.eval()
                self.model.enhance_net.train()
                train_outputs = self._train_epoch(epoch=epoch, phase="enhance", pbar=pbar)
                if "loss" not in train_outputs:
                    raise ValueError(
                        f"Expected 'loss' from 'self._train_epoch()', "
                        f"but got {train_outputs.keys()}."
                    )

                # 9.2. Val epoch
                val_outputs = {}
                if self.val_dataloader is not None:
                    self.model.eval()
                    val_outputs = self._val_epoch(epoch=epoch, pbar=pbar)

                # 9.3. Scheduler Step
                if self.scheduler is not None:
                    self.scheduler.step()

                # 9.4. Log
                if self.verbose:
                    self._log(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

                # 9.5. Save
                if self.save:
                    self._save(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

                # 9.6. Save debug
                if self.save_debug:
                    self._save_debug(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

        # 10. Main loop (Whole)
        '''
        current_epoch = epochs * 2
        end_epoch = epochs * 2 + 5
        self._init_optimizer(phase="whole")
        with create_progress_bar() as pbar:
            for epoch in pbar.track(
                sequence=range(current_epoch, end_epoch),
                total=(end_epoch - current_epoch),
                description=f"[bright_yellow]Training Whole Model"
            ):
                # 10.1. Train epoch
                self.model.decom_net.eval()
                self.model.enhance_net.train()
                train_outputs = self._train_epoch(epoch=epoch, phase="whole", pbar=pbar)
                if "loss" not in train_outputs:
                    raise ValueError(
                        f"Expected 'loss' from 'self._train_epoch()', "
                        f"but got {train_outputs.keys()}."
                    )

                # 10.2. Val epoch
                val_outputs = {}
                if self.val_dataloader is not None:
                    self.model.eval()
                    val_outputs = self._val_epoch(epoch=epoch, pbar=pbar)

                # 10.3. Scheduler Step
                if self.scheduler is not None:
                    self.scheduler.step()

                # 10.4. Log
                if self.verbose:
                    self._log(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

                # 10.5. Save
                if self.save:
                    self._save(epoch, train_outputs=train_outputs, val_outputs=val_outputs)

                # 10.6. Save debug
                if self.save_debug:
                    self._save_debug(epoch, train_outputs=train_outputs, val_outputs=val_outputs)
        '''

    # --- Training ---
    @override
    def _train_epoch(
        self,
        epoch: int,
        phase: Literal["decom", "enhance", "whole"],
        pbar: Progress
    ) -> dict:
        """Train an epoch.

        Args:
            epoch (int): The current epoch number.
            phase (str): The current training phase. Either "decom", "enhance",
                or "whole".
            pbar (Progress): The progress bar object.

        Returns:
            dict: A dictionary containing the training loss and other results
                for the epoch.
        """
        config = self.config
        device = self.device

        # 1. Define losses
        L = nn.L1Loss().to(device)

        # 2. Train loop (DecomNet)
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
            target = datapoint["target"]
            target = target.to(device)

            # 2.2. Forward pass
            if phase == "decom":
                outputs_low = self.model(image=image, decom=True)
            else:
                outputs_low = self.model(image=image, decom=False)
            outputs_high = self.model(image=image, decom=True)

            # 2.3. Extract outputs
            R_low = outputs_low["R"]
            L_low = outputs_low["L"]
            L_low_3 = torch.cat((L_low, L_low, L_low), dim=1)
            R_high = outputs_high["R"]
            L_high = outputs_high["L"]
            L_high_3 = torch.cat((L_high, L_high, L_high), dim=1)

            # 2.4. Calculate loss
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
            # EnhanceNet loss
            if phase != "decom":
                L_delta = outputs_low["L_delta"]
                L_delta_3 = torch.cat((L_delta, L_delta, L_delta), dim=1)
                l_relight = L(R_low * L_delta_3, target)
                l_smooth_delta = smooth(R_low, L_delta)
                loss += l_relight + 3 * l_smooth_delta

            # 2.5. Backward pass
            self.optimizer.zero_grad()
            loss.backward()
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
            outputs = self.model(image=image, decom=False)

            # 2.3. Extract outputs
            enhanced = outputs["enhanced"]
            R = outputs["R"]
            L = outputs["L"]
            L_delta = outputs["L_delta"]

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
                    "reflectance": R.cpu(),
                    "illumination": L.cpu(),
                    "illumination_delta": L_delta.cpu(),
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
            "reflectance": val_outputs["reflectance"],
            "illumination": normalize_minmax(val_outputs["illumination"]),
            "illumination_delta": normalize_minmax(val_outputs["illumination_delta"]),
        }
        self._save_image(epoch, debug_image, dirname=K.PRED_DIR, stem="debug", column_first=True)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion
