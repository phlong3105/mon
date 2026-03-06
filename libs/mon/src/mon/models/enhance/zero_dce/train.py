#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training Script.

This script provides a CLI for running Zero-DCE training on a given dataset.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE

    - Paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep Curve
      Estimation," IEEE TPAMI 2022.
    - Code: https://github.com/Li-Chongyi/Zero-DCE_extension
"""

from __future__ import annotations

__all__ = []

import numpy as np
import pyiqa
import torch
from rich.progress import Progress
from torch import nn

from mon import (
    Config,
    ConfigContext,
    create_progress_bar,
    log,
    metrics,
    MODELS,
    pascalize,
    Path,
    RunMode,
    Size,
    sys_ctx,
    Task,
    to_image_array,
)
from mon.dataset import DataLoader
from mon.models.enhance.zero_dce import loss as L
from mon.ops import draw_info, write_image

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region CONTROL
# ==============================================================================

def train(config: Config):
    # 1. Summarize the current run
    if config.verbose:
        config.log_summary()

    # 2. Setup environment
    device = config.device
    sys_ctx.set_random_seed(config.seed)

    # 3. Resolve pre-trained weights
    weights = config.finetune

    # 4. Define model
    imgsz = Size.from_value(config.eval_imgsz)
    scale_factor = config.model.get("scale_factor")
    if scale_factor:
        imgsz = Size(
            height=imgsz.height // scale_factor,
            width=imgsz.width // scale_factor,
        )

    model = MODELS.build(**config.model | { "weights": weights})
    model = model.to(device)
    model.train()

    # 5. Run benchmark
    if config.benchmark:
        metrics.benchmark(model, imgsz=imgsz)

    # 6. Define optimizer & scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )

    # 7. Define data
    train_dataloader = DataLoader.from_config(config.train_dataloader)
    val_dataloader = DataLoader.from_config(config.val_dataloader)

    # 8. Training loop
    epochs = config.epochs
    best_loss = float("inf")
    best_psnr = 0.0
    best_ssim = 0.0
    best_ssimc = 0.0
    config.output_dir.mkdir(exist_ok=True, parents=True)

    with create_progress_bar() as pbar:
        for i in pbar.track(
            sequence=range(epochs),
            total=epochs,
            description=f"[bright_yellow]Training"
        ):
            # 8.1. Train epoch
            train_outputs = train_epoch(i, config, model, optimizer, train_dataloader, pbar)
            loss = train_outputs.pop("loss")

            # 8.2. Val epoch
            val_outputs = val_epoch(i, config, model, val_dataloader, pbar)
            psnr = val_outputs.pop("psnr")
            ssim = val_outputs.pop("ssim")
            ssimc = val_outputs.pop("ssimc")

            # 8.3. Log
            if config.verbose:
                log(
                    f"Epoch: {(i + 1):03} | "
                    f"Loss: {loss:08.6f} | "
                    f"PSNR: {psnr:08.6f} | "
                    f"SSIM: {ssim:08.6f} | "
                    f"SSIM-C: {ssimc:08.6f}"
                )

            # 8.4. Save
            torch.save(model.state_dict(), config.output_dir / "last.pt")
            if loss < best_loss:
                best_loss = loss
                torch.save(model.state_dict(), config.output_dir / "best_loss.pt")
            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), config.output_dir / "best_psnr.pt")
            if ssim > best_ssim:
                best_ssim = ssim
                torch.save(model.state_dict(), config.output_dir / "best_ssim.pt")
            if ssimc > best_ssimc:
                best_ssimc = ssimc
                torch.save(model.state_dict(), config.output_dir / "best_ssimc.pt")

            # 8.5. Save debug
            if config.save_debug:
                debug = []
                for k, v in val_outputs.items():
                    image = to_image_array(torch.cat(list(v), dim=2).unsqueeze(0))
                    image = draw_info(image, [f"{pascalize(k)}"])
                    debug.append(image)
                debug = np.vstack(debug)
                save_path = config.output_dir / "debug" / f"debug_epoch_{i+1:03}.png"
                write_image(debug, save_path)


def train_epoch(
    epoch: int,
    config: Config,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    pbar: Progress
) -> dict:
    device = config.device

    # 1. Define losses
    L_tv = L.L_tv().to(device)
    L_spa = L.L_spa().to(device)
    L_col = L.L_col().to(device)
    L_exp = L.L_exp(16, config.loss.L_exp_mean).to(device)
    # Loss weights
    L_tv_w = config.loss.L_tv_w
    L_spa_w = config.loss.L_spa_w
    L_col_w = config.loss.L_col_w
    L_exp_w = config.loss.L_exp_w

    # 2. Train loop
    model.train()
    grad_clip_norm = config.grad_clip_norm
    train_outputs = {}
    losses = []

    task = pbar.add_task(
        f"[bright_yellow]Train Epoch {epoch+1:03}",
        total=len(train_dataloader)
    )
    for j, datapoint in enumerate(train_dataloader):
        image = datapoint["image"]
        image = image.to(device)

        # 2.1. Forward pass
        outputs = model(image)
        enhanced = outputs["enhanced"]
        r = outputs["r"]

        # 2.2. Calculate loss
        l_tv = L_tv_w * L_tv(r)
        l_spa = L_spa_w * torch.mean(L_spa(enhanced, image))
        l_col = L_col_w * torch.mean(L_col(enhanced))
        l_exp = L_exp_w * torch.mean(L_exp(enhanced))
        loss = l_tv + l_spa + l_col + l_exp

        # 2.3. Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        losses.append(loss.item())

        pbar.update(task, advance=1)
    pbar.remove_task(task)

    # 3. Output
    train_outputs |= {
        "loss": sum(losses) / len(losses),
    }
    return train_outputs


def val_epoch(
    epoch: int,
    config: Config,
    model: nn.Module,
    val_dataloader: DataLoader,
    pbar: Progress
) -> dict:
    device = config.device

    # 1. Define metrics
    psnr_metric = pyiqa.create_metric("psnr", device=device)
    ssim_metric = pyiqa.create_metric("ssim", device=device)
    ssimc_metric = pyiqa.create_metric("ssimc", device=device)

    # 2. Val loop
    model.eval()
    val_outputs = {}
    psnrs = []
    ssims = []
    ssimcs = []

    task = pbar.add_task(
        f"[bright_yellow]Val Epoch {epoch+1:03}",
        total=len(val_dataloader)
    )
    for j, datapoint in enumerate(val_dataloader):
        with torch.no_grad():
            image = datapoint["image"]
            target = datapoint["target"]
            image = image.to(device)
            target = target.to(device)

            # 2.1. Forward pass
            outputs = model(image)
            enhanced = outputs["enhanced"]

            # 2.2. Measure metrics
            psnr = torch.mean(psnr_metric(enhanced, target))
            ssim = torch.mean(ssim_metric(enhanced, target))
            ssimc = torch.mean(ssimc_metric(enhanced, target))
            psnrs.append(psnr)
            ssims.append(ssim)
            ssimcs.append(ssimc)

            # 2.3. Debug outputs
            if j == 0:
                val_outputs |= {
                    "image": image.cpu(),
                    "enhanced": enhanced.cpu(),
                }

            pbar.update(task, advance=1)
    pbar.remove_task(task)

    # 3. Output
    val_outputs |= {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "ssimc": sum(ssimcs) / len(ssimcs),
    }
    return val_outputs

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    # Load config
    config_ctx = ConfigContext.from_cli(
        root=current_dir,
        config_file="zero_dce_sice_me.yaml",
        task=Task.ENHANCE,
        mode=RunMode.TRAIN,
        arch="zero_dce",
        model="zero_dce",
        save=True,
        exist_ok=True,
        verbose=True,
    )
    config = config_ctx.config_for(RunMode.TRAIN)
    train(config)


if __name__ == "__main__":
    main()

# endregion
