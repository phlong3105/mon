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

import sys

import torch

from mon import (
    Config,
    ConfigContext,
    create_progress_bar,
    log,
    metrics,
    parse_imgsz,
    Path,
    RunMode,
    sys_ctx,
)
from mon.dataset import DataLoader

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]
if str(current_dir) not in sys.path:
    # Add the project root to sys.path so 'import zero_dce' works
    # even if you run this script from inside the folder
    sys.path.append(str(current_dir))

try:
    # Works when running as a module: python -m zero_dce.train
    from . import loss as L
    from .model import zero_dce, zero_dce_pp
except ImportError:
    # Works when running as a script: python train.py
    from model import zero_dce, zero_dce_pp
    import loss as L


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
    imgsz = parse_imgsz(config.eval_imgsz)
    scale_factor = config.model.get("scale_factor")
    if scale_factor:
         imgsz = (imgsz[0] // scale_factor, imgsz[1] // scale_factor)

    model = zero_dce(**config.model | { "weights": weights})
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

    # 7. Define losses
    L_tv = L.L_tv().to(device)
    L_spa = L.L_spa().to(device)
    L_col = L.L_col().to(device)
    L_exp = L.L_exp(16, config.loss.L_exp_mean).to(device)
    L_tv_w = config.loss.L_tv_w
    L_spa_w = config.loss.L_spa_w
    L_col_w = config.loss.L_col_w
    L_exp_w = config.loss.L_exp_w

    # 8. Define data
    train_dataloader = DataLoader.from_config(config.train_dataloader)
    val_dataloader = DataLoader.from_config(config.val_dataloader)

    # 9. Training loop
    epochs = config.epochs
    grad_clip_norm = config.grad_clip_norm
    best_loss = float("inf")
    best_psnr = 0.0

    with create_progress_bar() as pbar:
        for _ in pbar.track(
            sequence=range(epochs),
            total=epochs,
            description=f"[bright_yellow]Training"
        ):
            losses = []
            val_psnrs = []

            # 9.1. Train
            model.train()
            for i, datapoint in enumerate(train_dataloader):
                image = datapoint["image"]
                image = image.to(device)
                outputs = model(image)
                enhanced = outputs["enhanced"]
                r = outputs["r"]

                l_tv = L_tv_w * L_tv(r)
                l_spa = L_spa_w * torch.mean(L_spa(enhanced, image))
                l_col = L_col_w * torch.mean(L_col(enhanced))
                l_exp = L_exp_w * torch.mean(L_exp(enhanced))
                loss = l_tv + l_spa + l_col + l_exp

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            mean_loss = sum(losses) / len(losses)

            # 9.2. Val
            model.eval()
            for i, datapoint in enumerate(val_dataloader):
                with torch.no_grad():
                    image = datapoint["image"]
                    image = image.to(device)
                    target = datapoint["target"]
                    target = target.to(device)
                    outputs = model(image)
                    enhanced = outputs[-1]
                    mse = ((enhanced - target) ** 2).mean((2, 3))
                    psnr = (1 / mse).log10().mean() * 10
                val_psnrs.append(psnr.item())
            mean_psnr = sum(val_psnrs) / len(val_psnrs)

            # 9.3. Log
            if config.verbose:  # and ((i + 1) % display_iter) == 0:
                log(
                    f"Epoch: {(i + 1):03} | "
                    f"Loss: {mean_loss:08.6f} | "
                    f"PSNR: {mean_psnr:08.6f}"
                )

            # 9.4. Save
            torch.save(model.state_dict(), config.output_dir / "last.pt")
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), config.output_dir / "best_loss.pt")
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                torch.save(model.state_dict(), config.output_dir / "best_psnr.pt")

            # 9.4. Save debug
            # Do nothing

# endregion


# ==============================================================================
# region MAIN
# ==============================================================================

def main():
    # Load config
    config_ctx = ConfigContext.from_cli(
        root=current_dir,
        config_file="zero_dce_sice_me.yaml",
    )
    config = config_ctx.config_for(RunMode.TRAIN)
    train(config)


if __name__ == "__main__":
    main()

# endregion
