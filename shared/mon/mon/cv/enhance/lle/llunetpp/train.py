#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LL-UNet++ model training pipeline for low-light image enhancement.

References:
    - Paper: "LL-UNet++:UNet++ Based Nested Skip Connections Network for Low-Light
      Image Enhancement," TCI 2024.
    - Code: https://github.com/xiwang-online/LLUnetPlusPlus
"""

from collections import OrderedDict

import box
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import llunetpp
import mon

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Train -----
def train_epoch(train_dataloader, model, criterion, optimizer, device):
    loss_meters = llunetpp.AverageMeter()
    model.train()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(train_dataloader),
            total       = len(train_dataloader),
            description = f"[bright_yellow]Training"
        ):
            input  = datapoint["image"].to(device)
            target = datapoint["ref"].to(device)
            output = model(input)
            loss   = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meters.update(loss.item(), input.size(0))
            # mon.log(loss_meters.avg)
    return loss_meters.avg


def val_epoch(val_dataloader, model, criterion, device):
    loss_meters = llunetpp.AverageMeter()
    psnr_meters = mon.nn.PeakSignalNoiseRatio().to(device)
    ssim_meters = mon.nn.StructuralSimilarityIndexMeasure().to(device)
    model.eval()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(val_dataloader),
            total       = len(val_dataloader),
            description = f"[bright_yellow]Validating"
        ):
            input  = datapoint["image"].to(device)
            target = datapoint["ref"].to(device)
            output = model(input)
            loss   = criterion(output, target)
            loss_meters.update(loss.item(), input.size(0))
            psnr_meters.update(output, target)
            ssim_meters.update(output, target)
            # mon.log(loss_meters.avg)
    return loss_meters.avg, psnr_meters.compute(), ssim_meters.compute()


def train(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)
    
    # Device
    device = mon.create_device(args.device)
    cudnn.benchmark = True
    
    # Seed
    mon.set_random_seed(args.seed)
    
    # Pretrained
    pretrained = args.tuning
    if args.resume and args.resume.is_weights_file(exist=True):
        pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        mon.log(f"Pretrained: {None}, training from scratch.")

    # Model
    model = llunetpp.LLUnetPP(weights=pretrained)
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = mon.nn.Adam(model.parameters(), **args.optimizer)
    scheduler = mon.nn.ExponentialLR(optimizer, 0.99)
    
    # Loss
    criterion = llunetpp.Loss(*args.loss.loss_weights).to(device)
    
    # Log
    writer = SummaryWriter(log_dir=str(args.save_dir))
    log    = OrderedDict([
        ("epoch"     , []),
        ("train/loss", []),
        ("val/loss"  , []),
        ("val/psnr"  , []),
        ("val/ssim"  , []),
    ])
    best_loss = 1000
    best_psnr = 0
    best_ssim = 0
    
    # Data I/O
    args["train_dataloader"]["dataset"]["root"] = mon.data.parse_data_dir(args.root)
    args["val_dataloader"]["dataset"]["root"]   = mon.data.parse_data_dir(args.root)
    train_dataloader = mon.data.DataLoader(**args.train_dataloader)
    val_dataloader   = mon.data.DataLoader(**args.val_dataloader)

    # Train
    for epoch in range(args.epochs):
        train_loss  = train_epoch(train_dataloader, model, criterion, optimizer, device)
        val_results = val_epoch(val_dataloader,     model, criterion, device)
        val_loss    = float(val_results[0])
        val_psnr    = float(val_results[1].cpu().detach().numpy())
        val_ssim    = float(val_results[2].cpu().detach().numpy())
        scheduler.step()
        mon.log(
            "Epoch [%d/%d] train/loss %.4f - val/loss %.4f - val/psnr %.4f - val/ssim %.4f\n"
            % (epoch, args.epochs, train_loss, val_loss, val_psnr, val_ssim)
        )
        
        # Log
        log["epoch"].append(epoch)
        log["train/loss"].append(train_loss)
        log["val/loss"].append(val_loss)
        log["val/psnr"].append(val_psnr)
        log["val/ssim"].append(val_ssim)
        pd.DataFrame(log).to_csv(str(args.save_dir / "log.csv"))
        writer.add_scalars(
            "train",
            {"train/loss": train_loss},
            epoch,
        )
        writer.add_scalars(
            "val",
            {
                "val/loss": val_loss,
                "val/psnr": val_psnr,
                "val/ssim": val_ssim,
            },
            epoch,
        )
        
        # Save
        if val_loss < best_loss:
            torch.save(model.state_dict(), str(args.save_dir / "best.pt"))
            best_loss = val_loss
        if val_psnr > best_psnr:
            torch.save(model.state_dict(), str(args.save_dir / "best_psnr.pt"))
            best_psnr = val_psnr
        if val_ssim > best_ssim:
            torch.save(model.state_dict(), str(args.save_dir / "best_ssim.pt"))
            best_ssim = val_ssim
        torch.save(model.state_dict(), str(args.save_dir / "last.pt"))
        torch.cuda.empty_cache()
   
    writer.close()
    
    
# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(root=root_dir, model_root=root_dir)
    train(args)


if __name__ == "__main__":
    main()
