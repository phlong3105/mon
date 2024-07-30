#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    `<https://github.com/xiwang-online/LLUnetPlusPlus>`__
"""

from __future__ import annotations

import argparse
from collections import OrderedDict

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import mon
from average_meter import AverageMeter
from loss import Loss
from model import NestedUNet
from mon import albumentation as A

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def train_epoch(train_loader, model, criterion, optimizer, device):
    loss_meters = AverageMeter()
    model.train()
    with mon.get_progress_bar() as pbar:
        for input, target, meta in pbar.track(
            sequence    = train_loader,
            total       = len(train_loader),
            description = f"[bright_yellow] Training"
        ):
            input  = input.to(device)
            target = target.to(device)
            output = model(input)
            loss   = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meters.update(loss.item(), input.size(0))
            # console.log(loss_meters.avg)
    return loss_meters.avg


def val_epoch(val_loader, model, criterion, device):
    loss_meters = AverageMeter()
    psnr_meters = mon.PeakSignalNoiseRatio().to(device)
    ssim_meters = mon.StructuralSimilarityIndexMeasure().to(device)
    model.eval()
    with mon.get_progress_bar() as pbar:
        for input, target, meta in pbar.track(
            sequence    = val_loader,
            total       = len(val_loader),
            description = f"[bright_yellow] Validating"
        ):
            input  = input.to(device)
            target = target.to(device)
            output = model(input)
            loss   = criterion(output, target)
            loss_meters.update(loss.item(), input.size(0))
            psnr_meters.update(output, target)
            ssim_meters.update(output, target)
            # console.log(loss_meters.avg)
    return loss_meters.avg, psnr_meters.compute(), ssim_meters.compute()


def train(args: argparse.Namespace):
    # General config
    save_dir = mon.Path(args.save_dir)
    weights  = args.weights
    device   = mon.set_device(args.device)
    imgsz    = args.imgsz
    epochs   = args.epochs
    verbose  = args.verbose
    
    # Directory
    weights_dir = save_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    cudnn.benchmark = True
    
    # Model
    model = NestedUNet().to(device)
    model.train()
    
    # Loss
    criterion = Loss(*args.loss_weights).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    
    # Data I/O
    data_args = {
        "name"      : args.data,
        "root"      : mon.DATA_DIR / "llie",
        "transform" : A.Compose(transforms=[
            A.Resize(width=imgsz, height=imgsz),
        ]),
        "to_tensor" : True,
        "cache_data": False,
        "batch_size": args.batch_size,
        "devices"   : device,
        "shuffle"   : True,
        "verbose"   : verbose,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=data_args)
    datamodule.prepare_data()
    datamodule.setup(stage="training")
    train_dataloader = datamodule.train_dataloader
    val_dataloader   = datamodule.val_dataloader
    
    # Logging
    writer = SummaryWriter(log_dir=str(save_dir))
    log    = OrderedDict([
        ("epoch"     , []),
        ("lr"        , []),
        ("train/loss", []),
        ("val/loss"  , []),
        ("val/psnr"  , []),
        ("val/ssim"  , []),
    ])
    best_loss = 1000
    best_psnr = 0
    best_ssim = 0
    
    # Training
    for epoch in range(epochs):
        train_loss  = train_epoch(train_dataloader, model, criterion, optimizer, device)
        val_results = val_epoch(val_dataloader, model, criterion, device)
        val_loss    = float(val_results[0])
        val_psnr    = float(val_results[1].cpu().detach().numpy())
        val_ssim    = float(val_results[2].cpu().detach().numpy())
        scheduler.step()
        console.log(
            "Epoch [%d/%d] train/loss %.4f - val/loss %.4f - val/psnr %.4f - val/ssim %.4f\n"
            % (epoch, epochs, train_loss, val_loss, val_psnr, val_ssim)
        )
        
        # Log
        log["epoch"].append(epoch)
        log["lr"].append(args.lr)
        log["train/loss"].append(train_loss)
        log["val/loss"].append(val_loss)
        log["val/psnr"].append(val_psnr)
        log["val/ssim"].append(val_ssim)
        pd.DataFrame(log).to_csv(str(save_dir / "log.csv"))
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
            torch.save(model.state_dict(), str(weights_dir / "best.pt"))
            best_loss = val_loss
        if val_psnr > best_psnr:
            torch.save(model.state_dict(), str(weights_dir / "best_psnr.pt"))
            best_psnr = val_psnr
        if val_ssim > best_ssim:
            torch.save(model.state_dict(), str(weights_dir / "best_ssim.pt"))
            best_ssim = val_ssim
        torch.save(model.state_dict(), str(weights_dir / "last.pt"))
        torch.cuda.empty_cache()
   
    writer.close()
    
# endregion

    
# region Main

def main() -> str:
    args = mon.parse_train_args(model_root=_current_dir)
    train(args)


if __name__ == "__main__":
    main()

# endregion
