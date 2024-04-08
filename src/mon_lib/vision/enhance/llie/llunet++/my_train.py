#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    `<https://github.com/xiwang-online/LLUnetPlusPlus>`__
"""

from __future__ import annotations

import argparse
import os
import socket
from collections import OrderedDict

import click
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import mon
from averageMeter import AverageMeter
from loss import Loss
from model import NestedUNet
from mon import albumentation as A

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def _train_epoch(train_loader, model, criterion, optimizer, device):
    avg_meters = AverageMeter()
    model.train()
    pbar = tqdm(total=len(train_loader))
    for input, target, meta in train_loader:
        input  = input.to(device)
        target = target.to(device)
        output = model(input)
        loss   = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()  
        optimizer.step()
        avg_meters.update(loss.item(), input.size(0))
        pbar.set_postfix(loss=avg_meters.avg)
        pbar.update(1)
    pbar.close()
    return avg_meters.avg


def _val_epoch(val_loader, model, criterion, device):
    avg_meters = AverageMeter()
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, meta in val_loader:
            input  = input.to(device)
            target = target.to(device)
            output = model(input)
            loss   = criterion(output, target)
            avg_meters.update(loss.item(), input.size(0))
            pbar.set_postfix(loss=avg_meters.avg)
            pbar.update(1)
        pbar.close()

    return avg_meters.avg


def train(args: argparse.Namespace):
    weights      = args.weights
    weights      = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data         = mon.Path(args.data)
    save_dir     = mon.Path(args.save_dir)
    device       = args.device
    imgsz        = args.imgsz
    epochs       = args.epochs
    batch_size   = args.batch_size
    lr           = args.lr
    loss_weights = args.loss_weights
    verbose      = args.verbose
    weights_dir  = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    writer    = SummaryWriter(log_dir=str(save_dir / "tensorboard/loss"))
    criterion = Loss(*loss_weights)
    criterion = criterion.to(device)
    cudnn.benchmark = True
    
    model     = NestedUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    
    data_args = {
        "name": data,
        "root": mon.DATA_DIR / "llie",
        "transform" : A.Compose(transforms=[
            A.Resize(width=imgsz, height=imgsz),
        ]),
        "to_tensor" : True,
        "cache_data": False,
        "batch_size": batch_size,
        "devices"   : device,
        "shuffle"   : True,
        "verbose"   : verbose,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=data_args)
    datamodule.prepare_data()
    datamodule.setup(phase="training")
    
    log = OrderedDict([
        ("epoch"   , []),
        ("lr"      , []),
        ("loss"    , []),
        ("val_loss", []),
    ])
    best_loss = 1000

    for epoch in range(epochs):
        console.log("Epoch [%d/%d]" % (epoch, epochs))
        train_log = _train_epoch(datamodule.train_dataloader, model, criterion, optimizer, device)
        val_log   = _val_epoch(datamodule.val_dataloader, model, criterion, device)
        scheduler.step()
        console.log("loss %.4f  - val_loss %.4f" % (train_log, val_log))
        
        # Log
        log["epoch"].append(epoch)
        log["lr"].append(lr)
        log["loss"].append(train_log)
        log["val_loss"].append(val_log)
        pd.DataFrame(log).to_csv(str(save_dir / "log.csv"))
        writer.add_scalars("loss", {"train": train_log, "test": val_log}, epoch)
        writer.close()

        if val_log < best_loss or epoch > 290:
            torch.save(model.state_dict(), str(weights_dir/"best.pt"))
            best_loss = val_log
        torch.save(model.state_dict(), str(weights_dir/"last.pt"))
        torch.cuda.empty_cache()
    
# endregion

    
# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/train/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--local-rank", type=int, default=-1,   help="DDP parameter, do not modify.")
@click.option("--epochs",     type=int, default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",      type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--exist-ok",   is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    fullname  : str,
    save_dir  : str,
    local_rank: int,
    device    : str,
    epochs    : int,
    steps     : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Parse arguments
    weights  = weights  or args.get("weights")
    project  = args.get("project")
    fullname = fullname or args.get("name")
    device   = device   or args.get("device")
    epochs   = epochs   or args.get("epochs")
    exist_ok = exist_ok or args.get("exist_ok")
    verbose  = verbose  or args.get("verbose")
    
    # Prioritize input args --> config file args
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    project  = root.name or project
    save_dir = save_dir  or root / "run" / "train" / fullname
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = model
    args["project"]    = project
    args["name"]       = fullname
    args["save_dir"]   = save_dir
    args["device"]     = device
    args["local_rank"] = local_rank
    args["epochs"]     = epochs
    args["steps"]      = steps
    args["exist_ok"]   = exist_ok
    args["verbose"]    = verbose
    args = argparse.Namespace(**args)
    
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(args.save_dir))
    
    train(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
