#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import socket

import click
import torch

import mon
from model import DRIT
from mon import albumentation as A
from saver import Saver

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def train(args: argparse.Namespace):
    weights     = args.weights
    weights     = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data        = mon.Path(args.data)
    save_dir    = mon.Path(args.save_dir)
    device      = args.device
    imgsz       = args.imgsz
    verbose     = args.verbose
    batch_size  = args.batch_size
    
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Load model
    args.gpu = device
    model = DRIT(args)
    model.setgpu(args.gpu)
    if args.resume is None:
        model.initialize()
        ep0      = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(args.resume)
    model.set_scheduler(args, last_ep=ep0)
    ep0 += 1
    
    # Data I/O
    data_args = {
        "name"      : data,
        "root"      : mon.DATA_DIR / "llie",
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
    train_dataloader = datamodule.train_dataloader
    val_dataloader   = datamodule.val_dataloader
    
    # Saver for display and output
    args.display_dir = save_dir / "log"
    args.result_dir  = save_dir / "weights"
    saver = Saver(args)
    
    # Train
    max_it = 500000
    for ep in range(ep0, args.epochs):
        for it, (images_a, images_b, meta) in enumerate(train_dataloader):
            if images_a.size(0) != args.batch_size or images_b.size(0) != args.batch_size:
                continue
            
            # input data
            images_a = images_a.cuda(args.gpu).detach()
            images_b = images_b.cuda(args.gpu).detach()
            
            # update model
            if (it + 1) % args.d_iter != 0 and it < len(args.train_dataloader) - 2:
                model.update_D_content(images_a, images_b)
                continue
            else:
                model.update_D(images_a, images_b)
                model.update_EG()
            
            # save to display file
            if not args.no_display_img:
                saver.write_display(total_it, model)
            
            print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
            total_it += 1
            if total_it >= max_it:
                saver.write_img(-1, model)
                saver.write_model(-1, model)
                break
        
        # decay learning rate
        if args.n_epochs_decay > -1:
            model.update_lr()
        
        # save result image
        saver.write_img(ep, model)
        
        # Save network weights
        saver.write_model(ep, total_it, model)
    
    return

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
