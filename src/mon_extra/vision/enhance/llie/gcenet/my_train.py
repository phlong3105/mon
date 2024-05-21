#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import socket
from collections import OrderedDict

import click
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import model as mmodel
import mon
import myloss
from mon import albumentation as A

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def val_epoch(val_loader, model, device):
    psnr_meters = mon.PeakSignalNoiseRatio().to(device)
    ssim_meters = mon.StructuralSimilarityIndexMeasure().to(device)
    model.eval()
    for input, target, meta in val_loader:
        input  = input.to(device)
        target = target.to(device)
        enhanced_image_1, enhanced_image, E = model(input)
        psnr_meters.update(enhanced_image, target)
        ssim_meters.update(enhanced_image, target)
    return psnr_meters.compute(), ssim_meters.compute()


def train(args: argparse.Namespace):
    # General config
    weights  = args.weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    save_dir = mon.Path(args.save_dir)
    device   = mon.set_device(args.device)
    imgsz    = args.imgsz
    epochs   = args.epochs
    verbose  = args.verbose
    
    # Directory
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    cudnn.benchmark = True
    
    # Model
    DCE_net = mmodel.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)
    if mon.Path(weights).is_weights_file():
        DCE_net.load_state_dict(torch.load(weights))
    DCE_net.train()
    
    # Loss
    L_color = myloss.L_color()
    L_spa   = myloss.L_spa()
    L_exp   = myloss.L_exp(16, 0.6)
    L_tv    = myloss.L_TV()
    
    # Optimizer
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Data I/O
    data_args = {
        "name"      : args.data,
        "root"      : mon.DATA_DIR / "llie",
        "transform" : A.Compose(transforms=[
            A.Resize(width=imgsz, height=imgsz),
        ]),
        "to_tensor" : True,
        "cache_data": False,
        "batch_size": args.train_batch_size,
        "devices"   : device,
        "shuffle"   : True,
        "verbose"   : verbose,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=data_args)
    datamodule.prepare_data()
    datamodule.setup(phase="training")
    train_dataloader = datamodule.train_dataloader
    val_dataloader   = datamodule.val_dataloader
    
    # Logging
    best_psnr = 0
    best_ssim = 0
    
    # Training
    with mon.get_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Training"
        ):
            for iteration, (input, target, meta) in enumerate(train_dataloader):
                input = input.to(device)
                enhanced_image_1, enhanced_image, E = DCE_net(input)
                
                loss_tv  = 200 * L_tv(E)
                loss_spa = torch.mean(L_spa(enhanced_image, input))
                loss_col =   5 * torch.mean(L_color(enhanced_image))
                loss_exp =  10 * torch.mean(L_exp(enhanced_image))
                loss     = loss_tv + loss_spa + loss_col + loss_exp
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(DCE_net.parameters(), args.grad_clip_norm)
                optimizer.step()
                
                if ((iteration + 1) % args.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % args.checkpoint_iter) == 0:
                    torch.save(DCE_net.state_dict(), weights_dir / "best.pt")
            
            # Validation
            val_results = val_epoch(val_dataloader, DCE_net, device)
            val_psnr    = float(val_results[0].cpu().detach().numpy())
            val_ssim    = float(val_results[1].cpu().detach().numpy())
            
            # Log
            console.log(
                "Epoch [%d/%d] val/psnr %.4f - val/ssim %.4f\n"
                % (epoch, epochs, val_psnr, val_ssim)
            )
            
            # Save
            if val_psnr > best_psnr:
                torch.save(DCE_net.state_dict(), str(weights_dir / "best_psnr.pt"))
                best_psnr = val_psnr
            if val_ssim > best_ssim:
                torch.save(DCE_net.state_dict(), str(weights_dir / "best_ssim.pt"))
                best_ssim = val_ssim
            torch.save(DCE_net.state_dict(), str(weights_dir / "last.pt"))
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
    fullname = fullname or args.get("fullname")
    device   = device   or args.get("device")
    epochs   = epochs   or args.get("epochs")
    exist_ok = exist_ok or args.get("exist_ok")
    verbose  = verbose  or args.get("verbose")
    
    # Prioritize input args --> config file args
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    save_dir = save_dir or root / "run" / "train" / fullname
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = model
    args["fullname"]   = fullname
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
