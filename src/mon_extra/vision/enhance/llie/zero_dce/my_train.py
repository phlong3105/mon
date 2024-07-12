#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import socket

import click
import torch
import torch.optim

import dataloader
import model as mmodel
import mon
import myloss

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


def train(args: argparse.Namespace):
    # General config
    save_dir = mon.Path(args.save_dir)
    weights  = args.weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    device   = mon.set_device(args.device)
    epochs   = args.epochs
    verbose  = args.verbose
    
    # Directory
    weights_dir = save_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    DCE_net = mmodel.enhance_net_nopool().to(device)
    DCE_net.apply(weights_init)
    if mon.Path(weights).is_weights_file():
        DCE_net.load_state_dict(torch.load(weights))
    DCE_net.train()
    
    # Loss
    L_color = myloss.L_color()
    L_spa   = myloss.L_spa()
    L_exp   = myloss.L_exp(16 , 0.6)
    L_tv    = myloss.L_TV()
    
    # Optimizer
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Data I/O
    train_dataset = dataloader.lowlight_loader(args.data)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.train_batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True
    )
    
    # Training
    with mon.get_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Predicting"
        ):
            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight = img_lowlight.to(device)
                enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
                
                loss_tv  = 200 * L_tv(A)
                loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
                loss_col = 5   * torch.mean(L_color(enhanced_image))
                loss_exp = 10  * torch.mean(L_exp(enhanced_image))
                loss     = loss_tv + loss_spa + loss_col + loss_exp
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(DCE_net.parameters(), args.grad_clip_norm)
                optimizer.step()
                
                if ((iteration + 1) % args.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % args.checkpoints_iter) == 0:
                    torch.save(DCE_net.state_dict(), weights_dir / "best.pt")

# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--arch",       type=str, default=None, help="Model architecture.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--project",    type=str, default=None, help="Project name.")
@click.option("--variant",    type=str, default=None, help="Variant name.")
@click.option("--fullname",   type=str, default=None, help="Fullname to save the model's weight.")
@click.option("--save-dir",   type=str, default=None, help="Save results to root/run/train/arch/model/data or root/run/train/arch/project/variant.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--local-rank", type=int, default=-1,   help="DDP parameter, do not modify.")
@click.option("--epochs",     type=int, default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",      type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--exist-ok",   is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    config    : str,
    arch      : str,
    model     : str,
    root      : str,
    project   : str,
    variant   : str,
    fullname  : str,
    save_dir  : str,
    weights   : str,
    device    : str,
    local_rank: int,
    epochs    : int,
    steps     : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args   = mon.load_config(config)
    
    # Parse arguments
    model    = model    or args.get("model")
    data     =             args.get("data")
    root     = root     or args.get("root")
    project  = project  or args.get("project")
    variant  = variant  or args.get("variant")
    fullname = fullname or args.get("fullname")
    save_dir = save_dir or args.get("save_dir")
    weights  = weights  or args.get("weights")
    device   = device   or args.get("device")
    epochs   = epochs   or args.get("epochs")
    steps    = steps    or args.get("steps")
    exist_ok = exist_ok or args.get("exist_ok")
    verbose  = verbose  or args.get("verbose")
    
    # Prioritize input args --> config file args
    root     = mon.Path(root)
    save_dir = save_dir or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    save_dir = mon.Path(save_dir)
    weights  = mon.to_list(weights)
    device   = mon.parse_device(device)
    
    # Update arguments
    args["hostname"]   = hostname
    args["config"]     = config
    args["opt"]        = config
    args["arch"]       = arch
    args["model"]      = model
    args["root"]       = root
    args["project"]    = project
    args["variant"]    = variant
    args["fullname"]   = fullname
    args["save_dir"]   = save_dir
    args["weights"]    = weights
    args["device"]     = device
    args["epochs"]     = epochs
    args["local_rank"] = local_rank
    args["steps"]      = steps
    args["exist_ok"]   = exist_ok
    args["verbose"]    = verbose
    args = argparse.Namespace(**args)
    
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(args.save_dir))
        
    train(args)
    return str(args.save_dir)


if __name__ == "__main__":
    args = mon.parse_predict_args()
    train(args)

# endregion
