#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import socket

import click
import torch
import torch.optim

import dataloader
import model as mmodel
import myloss
from mon import core

console       = core.console
_current_file = core.Path(__file__).absolute()
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
    weights          = args.weights
    weights          = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    input_dir        = core.Path(args.train)
    save_dir         = core.Path(args.save_dir)
    device           = args.device
    epochs           = args.epochs
    scale_factor     = args.scale_factor
    train_batch_size = args.train_batch_size
    num_workers      = args.num_workers
    lr               = args.lr
    weight_decay     = args.weight_decay
    grad_clip_norm   = args.grad_clip_norm
    display_iter     = args.display_iter
    checkpoint_iter  = args.checkpoints_iter
    weights_dir      = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    DCE_net = mmodel.enhance_net_nopool(scale_factor).to(device)
    # DCE_net.apply(weights_init)
    if core.Path(weights).is_weights_file():
        DCE_net.load_state_dict(torch.load(weights))
    
    train_dataset = dataloader.lowlight_loader(input_dir)
    train_loader  = torch.utils.data.DataLoader(
	    train_dataset,
	    batch_size  = train_batch_size,
	    shuffle     = True,
	    num_workers = num_workers,
	    pin_memory  = True
    )
    
    L_color   = myloss.L_color()
    L_spa     = myloss.L_spa()
    L_exp     = myloss.L_exp(16)
    # L_exp = myloss.L_exp(16,0.6)
    L_tv      = myloss.L_TV()
    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=lr, weight_decay=weight_decay)
    DCE_net.train()

    with core.get_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Inferring"
        ):
            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight      = img_lowlight.to(device)
                enhanced_image, A = DCE_net(img_lowlight)
                
                # loss_tv = 200 * L_tv(A)
                loss_tv  = 1600 * L_tv(A)
                loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
                loss_col =  5 * torch.mean(L_color(enhanced_image))
                loss_exp = 10 * torch.mean(L_exp(enhanced_image, 0.6))
                loss     = loss_tv + loss_spa + loss_col + loss_exp
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(DCE_net.parameters(), grad_clip_norm)
                optimizer.step()
    
                if ((iteration + 1) % display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % checkpoint_iter) == 0:
                    torch.save(DCE_net.state_dict(), weights_dir / "best.pt")

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
    device    : str,
    local_rank: int,
    epochs    : int,
    steps     : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config = core.parse_config_file(project_root=_current_dir / "config", config=config)
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    root     = core.Path(root)
    weights  = weights   or args["weights"]
    project  = root.name or args["project"]
    fullname = fullname  or args["name"]
    save_dir = save_dir  or root / "run" / "train" / fullname
    save_dir = core.Path(save_dir)
    device   = device    or args["device"]
    device   = core.parse_device(device)
    epochs   = epochs    or args["epochs"]
    exist_ok = exist_ok  or args["exist_ok"]
    verbose  = verbose   or args["verbose"]
    
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
        core.delete_dir(paths=core.Path(args.save_dir))
    
    train(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion