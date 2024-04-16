#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    `<https://github.com/zzyfd/STAR-pytorch/tree/main>`__
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import time

import click
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import models.model
import mon
from dataloaders.dataloader_fivek import EnhanceDataset_FiveK

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args: argparse.Namespace):
    # General config
    weights  = args.weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    save_dir = mon.Path(args.save_dir)
    device   = args.device
    imgsz    = args.imgsz
    epochs   = args.epochs
    verbose  = args.verbose
    
    # Directory
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    
    # Seed
    cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    # Data I/O
    train_dataset = EnhanceDataset_FiveK(
        images_path  = args.data,
        image_size   = args.image_h,
        is_yuv       = False,
        image_size_w = args.image_w,
    )
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.train_batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    
    # Model
    # DCE_net = models.build_model(args).to(device)
    DCE_net = models.model.enhance_net_litr().to(device)
    DCE_net.apply(weights_init)
    DCE_net.train()
    if args.load_pretrain:
        DCE_net.load_state_dict(torch.load(weights), strict=True)
    if args.parallel:
        DCE_net = nn.DataParallel(DCE_net)
    
    # Loss
    import losses
    L_L1    = nn.L1Loss().to(device) if not args.l2_loss else nn.MSELoss().to(device)
    L_color = losses.L_color().to(device)
    
    # Optimizer
    optimizer = optim.Adam(DCE_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_type == "cos":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    elif args.lr_type == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=epochs // 3 * len(train_loader))
    else:
        lr_scheduler = None
    
    # Logging
    writer    = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
    best_loss = 100
    
    # Training
    for epoch in range(epochs):
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for iteration, img_lowlight in pbar.track(
                sequence    = enumerate(train_loader),
                total       = len(train_loader),
                description = f"[bright_yellow] Training"
            ):
                input, target = img_lowlight
                input         = input.to(device)
                target        = target.to(device)
                img_input_ds  = F.interpolate(input, (args.image_ds, args.image_ds), mode="area")
                img_input_ds  = img_input_ds.to(device)
                torch.cuda.synchronize()
                
                # Forward
                start_time = time.time()
                enhanced_image, x_r = DCE_net(img_input_ds, img_in=input)
                torch.cuda.synchronize()
                run_time   = time.time() - start_time
                sum_time  += run_time
                
                # Loss
                loss_color = torch.mean(L_color(enhanced_image)) if args.color_loss else torch.zeros([]).to(device)
                loss_l1    = L_L1(enhanced_image, target)        if not args.no_l1  else torch.zeros([]).to(device)
                loss_cos   = 1 - nn.functional.cosine_similarity(enhanced_image, target, dim=1).mean() if args.cos_loss else torch.zeros([]).to(device)
                if args.mul_loss:
                    loss = loss_l1 * loss_cos
                else:
                    loss = loss_l1 + loss_color + 100 * loss_cos
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(DCE_net.parameters(), args.grad_clip_norm)
                optimizer.step()
                if args.lr_type != "fix":
                    lr_scheduler.step()
                
                # Log
                if iteration == 0:
                    A = x_r
                    n, _, h, w = A.shape
                    A = A.sub(A.view(n, _, -1).min(dim=-1)[0].view(n, _, 1, 1)).div(A.view(n, _, -1).max(dim=-1)[0].view(n, _, 1, 1) - A.view(n, _, -1).min(dim=-1)[0].view(n, _, 1, 1)).squeeze()
                    writer.add_image("input_enhanced_ref_residual", torch.cat([input[0], enhanced_image[0], target[0], torch.abs(enhanced_image[0] - target[0])] + [torch.stack((A[0, i], A[0, i], A[0, i]), 0) for i in range(A.shape[1])], 2), epoch)
    
                if (iteration % args.display_iter) == 0:
                    istep = epoch * len(train_loader) + iteration
                    console.log(
                        "Loss at iteration",
                        iteration, ":", loss_l1.item(),
                        # " | LR : ", optimizer.param_groups[0]["lr"],
                        # "Batch time: ", run_time,
                        # "Batch Time AVG: ", sum_time / (iteration + 1),
                        # "Cos loss: ", loss_cos.item()
                    )
                    writer.add_scalar("loss",       loss_l1,    istep)
                    writer.add_scalar("Color_loss", loss_color, istep)
                    writer.add_scalar("Cos_loss",   loss_cos,   istep)
                    writer.add_scalar("lr",         optimizer.param_groups[0]["lr"], istep)
                
                # Save
                if loss < best_loss:
                    best_loss = loss
                    torch.save(DCE_net.state_dict(), str(weights_dir/"best.pt"))
                torch.save(DCE_net.state_dict(), str(weights_dir/"last.pt"))

# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",     type=str, default=None, help="Project root.")
@click.option("--config",   type=str, default=None, help="Model config.")
@click.option("--weights",  type=str, default=None, help="Weights paths.")
@click.option("--model",    type=str, default=None, help="Model name.")
@click.option("--fullname", type=str, default=None, help="Save results to root/run/train/fullname.")
@click.option("--save-dir", type=str, default=None, help="Optional saving directory.")
@click.option("--device",   type=str, default=None, help="Running devices.")
@click.option("--epochs",   type=int, default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",    type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--exist-ok", is_flag=True)
@click.option("--verbose",  is_flag=True)
def main(
    root    : str,
    config  : str,
    weights : str,
    model   : str,
    fullname: str,
    save_dir: str,
    device  : str,
    epochs  : int,
    steps   : int,
    exist_ok: bool,
    verbose : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Parse arguments
    weights  = weights  or args.get("weights")
    project  = args.get("project")
    fullname = fullname or args.get("name")
    save_dir = save_dir or args.get("save_dir")
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
    args["root"]     = root
    args["config"]   = config
    args["weights"]  = weights
    args["model"]    = model
    args["project"]  = project
    args["name"]     = fullname
    args["save_dir"] = save_dir
    args["device"]   = device
    args["epochs"]   = epochs
    args["steps"]    = steps
    args["exist_ok"] = exist_ok
    args["verbose"]  = verbose
    args = argparse.Namespace(**args)
    
    if not exist_ok:
        mon.delete_dir(paths=mon.Path(args.save_dir))
    
    train(args)
    return str(args.save_dir)


if __name__ == "__main__":
    main()

# endregion
