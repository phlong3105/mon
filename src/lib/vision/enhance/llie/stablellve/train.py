#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/zkawfanx/StableLLVE

from __future__ import annotations

import argparse
import os
import random
import socket
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mon
from dataloader import llenDataset
from model import UNet
from mon import DATA_DIR, RUN_DIR
from warp import WarpingLayerBWFlow

console = mon.console


def save_checkpoint(state, epoch, output_directory):
    checkpoint_filename = os.path.join(output_directory, "checkpoint-" + str(epoch) + ".pth")
    torch.save(state, checkpoint_filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-light enhancement")
    parser.add_argument("--input-dir",      type=str,   default=DATA_DIR, help="path to the dataset")
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--batch-size",     type=int,   default=1,        help="[train] batch size(default: 1)")
    parser.add_argument("--lr",             type=float, default=1e-4,     help="learning rate (default: 1e-4)")
    parser.add_argument("--weight",         type=float, default=20,       help="weight of consistency loss")
    parser.add_argument("--gpu",            type=str,   default="0",      help="GPU id to use (default: 0)")
    parser.add_argument("--log-dir",        type=str,   default=RUN_DIR / "train/vision/enhance/llie/stablellve", help="folder to log")
    parser.add_argument("--checkpoint-dir", type=str,   default=RUN_DIR / "train/vision/enhance/llie/stablellve", help="path to checkpoint")
    args = parser.parse_args()
    return args


def train():
    args = parse_args()

    args.checkpoints_dir = mon.Path(args.checkpoints_dir)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train_set    = llenDataset(args.data, type="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    torch.manual_seed(ord("c") + 137)
    random.seed(ord("c") + 137)
    np.random.seed(ord("c") + 137)
    
    start_epoch = 0
    model       = UNet(n_channels=3, bilinear=True).cuda()
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    criterion = nn.L1Loss()
    warp = WarpingLayerBWFlow().cuda()
    
    # Create logger
    if args.log_dir is None:
        log_dir = os.path.join(os.path.abspath(os.getcwd()), "logs", datetime.now().strftime("%b%d_%H-%M-%S_") + socket.gethostname())
    else:
        log_dir = os.path.join(os.path.abspath(os.getcwd()), "logs", args.log_dir)
    
    os.makedirs(log_dir)
    logger = SummaryWriter(log_dir)
    
    # Log arguments
    with open(os.path.join(log_dir, "config.txt"), "a") as f:
        print(args, file=f)
    
    iters = 0

    with mon.get_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(start_epoch, args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow] Inferring"
        ):
            # Log learning rate
            for i, param_group in enumerate(optimizer.param_groups):
                logger.add_scalar("Lr/lr_" + str(i), float(param_group["lr"]), epoch)

            # Training stage
            console.log("Epoch", epoch, "train in progress…")
            model.train()

            for i, (input, target, flow) in enumerate(train_loader):
                input, target, flow= input.cuda(), target.cuda(), flow.cuda()

                # the 1st pass
                pred = model(input)
                loss = criterion(pred, target)

                # the 2nd pass
                input_t      = warp(input, flow)
                input_t_pred = model(input_t)
                pred_t       = warp(pred, flow)

                loss_t       = criterion(input_t_pred, pred_t)
                total_loss   = loss + loss_t * args.weight

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                logger.add_scalar("Train/Loss", loss.item(), iters)
                logger.add_scalar("Train/Loss_t", loss_t.item(), iters)
                iters += 1

                if (i + 1) % 10 == 0:
                    console.log(
                        "Train Epoch: {0} [{1}/{2}]\t"
                        "l1Loss={Loss1:.8f} "
                        "conLoss={Loss2:.8f} ".format(
                        epoch, i + 1, len(train_loader), Loss1=loss.item(), Loss2=loss_t.item())
                    )

            save_checkpoint(model.state_dict(), epoch, str(args.checkpoints_dir))

    logger.close()


if __name__ == "__main__":
    train()
