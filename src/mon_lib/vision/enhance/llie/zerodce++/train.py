#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os

import torch
import torch.optim

import dataloader
import model
import mon
import myloss
from mon import RUN_DIR, ZOO_DIR


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args.checkpoints_dir = mon.Path(args.checkpoints_dir)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    scale_factor = args.scale_factor
    DCE_net      = model.enhance_net_nopool(scale_factor).cuda()

    # DCE_net.apply(weights_init)
    if args.load_pretrain:
        DCE_net.load_state_dict(torch.load(args.weights))
    
    train_dataset = dataloader.lowlight_loader(args.input_dir)
    train_loader  = torch.utils.data.DataLoader(
	    train_dataset,
	    batch_size  = args.train_batch_size,
	    shuffle     = True,
	    num_workers = args.num_workers,
	    pin_memory  = True
    )
    
    L_color   = myloss.L_color()
    L_spa     = myloss.L_spa()
    L_exp     = myloss.L_exp(16)
    # L_exp = myloss.L_exp(16,0.6)
    L_tv      = myloss.L_TV()
    optimizer = torch.optim.Adam(
	    DCE_net.parameters(),
	    lr           = args.lr,
	    weight_decay = args.weight_decay
    )
    DCE_net.train()

    with mon.get_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow] Inferring"
        ):
            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight      = img_lowlight.cuda()
                enhanced_image, A = DCE_net(img_lowlight)
                
                # loss_tv = 200 * L_tv(A)
                loss_tv  = 1600 * L_tv(A)
                loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
                loss_col =  5 * torch.mean(L_color(enhanced_image))
                loss_exp = 10 * torch.mean(L_exp(enhanced_image, 0.6))
                loss     = loss_tv + loss_spa + loss_col + loss_exp
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(DCE_net.parameters(), args.grad_clip_norm)
                optimizer.step()
    
                if ((iteration + 1) % args.display_iter) == 0:
                    print("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % args.checkpoint_iter) == 0:
                    torch.save(DCE_net.state_dict(), args.checkpoints_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",        type=str,   default="data/train_data/")
    parser.add_argument("--weights",          type=str,   default=ZOO_DIR / "vision/enhance/llie/zerodce++/best.pth")
    parser.add_argument("--load-pretrain",    type=bool,  default=False)
    parser.add_argument("--lr",               type=float, default=0.0001)
    parser.add_argument("--weight-decay",     type=float, default=0.0001)
    parser.add_argument("--grad-clip-norm",   type=float, default=0.1)
    parser.add_argument("--scale-factor",     type=int,   default=1)
    parser.add_argument("--epochs",           type=int,   default=100)
    parser.add_argument("--train-batch-size", type=int,   default=8)
    parser.add_argument("--val-batch-size",   type=int,   default=8)
    parser.add_argument("--num-workers",      type=int,   default=4)
    parser.add_argument("--display-iter",     type=int,   default=10)
    parser.add_argument("--checkpoints-iter", type=int,   default=10)
    parser.add_argument("--checkpoints-dir",  type=str,   default=RUN_DIR / "train/vision/enhance/llie/zerodce++")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
