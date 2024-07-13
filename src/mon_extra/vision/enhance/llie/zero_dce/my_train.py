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

def main() -> str:
    args = mon.parse_train_args(model_root=_current_dir)
    train(args)


if __name__ == "__main__":
    main()

# endregion
