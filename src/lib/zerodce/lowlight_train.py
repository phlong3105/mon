#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os

import torch
import torch.optim

import dataloader
import model
import Myloss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.apply(weights_init)
    if config.load_pretrain:
        DCE_net.load_state_dict(torch.load(config.weights))
   
    train_dataset = dataloader.lowlight_loader(config.data)
    train_loader  = torch.utils.data.DataLoader(
	    train_dataset,
	    batch_size  = config.train_batch_size,
	    shuffle     = True,
	    num_workers = config.num_workers,
	    pin_memory  = True
    )

    L_color   = Myloss.L_color()
    L_spa     = Myloss.L_spa()
    L_exp     = Myloss.L_exp(16, 0.6)
    L_tv      = Myloss.L_TV()
    optimizer = torch.optim.Adam(
	    DCE_net.parameters(),
	    lr           = config.lr,
	    weight_decay = config.weight_decay
    )
    DCE_net.train()

    for epoch in range(config.epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()
            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)

            loss_tv  = 200 * L_tv(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5  * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))
            loss     = loss_tv + loss_spa + loss_col + loss_exp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.checkpoint_iter) == 0:
                torch.save(
	                DCE_net.state_dict(),
	                config.checkpoints_dir + "best.pth"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument("--data",             type=str,   default="data/train_data/")
    parser.add_argument("--lr",               type=float, default=0.0001)
    parser.add_argument("--weight-decay",     type=float, default=0.0001)
    parser.add_argument("--grad-clip-norm",   type=float, default=0.1)
    parser.add_argument("--epochs",           type=int,   default=200)
    parser.add_argument("--train-batch-size", type=int,   default=8)
    parser.add_argument("--val-batch-size",   type=int,   default=4)
    parser.add_argument("--num-workers",      type=int,   default=4)
    parser.add_argument("--display-iter",     type=int,   default=10)
    parser.add_argument("--checkpoints-iter", type=int,   default=10)
    parser.add_argument("--checkpoints-dir",  type=str,   default="weights/")
    parser.add_argument("--weights",          type=str,   default="weights/Epoch99.pth")
    parser.add_argument("--load-pretrain",    type=bool,  default=False)
    config = parser.parse_args()
	
    if not os.path.exists(config.checkpoints_dir):
        os.mkdir(config.checkpoints_dir)

    train(config)
