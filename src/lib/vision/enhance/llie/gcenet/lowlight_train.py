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
import Myloss
console = mon.console


def weights_init(m):
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    '''
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "conv"):
            m.conv.weight.data.normal_(0.0, 0.02)
        elif hasattr(m, "dw_conv"):
            m.dw_conv.weight.data.normal_(0.0, 0.02)
        elif hasattr(m, "pw_conv"):
            m.pw_conv.weight.data.normal_(0.0, 0.02)
        else:
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    DCE_net = model.enhance_net_nopool(
        variant = args.variant,
    ).cuda()
    DCE_net.apply(weights_init)
    # if args.load_pretrain:
    #     DCE_net.load_state_dict(torch.load(args.weights))
   
    train_dataset = dataloader.lowlight_loader(args.data)
    train_loader  = torch.utils.data.DataLoader(
	    train_dataset,
	    batch_size  = args.train_batch_size,
	    shuffle     = True,
	    num_workers = args.num_workers,
	    pin_memory  = True
    )

    L_col     = Myloss.L_color()
    L_edge    = mon.EdgeConstancyLoss()
    L_exp     = Myloss.L_exp(16, 0.6)
    L_kl      = mon.ChannelConsistencyLoss()
    L_spa     = Myloss.L_spa()
    L_tvA     = Myloss.L_TV()
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
            description = f"[bright_yellow] Training"
        ):
            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight = img_lowlight.cuda()
                enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
                
                loss_tv   = 200 * L_tvA(A)
                loss_spa  = 1   * torch.mean(L_spa(enhanced_image, img_lowlight))
                loss_col  = 5   * torch.mean(L_col(enhanced_image))
                loss_exp  = 10  * torch.mean(L_exp(enhanced_image))
                loss_edge = 1   * L_edge(input=enhanced_image, target=img_lowlight)
                loss_kl   = 0.1 * L_kl(input=enhanced_image, target=img_lowlight)
                loss      = loss_tv + loss_spa + loss_col + loss_exp + loss_edge + loss_kl
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(DCE_net.parameters(), args.grad_clip_norm)
                optimizer.step()
    
                if ((iteration + 1) % args.display_iter) == 0:
                    console.log("Loss at iteration", iteration + 1, ":", loss.item())
                if ((iteration + 1) % args.checkpoints_iter) == 0:
                    torch.save(DCE_net.state_dict(), args.checkpoints_dir / "best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",             type=str,   default="data/train_data/")
    parser.add_argument("--weights",          type=str,   default="weights/Epoch99.pth")
    parser.add_argument("--load-pretrain",    type=bool,  default=False)
    parser.add_argument("--project",          type=str,   default=None)
    parser.add_argument("--name",             type=str,   default=None)
    parser.add_argument("--variant",          type=str,   default="00000")
    parser.add_argument("--lr",               type=float, default=0.0001)
    parser.add_argument("--weight-decay",     type=float, default=0.0001)
    parser.add_argument("--grad-clip-norm",   type=float, default=0.1)
    parser.add_argument("--epochs",           type=int,   default=200)
    parser.add_argument("--train-batch-size", type=int,   default=8)
    parser.add_argument("--val-batch-size",   type=int,   default=4)
    parser.add_argument("--num-workers",      type=int,   default=4)
    parser.add_argument("--display-iter",     type=int,   default=10)
    parser.add_argument("--checkpoints-iter", type=int,   default=10)
    parser.add_argument("--checkpoints-dir",  type=str,   default=mon.RUN_DIR/"train")
    args = parser.parse_args()
	   
    args.checkpoints_dir = mon.Path(args.checkpoints_dir)
    # if args.project is not None and args.project != "":
    #     args.checkpoints_dir /= args.project
    # if args.name is not None and args.name != "":
    #    args.checkpoints_dir /= args.name
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    console.log(args.checkpoints_dir)
    
    train(args)
