#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_enhance

from __future__ import annotations

import argparse
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from IQA_pytorch import SSIM
from torchvision.models import vgg16

import mon
from data_loaders.lol_v1_new import lowlight_loader_new
from model.IAT_main import IAT
from mon import ZOO_DIR, RUN_DIR
from utils import LossNetwork, PSNR, validation

console = mon.console


def train(args: argparse.Namespace):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.checkpoints_dir = mon.Path(args.checkpoints_dir)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Model Setting
    model = IAT().cuda()
    if args.load_pretrain is not None:
        model.load_state_dict(torch.load(args.weights))
    
    # Data Setting
    train_dataset = lowlight_loader_new(images_path=str(args.data_train))
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = 8,
        pin_memory  = True
    )
    val_dataset = lowlight_loader_new(images_path=str(args.data_val), mode="test")
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 8,
        pin_memory  = True
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = args.lr,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
        weight_decay = args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    device = next(model.parameters()).device
    console.log("The device is:", device)
    
    # Loss & Optimizer Setting & Metric
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()
    
    for param in vgg_model.parameters():
        param.requires_grad = False
    
    # L1_loss = CharbonnierLoss()
    L1_loss        = nn.L1Loss()
    L1_smooth_loss = F.smooth_l1_loss
    
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    
    ssim      = SSIM()
    psnr      = PSNR()
    ssim_high = 0
    psnr_high = 0
    
    model.train()
    console.log("######## Start IAT Training #########")
    with mon.get_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow] Inferring"
        ):
            # adjust_learning_rate(optimizer, epoch)
            for iteration, images in enumerate(train_loader):
                low_image, high_image = images[0].cuda(), images[1].cuda()
                # Checking!
                # visualization(low_img,  "show/low",  iteration)
                # visualization(high_img, "show/high", iteration)
                optimizer.zero_grad()
                model.train()
                mul, add, enhance_img = model(low_image)
        
                loss = L1_loss(enhance_img, high_image)
                # loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img)
                loss.backward()
                optimizer.step()
                scheduler.step()
        
                if ((iteration + 1) % args.display_iter) == 0:
                    console.log("Loss at iteration", iteration + 1, ":", loss.item())
    
        # Evaluation Model
        model.eval()
        PSNR_mean, SSIM_mean = validation(model, val_loader)
    
        with open(str(args.checkpoints_dir / "log.txt"), "a+") as f:
            f.write("epoch" + str(epoch) + ":" + "the SSIM is" + str(SSIM_mean) + "the PSNR is" + str(PSNR_mean) + "\n")
        f.close()
        
        if SSIM_mean > ssim_high:
            ssim_high = SSIM_mean
            console.log("The highest SSIM value is:", str(ssim_high))
            torch.save(model.state_dict(), args.checkpoints_dir / "best.pt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-train",       type=str,   default="/data/unagi0/cui_data/light_dataset/LOL_v1/our485_patch/low/")
    parser.add_argument("--data-val",         type=str,   default="/data/unagi0/cui_data/light_dataset/LOL_v1/eval15/low/")
    parser.add_argument("--weights",          type=str,   default=ZOO_DIR / "vision/enhance/llie/iat/iat-lol-v1.pth")
    parser.add_argument("--load-pretrain",    type=bool,  default=False)
    parser.add_argument("--batch-size",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=2e-4)   # for batch size 4x2=8
    parser.add_argument("--weight-decay",     type=float, default=0.0004)
    parser.add_argument("--epochs",           type=int,   default=200)
    parser.add_argument("--local-rank",       type=int,   default=-1, help="")
    parser.add_argument("--normalize",        action="store_true", help="Default not Normalize in exposure training.")
    parser.add_argument("--gpu",              type=str,   default=0)
    parser.add_argument("--display-iter",     type=int,   default=10)
    parser.add_argument("--checkpoints-iter", type=int,   default=10)
    parser.add_argument("--checkpoints-dir",  type=str,   default=RUN_DIR / "train/vision/enhance/llie/iat/lol_v1_patch")
    args = parser.parse_args()
    train(args)
