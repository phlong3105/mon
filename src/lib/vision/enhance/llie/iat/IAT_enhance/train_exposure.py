#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_enhance

from __future__ import annotations

import argparse
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from IQA_pytorch import SSIM
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import vgg16

import mon
from data_loaders.exposure import exposure_loader
from model.IAT_main import IAT
from mon import RUN_DIR, ZOO_DIR
from utils import get_dist_info, LossNetwork, PSNR, validation

console = mon.console

console.log(torch.cuda.device_count())
dist.init_process_group(backend="nccl")


def train(args: argparse.Namespace):
    # Distribute Setting
    torch.cuda.set_device(args.local_rank)
    args.checkpoints_dir = mon.Path(args.checkpoints_dir)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Seed
    seed = random.randint(1, 10000)
    # print("Random seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Model Setting
    model = IAT(type="exp").cuda()
    if args.load_pretrain is not None:
        model.load_state_dict(torch.load(args.weights))
    
    # Data Setting
    train_dataset = exposure_loader(images_path=str(args.data_train), normalize=args.normalize)
    train_sampler = DistributedSampler(train_dataset, drop_last=True, seed=seed)
    train_loader  = DataLoader(
        train_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = 8,
        pin_memory  = True,
        sampler     = train_sampler
    )
    val_dataset = exposure_loader(
        images_path = str(args.data_val),
        mode        = "val",
        normalize   = args.normalize
    )
    val_loader  = DataLoader(
        val_dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 8,
        pin_memory  = True
    )
    
    # Loss & Optimizer Setting & Metric
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.cuda()
    
    for param in vgg_model.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam([{"params": model.global_net.parameters(),"lr":config.lr*0.1},
    #             {"params": model.local_net.parameters(),"lr":config.lr}], lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    device = next(model.parameters()).device
    console.log("The device is:", device)
    
    L1_loss        = nn.L1Loss()
    L1_smooth_loss = F.smooth_l1_loss
    
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    
    ssim      = SSIM()
    psnr      = PSNR()
    ssim_high = 0
    psnr_high = 0
    rank, _   = get_dist_info()
    
    model.train()
    console.log("######## Start IAT Training #########")
    with mon.get_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow] Inferring"
        ):
            # adjust_learning_rate(optimizer, epoch)
            train_sampler.set_epoch(epoch)
            for iteration, images in enumerate(train_loader):
                low_image, high_image = images[0].cuda(), images[1].cuda()
                # Checking!
                # visualization(low_img,  "show/low",  iteration)
                # visualization(high_img, "show/high", iteration)
                optimizer.zero_grad()
                model.train()
                mul, add, enhance_image = model(low_image)
            
                loss = L1_loss(enhance_image, high_image)
                # loss = L1_smooth_loss(enhance_img, high_img)+0.04*loss_network(enhance_img, high_img)
                loss.backward()
                optimizer.step()
                scheduler.step()
        
                if ((iteration + 1) % args.display_iter) == 0:
                    console.log("Loss at iteration", iteration + 1, ":", loss.item())
    
        # Evaluation Model
        if rank == 0:
            model.eval()
            PSNR_mean, SSIM_mean = validation(model, val_loader)
            
            with open(str(args.checkpoints_dir / "log.txt"), "a+") as f:
                f.write("epoch" + str(epoch) + ":" + "the SSIM is" + str(SSIM_mean) + "the PSNR is" + str(PSNR_mean) + "\n")
            f.close()
            
            if SSIM_mean > ssim_high:
                ssim_high = SSIM_mean
                console.log("The highest SSIM value is:", str(ssim_high))
                torch.save(model.state_dict(), args.checkpoints_dir / "best.pt")
    
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-train",       type=str,   default="/data/unagi0/cui_data/light_dataset/Exposure_CVPR21/train/INPUT_IMAGES")
    parser.add_argument("--data-val",         type=str,   default="/data/unagi0/cui_data/light_dataset/Exposure_CVPR21/validation/INPUT_IMAGES")
    parser.add_argument("--weights",          type=str,   default=ZOO_DIR / "vision/enhance/llie/iat/iat-exposure.pth")
    parser.add_argument("--load-pretrain",    type=bool,  default=False)
    parser.add_argument("--batch-size",       type=int,   default=8)
    parser.add_argument("--lr",               type=float, default=2e-4)   # for batch size 4x2=8
    parser.add_argument("--weight-decay",     type=float, default=0.0004)
    parser.add_argument("--epochs",           type=int,   default=200)
    parser.add_argument("--local-rank",       type=int,   default=-1, help="")
    parser.add_argument("--normalize",        action="store_true", help="Default not Normalize in exposure training.")
    parser.add_argument("--display-iter",     type=int,   default=10)
    parser.add_argument("--checkpoints-iter", type=int,   default=10)
    parser.add_argument("--checkpoints-dir",  type=str,   default=RUN_DIR / "train/vision/enhance/llie/iat/exposure")
    args = parser.parse_args()
    train(args)
