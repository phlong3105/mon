#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    `<https://github.com/Fediory/HVI-CIDNet>`__
"""

from __future__ import annotations

import argparse
import copy
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import mon
from data.data import (
    get_eval_set, get_lol_v1_training_set, get_lol_v2_synthetic_training_set, get_lol_v2_training_set,
    get_sice_eval_set, get_sice_training_set, get_sid_training_set, get_training_set_blur,
)
from data.scheduler import CosineAnnealingRestartCyclicLR, CosineAnnealingRestartLR, GradualWarmupScheduler
from eval import eval
from loss.losses import EdgeLoss, L1Loss, PerceptualLoss, SSIM
from measure import metrics
from net.cidnet import CIDNet

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def train(args: argparse.Namespace):
    # General config
    save_dir = mon.Path(args.save_dir)
    weights  = args.weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    device   = mon.set_device(args.device)
    imgsz    = args.imgsz
    epochs   = args.epochs
    verbose  = args.verbose
    
    # Directory
    weights_dir = save_dir  # / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    debug_dir   = save_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Seed
    mon.set_random_seed(random.randint(1, 1000000))
    
    # Device
    cudnn.benchmark = True
    
    # Model
    model = CIDNet().to(device)
    if args.start_epoch > 0:
        model.load_state_dict(torch.load(str(weights), map_location=lambda storage, loc: storage))
    model.train()
    
    # Loss
    L1_weight = args.L1_weight
    D_weight  = args.D_weight
    E_weight  = args.E_weight
    P_weight  = 1.0
    
    L1_loss   = L1Loss(loss_weight=L1_weight, reduction="mean").to(device)
    D_loss    = SSIM(weight=D_weight).to(device)
    E_loss    = EdgeLoss(loss_weight=E_weight).to(device)
    P_loss    = PerceptualLoss(
        layer_weights     = {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
        perceptual_weight = P_weight,
        criterion         = "mse",
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))
    if args.cos_restart_cyclic:
        if args.start_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer       = optimizer,
                periods         = [(args.epochs // 4) - args.warmup_epochs, (args.epochs * 3) // 4],
                restart_weights = [1, 1],
                eta_mins        = [0.0002, 0.0000001]
            )
            scheduler = GradualWarmupScheduler(
                optimizer       = optimizer,
                multiplier      = 1,
                total_epoch     = args.warmup_epochs,
                after_scheduler = scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer       = optimizer,
                periods         = [args.epochs // 4, (args.epochs * 3) // 4],
                restart_weights = [1, 1],
                eta_mins        = [0.0002, 0.0000001]
            )
    elif args.cos_restart:
        if args.start_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer       = optimizer,
                periods         = [args.epochs - args.warmup_epochs - args.start_epoch],
                restart_weights = [1],
                eta_min         = 1e-7
            )
            scheduler = GradualWarmupScheduler(
                optimizer       = optimizer,
                multiplier      = 1,
                total_epoch     = args.warmup_epochs,
                after_scheduler = scheduler_step
            )
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer       = optimizer,
                periods         = [args.epochs - args.start_epoch],
                restart_weights = [1],
                eta_min         = 1e-7
            )
    else:
        raise Exception("Should choose a scheduler")
    
    # Data I/O
    if args.lol_v1 or args.lol_blur or args.lol_v2_real or args.lol_v2_synthetic or args.sid or args.sice_mix or args.sice_grad:
        args.data_train_lol_blur         = str(mon.DATA_DIR / args.data_train_lol_blur)
        args.data_train_lol_v1           = str(mon.DATA_DIR / args.data_train_lol_v1)
        args.data_train_lol_v2_real      = str(mon.DATA_DIR / args.data_train_lol_v2_real)
        args.data_train_lol_v2_synthetic = str(mon.DATA_DIR / args.data_train_lol_v2_synthetic)
        args.data_train_sid              = str(mon.DATA_DIR / args.data_train_sid)
        args.data_train_sice             = str(mon.DATA_DIR / args.data_train_sice)
        
        args.data_val_lol_blur           = str(mon.DATA_DIR / args.data_val_lol_blur)
        args.data_val_lol_v1             = str(mon.DATA_DIR / args.data_val_lol_v1)
        args.data_val_lol_v2_real        = str(mon.DATA_DIR / args.data_val_lol_v2_real)
        args.data_val_lol_v2_synthetic   = str(mon.DATA_DIR / args.data_val_lol_v2_synthetic)
        args.data_val_sid                = str(mon.DATA_DIR / args.data_val_sid)
        args.data_val_sice               = str(mon.DATA_DIR / args.data_val_sice)
        
        args.data_valgt_lol_blur         = str(mon.DATA_DIR / args.data_valgt_lol_blur)
        args.data_valgt_lol_v1           = str(mon.DATA_DIR / args.data_valgt_lol_v1)
        args.data_valgt_lol_v2_real      = str(mon.DATA_DIR / args.data_valgt_lol_v2_real)
        args.data_valgt_lol_v2_synthetic = str(mon.DATA_DIR / args.data_valgt_lol_v2_synthetic)
        args.data_valgt_sid              = str(mon.DATA_DIR / args.data_valgt_sid)
        args.data_valgt_sice             = str(mon.DATA_DIR / args.data_valgt_sice)
        
        if args.lol_v1:
            train_set            = get_lol_v1_training_set(args.data_train_lol_v1, size=args.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
            test_set             = get_eval_set(args.data_val_lol_v1)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)
            output_folder        = "lol_v1"
            label_dir            = args.data_valgt_lol_v1
            norm_size            = True
        elif args.lol_blur:
            train_set            = get_training_set_blur(args.data_train_lol_blur, size=args.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
            test_set             = get_eval_set(args.data_val_lol_blur)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)
            output_folder        = "lol_blur"
            label_dir            = args.data_valgt_lol_blur
            norm_size            = True
        elif args.lol_v2_real:
            train_set            = get_lol_v2_training_set(args.data_train_lol_v2_real, size=args.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
            test_set             = get_eval_set(args.data_val_lol_v2_real)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)
            output_folder        = "lol_v2_real"
            label_dir            = args.data_valgt_lol_v2_real
            norm_size            = True
        elif args.lol_v2_synthetic:
            train_set            = get_lol_v2_synthetic_training_set(args.data_train_lol_v2_synthetic, size=args.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
            test_set             = get_eval_set(args.data_val_lol_v2_synthetic)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)
            output_folder        = "lol_v2_synthetic"
            label_dir            = args.data_valgt_lol_v2_synthetic
        elif args.sid:
            train_set            = get_sid_training_set(args.data_train_sid, size=args.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
            test_set             = get_eval_set(args.data_val_sid)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)
            output_folder        = "sid"
            label_dir            = args.data_valgt_sid
            norm_size            = True
            npy                  = True
        elif args.sice_mix:
            train_set            = get_sice_training_set(args.data_train_sice, size=args.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
            test_set             = get_sice_eval_set(args.data_val_sice_mix)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)
            output_folder        = "sice_mix"
            label_dir            = args.data_valgt_sice_mix
            norm_size            = False
        elif args.sice_grad:
            train_set            = get_sice_training_set(args.data_train_sice, size=args.crop_size)
            training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)
            test_set             = get_sice_eval_set(args.data_val_sice_grad)
            testing_data_loader  = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=1, shuffle=False)
            output_folder        = "sice_grad"
            label_dir            = args.data_valgt_sice_grad
            norm_size            = False
        else:
            raise Exception("Should choose a dataset")
    else:
        raise Exception("Should choose a dataset")
    
    # Logging
    writer     = SummaryWriter(log_dir=str(save_dir))
    psnr       = []
    ssim       = []
    lpips      = []
    loss_print = 0
    pic_cnt    = 0
    best_psnr  = 0
    best_ssim  = 0
    best_lpips = 100
    
    # Training
    start_epoch = args.start_epoch if args.start_epoch > 0 else 0
    for epoch in range(start_epoch + 1, args.epochs + start_epoch + 1):
        model.train()
        loss_last_10 = 0
        pic_last_10  = 0
        train_len    = len(training_data_loader)
        iter         = 0
        torch.autograd.set_detect_anomaly(True)
        
        # Train epoch
        with mon.get_progress_bar() as pbar:
            for batch in pbar.track(
                sequence    = training_data_loader,
                total       = len(training_data_loader),
                description = f"[bright_yellow] Training"
            ):
                # Forward
                im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
                im1         = im1.to(device)
                im2         = im2.to(device)
                output_rgb  = model(im1)
                gt_rgb      = im2
                output_hvi  = model.HVIT(output_rgb)
                gt_hvi      = model.HVIT(gt_rgb)
                loss_hvi    = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + args.P_weight * P_loss(output_hvi, gt_hvi)[0]
                loss_rgb    = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + args.P_weight * P_loss(output_rgb, gt_rgb)[0]
                loss        = loss_rgb + args.HVI_weight * loss_hvi
                iter       += 1
                
                # Optimizer
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_print    = loss_print   + loss.item()
                loss_last_10  = loss_last_10 + loss.item()
                pic_cnt      += 1
                pic_last_10  += 1
                if iter == train_len:
                    output_img = transforms.ToPILImage()(output_rgb[0].squeeze(0))
                    gt_img     = transforms.ToPILImage()(gt_rgb[0].squeeze(0))
                    if not os.path.exists(str(debug_dir)):
                        os.mkdir(str(debug_dir))
                    output_img.save(str(debug_dir) + "/" + "test.png")
                    gt_img.save(str(debug_dir) + "/" + "gt.png")
        
        scheduler.step()
        
        # Eval
        im_dir = str(debug_dir / output_folder / "*.png")
        eval(
            model               = copy.deepcopy(model),
            testing_data_loader = testing_data_loader,
            model_path          = None,
            output_folder       = str(debug_dir / output_folder),
            norm_size           = norm_size,
            lol_v1              = args.lol_v1,
            lol_v2              = args.lol_v2_real or args.data_train_lol_v2_synthetic,
            alpha               = 0.8
        )
        avg_loss = loss_last_10 / pic_last_10
        avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_gt_mean=False)
        
        # Log
        print("===> Epoch[{}] ".format(epoch))
        print("===> Loss: {:.4f} ".format(avg_loss))
        print("===> Learning rate: {:.8f}".format(optimizer.param_groups[0]["lr"]))
        print("===> PSNR: {:.4f} ".format(avg_psnr))
        print("===> SSIM: {:.4f} ".format(avg_ssim))
        print("===> LPIPS: {:.4f} ".format(avg_lpips))
        print()
        writer.add_scalar("train/loss", avg_loss , epoch)
        writer.add_scalar("val/psnr"  , avg_psnr , epoch)
        writer.add_scalar("val/ssim"  , avg_ssim , epoch)
        writer.add_scalar("val/lpips" , avg_lpips, epoch)
        psnr.append(avg_psnr)
        ssim.append(avg_ssim)
        lpips.append(avg_lpips)
    
        # Save
        if best_psnr < avg_psnr:
            torch.save(model.state_dict(), str(weights_dir / "best_psnr.pt"))
            best_psnr = avg_psnr
        if best_ssim < avg_ssim:
            torch.save(model.state_dict(), str(weights_dir / "best_ssim.pt"))
            best_ssim = avg_ssim
        if best_lpips > avg_lpips:
            torch.save(model.state_dict(), str(weights_dir / "best_lpips.pt"))
            best_lpips = avg_lpips
        torch.save(model.state_dict(), str(weights_dir / "last.pt"))
        torch.cuda.empty_cache()
    
# endregion


# region Main

def main() -> str:
    args = mon.parse_train_args(model_root=_current_dir)
    train(args)


if __name__ == "__main__":
    main()
    
# endregion
