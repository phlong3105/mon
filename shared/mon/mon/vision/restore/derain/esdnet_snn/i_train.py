#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import time

import box
import torch.optim
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims
import utils
from dataset_load import Dataload
from losses import *
from model import model
from spikingjelly.activation_based import functional

os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark     = True

# A workaround for the bug in numpy >= 1.2.4
np.int   = np.int32
np.float = np.float64
np.bool  = np.bool_

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Train -----
def train(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data I/O
    data_root     = mon.parse_data_dir(args.root, data_dir=args.datamodule.root)
    train_dir     = data_root / "train"
    train_dataset = Dataload(data_dir=train_dir, patch_size=args.datamodule.patch_size_train)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.datamodule.batch_size,
        shuffle     = args.datamodule.shuffle,
        num_workers = 4,
        drop_last   = False,
        pin_memory  = True
    )

    if (data_root / "val").exists():
        val_dir = data_root / "val"
    elif (data_root / "test").exists():
        val_dir = data_root / "test"
    elif args.data in ["rain13k"]:
        val_dir = mon.ROOT_DIR / "data" / "enhance" / "rain100" / "test"
    else:
        raise ValueError("No validation dataset found.")
    val_dataset = Dataload(data_dir=val_dir, patch_size=args.datamodule.patch_size_test)
    val_loader  = torch.utils.data.DataLoader(
        val_dataset,
        batch_size  = args.datamodule.batch_size,
        shuffle     = False,
        num_workers = 1,
        drop_last   = False,
        pin_memory  = True
    )

    # Pretrained
    pretrained = args.tuning
    if args.resume and args.resume.is_weights_file(exist=True):
        pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        mon.log(f"Pretrained: {None}, training from scratch.")

    # Model
    model_ = model
    model_.to(device)
    start_epoch      = 0
    optim_state_dict = None
    if pretrained and Path(pretrained).is_weights_file():
        state_dict = torch.load(pretrained, map_location=device, weights_only=True)
        if pretrained.is_ckpt_file():
            state_dict       = state_dict["state_dict"]
            start_epoch      = state_dict["epoch"]
            optim_state_dict = state_dict["optimizer"]
        model_.load_state_dict(state_dict)
    functional.set_step_mode(model_, step_mode="m")
    functional.set_backend(model_,   backend="cupy")
    
    # Loss
    # criterion = nn.MSELoss().to(device)
    criterion = utils.SSIM().to(device)
    # criterion = nn.SmoothL1Loss().to(device)
    # criterion = PSNRLoss().to(device)

    # Optimizer
    optimizer        = optim.AdamW(model_.parameters(), lr=args.optimizer.lr, eps=1e-8)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs - args.optimizer.warmup_epochs, eta_min=args.optimizer.min_lr)
    scheduler        = mon.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.optimizer.warmup_epochs, after_scheduler=scheduler_cosine)
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
        
    # Train
    writer          = SummaryWriter(args.save_dir)
    scaler          = torch.cuda.amp.GradScaler()
    best_psnr       = 0
    best_ssim       = 0
    best_psnr_epoch = 0
    best_ssim_epoch = 0
    iter            = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        epoch_loss       = 0
        scaled_loss      = 0
        train_psnrs      = []
        model_.train()
        
        # Train
        with mon.create_progress_bar() as pbar:
            for i, data in pbar.track(
                sequence    = enumerate(train_loader),
                total       = len(train_loader),
                description = f"[bright_yellow]Training"
            ):
                for param in model_.parameters():
                    param.grad = None
                image    = data[0].to(device)
                ref      = data[1].to(device)
                enhanced = model_(image)
                if args.trainer.use_amp:
                    with torch.cuda.amp.autocast():
                        train_ssim = criterion(enhanced, ref)
                        loss       = 1 - train_ssim
                    scaler.scale(loss).backward()
                    # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    functional.reset_net(model_)
                else:
                    train_ssim = criterion(enhanced, ref)
                    loss       = 1 - train_ssim
                    loss.backward()
                    scaled_loss += loss.item()
                    # torch.nn.utils.clip_grad_norm_(model_restoration.parameters(), clip_grad)
                    optimizer.step()
                    functional.reset_net(model_)
                torch.cuda.synchronize()
                epoch_loss += loss.item()
                iter       += 1
                for res, tar in zip(enhanced, ref):
                    train_psnrs.append(utils.torchPSNR(res, tar))
                train_psnr = torch.stack(train_psnrs).mean().item()
                train_ssim = train_ssim.item()
                
                writer.add_scalar("loss/iter_loss",  loss.item(), iter)
                writer.add_scalar("loss/epoch_loss", epoch_loss, epoch)
                writer.add_scalar("lr/epoch_loss",   scheduler.get_lr()[0], epoch)
                
            # Val
            if epoch % 1 == 0:
                model_.eval()
                val_psnrs = []
                for ii, data_val in enumerate(val_loader):
                    image = data_val[0].to(device)
                    ref   = data_val[1].to(device)
                    
                    with torch.no_grad():
                        enhanced = model_(image)
                    functional.reset_net(model_)
                    
                    for res, tar in zip(enhanced, ref):
                        val_psnrs.append(utils.torchPSNR(res, tar))

                val_psnr = torch.stack(val_psnrs).mean().item()
                val_ssim = criterion(enhanced, ref).item()
                writer.add_scalar("val/psnr", val_psnr, epoch)
                writer.add_scalar("val/ssim", val_ssim, epoch)
                if val_psnr > best_psnr:
                    best_psnr       = val_psnr
                    best_psnr_epoch = epoch
                    torch.save(model_.state_dict(), str(args.save_dir / f"{args.fullname}_best_psnr.pt"))
                if val_ssim > best_ssim:
                    best_ssim       = val_ssim
                    best_ssim_epoch = epoch
                    torch.save(model_.state_dict(), str(args.save_dir / f"{args.fullname}_best_ssim.pt"))
                print("[Epoch %d Validating PSNR: %2.4f --- best_psnr_epoch %d Test_PSNR %2.4f]" % (epoch, val_psnr, best_psnr_epoch, best_psnr))
                print("[Epoch %d Validating SSIM: %2.4f --- best_ssim_epoch %d Test_SSIM %2.4f]" % (epoch, val_ssim, best_ssim_epoch, best_ssim))
            
            # Save
            torch.save(
                {
                    "epoch"     : epoch,
                    "state_dict": model_.state_dict(),
                    "optimizer" : optimizer.state_dict()
                },
                str(args.save_dir / f"{args.fullname}_last.ckpt")
            )
            torch.save(model_.state_dict(), str(args.save_dir / f"{args.fullname}_last.pt"))
            scheduler.step()
            print("-" * 150)
            print(
                "Epoch: {}\t"
                "Time: {:.4f}\t"
                "Loss: {:.4f}\t"
                "Train PSNR: {:.4f}\t"
                "Train SSIM: {:.4f}\t"
                "Learning Rate: {:.8f}\t"
                "Validate PSNR: {:.4f}\t"
                "Validate SSIM: {:.4f}".format(
                    epoch,
                    time.time() - epoch_start_time,
                    loss.item(),
                    train_psnr,
                    train_ssim,
                    scheduler.get_lr()[0],
                    val_psnr,
                    val_ssim,
                )
            )
            print("-" * 150)
    writer.close()
        

# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(root=root_dir, model_root=root_dir)
    train(args)


if __name__ == "__main__":
    main()
