#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import socket
import time

import click
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.optim
from torch.utils.tensorboard import SummaryWriter

import dataloader
import mon
import myloss
from model import Image_network

console       = mon.console
_current_file = mon.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train

def get_hist(filename):
    src    = cv2.imread(filename)
    src    = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    hist_s = np.zeros((3, 256))

    for (j, color) in enumerate(("red", "green", "blue")):
        s = src[..., j]
        hist_s[j, ...], _ = np.histogram(s.flatten(), 256, [0, 256])
        hist_s[j, ...] = hist_s[j, ...] / np.sum(hist_s[j, ...])

    hist_s = torch.from_numpy(hist_s).float()

    return hist_s


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            # print(param)
            param.requires_grad = False
            # print(param)
        dfs_freeze(child)
        

def eval(val_loader, model, device):
    psnr_meters = mon.PeakSignalNoiseRatio().to(device)
    ssim_meters = mon.StructuralSimilarityIndexMeasure().to(device)
    model.eval()
    for iteration, (low, gt, hist) in enumerate(val_loader):
        low  = low.to(device)
        gt   = gt.to(device)
        hist = hist.to(device)
        img, tf, w, _ = model(low, hist)
        psnr_meters.update(img, gt)
        ssim_meters.update(img, gt)
    avg_psnr = float(psnr_meters.compute().cpu().detach().numpy())
    avg_ssim = float(ssim_meters.compute().cpu().detach().numpy())
    return avg_psnr, avg_ssim


def train(args: argparse.Namespace):
    # General config
    weights  = args.weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    save_dir = mon.Path(args.save_dir)
    device   = args.device
    epochs   = args.epochs
    verbose  = args.verbose
    
    # Directory
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if torch.cuda.is_available():
        cudnn.benchmark = True
    else:
        raise Exception("No GPU found, please run without --cuda")
    device = device[0] if isinstance(device, list) else device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{device}"
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # change allocation of current GPU
    
    # Model
    Imgnet = Image_network()
    Imgnet.apply(weights_init)
    Imgnet = Imgnet.to(device)
    Imgnet.train()
    
    num_params = 0
    for param in Imgnet.parameters():
        num_params += param.numel()
    console.log("# of Imgnet params : %d" % num_params)
    
    # Loss
    loss_c = torch.nn.MSELoss().to(device)
    loss_e = myloss.entropy_loss().to(device)
    cos    = torch.nn.CosineSimilarity(dim=1)
    loss_t = myloss.totalvariation_loss().to(device)
    
    cont_c        = 0.5
    cont_e        = 0.2
    cont_cs       = 0.3

    lambda_c      = 0
    lambda_e      = 0
    lambda_cs     = 0

    difficulty_c  = 0
    difficulty_e  = 0
    difficulty_cs = 0

    loss_col_0    = 0
    loss_ent_0    = 0
    loss_cos_0    = 0
    loss_0        = 0

    loss_col_2    = 0
    loss_ent_2    = 0
    loss_cos_2    = 0
    loss_2        = 0
    
    # Optimizer
    optimizer_img = torch.optim.Adam(Imgnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Data I/O
    train_dataset = dataloader.InputLoader(args.data, "train")
    train_loader  = torch.utils.data.DataLoader(
        dataset     = train_dataset,
        batch_size  = args.train_batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True
    )
    val_dataset = dataloader.InputLoader(args.data, "test")
    val_loader  = torch.utils.data.DataLoader(
        dataset     = val_dataset,
        batch_size  = args.val_batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True
    )
    
    # Logging
    writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
    
    # Training
    best_psnr = 0
    best_ssim = 0
    with mon.get_progress_bar() as pbar:
        for epoch in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Training"
        ):
            console.log("epoch :", epoch + 1)
            sum_loss_col = 0
            sum_loss_ent = 0
            sum_loss_cos = 0
            sum_loss_tv  = 0
            sum_loss     = 0
            sum_loss_    = 0
            
            for iteration, (low, gt, hist) in enumerate(train_loader):
                Imgnet.train()
                low  = low.to(device)
                gt   = gt.to(device)
                hist = hist.to(device)
                
                # Forward
                img, tf, w, _ = Imgnet(low, hist)
                img  = img.to(device)
                
                # Loss
                loss_img = loss_c(img, gt)
                loss_ent = loss_e(w)
                loss_col = torch.mean(1 - torch.abs(cos(gt, img)))
                loss_tv  = loss_t(w)
               
                if epoch == 0:
                    loss_f = (cont_c * loss_img + cont_e * loss_ent + loss_tv + cont_cs * loss_col)
                elif epoch == 1:
                    loss_f = (lambda_c * loss_img) + (lambda_e * loss_ent) + loss_tv + (lambda_cs * loss_col)
                else:
                    loss_f = (lambda_c * difficulty_c * loss_img) + (lambda_e * difficulty_e * loss_ent) + loss_tv + (lambda_cs * difficulty_cs * loss_col)
                loss_ = loss_f - loss_tv
                
                # Optimizer
                optimizer_img.zero_grad()
                loss_f.backward()
                torch.nn.utils.clip_grad_norm_(Imgnet.parameters(), args.grad_clip_norm)
                optimizer_img.step()
                
                # Log
                sum_loss_col += loss_img.item()
                sum_loss_ent += loss_ent.item()
                sum_loss_cos += loss_col.item()
                sum_loss_tv  += loss_tv.item()
                sum_loss     += loss_f.item()
                sum_loss_    += loss_.item()
                
                if iteration == (len(train_loader) - 1):
                    console.log("Total Loss:", loss_f.item())
                    loss_col_0 = sum_loss_col / len(train_loader)
                    loss_ent_0 = sum_loss_ent / len(train_loader)
                    loss_cos_0 = sum_loss_cos / len(train_loader)
                    loss_0     = sum_loss_    / len(train_loader)
                    writer.add_scalar("color_loss",                   sum_loss_col / len(train_loader), epoch + 1)
                    writer.add_scalar("transformation_function_loss", sum_loss_ent / len(train_loader), epoch + 1)
                    writer.add_scalar("cosine_similarity_loss",       sum_loss_cos / len(train_loader), epoch + 1)
                    writer.add_scalar("total_variation_loss",          sum_loss_tv / len(train_loader), epoch + 1)
                    writer.add_scalar("total_loss",                       sum_loss / len(train_loader), epoch + 1)
            
            if epoch == 0:
                loss_col_1 = loss_col_0
                loss_ent_1 = loss_ent_0
                loss_cos_1 = loss_cos_0
                loss_1     = loss_0
                # Get loss weights
                lambda_c   = cont_c  * (loss_1 / loss_col_1)
                lambda_e   = cont_e  * (loss_1 / loss_ent_1)
                lambda_cs  = cont_cs * (loss_1 / loss_cos_1)
                # print()
                # print("lambda_c\t" + str(lambda_c))
                # print("lambda_e\t" + str(lambda_e))
                # print("lambda_cs\t" + str(lambda_cs))
                # print()
                # Update previous losses
                loss_col_2 = loss_col_1
                loss_ent_2 = loss_ent_1
                loss_cos_2 = loss_cos_1
                loss_2     = loss_1
            else:
                loss_col_1 = loss_col_0
                loss_ent_1 = loss_ent_0
                loss_cos_1 = loss_cos_0
                loss_1     = loss_0
                # Get loss weights
                lambda_c   = cont_c * (loss_1 / loss_col_1)
                lambda_e   = cont_e * (loss_1 / loss_ent_1)
                lambda_cs  = cont_cs * (loss_1 / loss_cos_1)
                # print()
                # print("lambda_c\t" + str(lambda_c))
                # print("lambda_e\t" + str(lambda_e))
                # print("lambda_cs\t" + str(lambda_cs))
                # print()
                # Get difficulties
                difficulty_c  = ((loss_col_1 / loss_col_2) / (loss_1 / loss_2)) ** args.beta
                difficulty_e  = ((loss_ent_1 / loss_ent_2) / (loss_1 / loss_2)) ** args.beta
                difficulty_cs = ((loss_cos_1 / loss_cos_2) / (loss_1 / loss_2)) ** args.beta
                # print("difficulty_c\t"  + str(difficulty_c))
                # print("difficulty_e\t"  + str(difficulty_e))
                # print("difficulty_cs\t" + str(difficulty_cs))
                # print()
                # Update previous losses
                loss_col_2 = loss_col_1
                loss_ent_2 = loss_ent_1
                loss_cos_2 = loss_cos_1
                loss_2     = loss_1
            
            # Eval
            psnr, ssim = eval(val_loader, Imgnet, device)
            writer.add_scalar("psnr", psnr, epoch + 1)
            writer.add_scalar("ssim", ssim, epoch + 1)
            print()
            
            # Save
            if best_psnr < psnr:
                best_psnr = psnr
                torch.save(Imgnet.state_dict(), str(weights_dir / "best_psnr.pt"))
            if best_ssim < ssim:
                best_ssim = ssim
                torch.save(Imgnet.state_dict(), str(weights_dir / "best_ssim.pt"))
            torch.save(Imgnet.state_dict(), str(weights_dir / "last.pt"))
            
    writer.close()
    
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
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    epochs    : int,
    steps     : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=_current_dir / "config", config=config)
    args     = mon.load_config(config)
    
    # Parse arguments
    weights  = weights  or args.get("weights")
    fullname = fullname or args.get("fullname")
    device   = device   or args.get("device")
    epochs   = epochs   or args.get("epochs")
    exist_ok = exist_ok or args.get("exist_ok")
    verbose  = verbose  or args.get("verbose")
    
    # Prioritize input args --> config file args
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    save_dir = save_dir or root / "run" / "train" / fullname
    save_dir = mon.Path(save_dir)
    device   = mon.parse_device(device)
    
    # Update arguments
    args["root"]     = root
    args["config"]   = config
    args["weights"]  = weights
    args["model"]    = model
    args["fullname"] = fullname
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
