#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements HVI-CIDNet model training pipeline for low-light image enhancement.

References:
    - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
    - Code: https://github.com/Fediory/HVI-CIDNet
"""

import random

import box
import torch
import torch.backends.cudnn as cudnn
import torchvision

import cidnet
import mon

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Train -----
def train(args: dict | box.Box) -> str:
    gamma              = args.network.gamma
    start_gamma        = args.network.start_gamma
    end_gamma          = args.network.end_gamma
    lr                 = args.optimizer.lr
    cos_restart_cyclic = args.optimizer.cos_restart_cyclic
    cos_restart        = args.optimizer.cos_restart
    HVI_weight         = args.loss.HVI_weight
    L1_weight          = args.loss.L1_weight
    D_weight           = args.loss.D_weight
    E_weight           = args.loss.E_weight
    P_weight           = args.loss.P_weight
    start_epoch        = 0
    warmup_epochs      = args.trainer.warmup_epochs
    start_warmup       = args.trainer.start_warmup
    grad_detect        = args.trainer.grad_detect
    grad_clip          = args.trainer.grad_clip

    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)
    cudnn.benchmark = True

    # Seed
    mon.set_random_seed(args.seed)

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
    model = cidnet.HVI_CIDNet(weights=pretrained)
    # if pretrained and pretrained.is_weights_file():
    #     model.load_state_dict(torch.load(pretrained, map_location=lambda storage, loc: storage))
    model = model.to(device)

    # Optimizer
    optimizer = mon.nn.Adam(model.parameters(), lr=lr)
    if cos_restart_cyclic:
        if start_warmup:
            scheduler_step = cidnet.CosineAnnealingRestartCyclicLR(
                optimizer       = optimizer,
                periods         = [(args.epochs // 4) - warmup_epochs, (args.epochs * 3) // 4],
                restart_weights = [1, 1],
                eta_mins        = [0.0002, 0.0000001]
            )
            scheduler = cidnet.GradualWarmupScheduler(
                optimizer,
                multiplier      = 1,
                total_epoch     = warmup_epochs,
                after_scheduler = scheduler_step
            )
        else:
            scheduler = cidnet.CosineAnnealingRestartCyclicLR(
                optimizer       = optimizer,
                periods         = [args.epochs // 4, (args.epochs * 3) // 4],
                restart_weights = [1, 1],
                eta_mins        = [0.0002, 0.0000001]
            )
    elif cos_restart:
        if start_warmup:
            scheduler_step = cidnet.CosineAnnealingRestartLR(
                optimizer       = optimizer,
                periods         = [args.epochs - warmup_epochs - start_epoch],
                restart_weights = [1],
                eta_min         = 1e-7
            )
            scheduler = cidnet.GradualWarmupScheduler(
                optimizer,
                multiplier      = 1,
                total_epoch     = warmup_epochs,
                after_scheduler = scheduler_step
            )
        else:
            scheduler = cidnet.CosineAnnealingRestartLR(
                optimizer       = optimizer,
                periods         = [args.epochs - start_epoch],
                restart_weights = [1],
                eta_min         = 1e-7
            )
    else:
        raise Exception("Should choose a scheduler.")
    
    # Loss
    L1_loss = cidnet.L1Loss(loss_weight=L1_weight, reduction="mean").to(device)
    D_loss  = cidnet.SSIM(weight=D_weight).to(device)
    E_loss  = cidnet.EdgeLoss(loss_weight=E_weight).to(device)
    P_loss  = cidnet.PerceptualLoss(
        {"conv1_2": 1, "conv2_2": 1, "conv3_4": 1, "conv4_4": 1},
        perceptual_weight = P_weight,
        criterion         = "mse"
    ).to(device)
    
    # Data I/O
    args["train_dataloader"]["dataset"]["root"] = mon.data.parse_data_dir(args.root)
    args["val_dataloader"]["dataset"]["root"]   = mon.data.parse_data_dir(args.root)
    train_dataloader = mon.data.DataLoader(**args.train_dataloader)
    val_dataloader   = mon.data.DataLoader(**args.val_dataloader)

    # Training
    with mon.create_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow]Training"
        ):
            model.train()
            loss_print   = 0
            loss_last_10 = 0
            pic_cnt      = 0
            pic_last_10  = 0
            train_len    = len(train_dataloader)
            torch.autograd.set_detect_anomaly(grad_detect)
            for j, datapoint in enumerate(train_dataloader):
                # Input
                image   = datapoint["image"].to(device)
                ref_rgb = datapoint["ref_image"].to(device)
                
                # Enhance
                if gamma:  # Use random gamma function (enhancement curve) to improve generalization
                    gamma = random.randint(start_gamma, end_gamma) / 100.0
                    enhanced_rgb = model(image ** gamma)
                else:
                    enhanced_rgb = model(image)
                enhanced_hvi = model.HVIT(enhanced_rgb)
                ref_hvi      = model.HVIT(ref_rgb)
                
                # Loss
                loss_hvi = (
                    L1_loss(enhanced_hvi, ref_hvi)
                    + D_loss(enhanced_hvi, ref_hvi)
                    + E_loss(enhanced_hvi, ref_hvi)
                    + P_weight * P_loss(enhanced_hvi, ref_hvi)[0]
                )
                loss_rgb = (
                    L1_loss(enhanced_rgb, ref_rgb)
                    + D_loss(enhanced_rgb, ref_rgb)
                    + E_loss(enhanced_rgb, ref_rgb)
                    + P_weight * P_loss(enhanced_rgb, ref_rgb)[0]
                )
                loss  = loss_rgb + HVI_weight * loss_hvi
                
                # Backward
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Log (debug)
                loss_print    = loss_print   + loss.item()
                loss_last_10  = loss_last_10 + loss.item()
                pic_cnt      += 1
                pic_last_10  += 1
                if j == train_len:
                    enhanced_img = torchvision.transforms.ToPILImage()(enhanced_rgb[0].squeeze(0))
                    ref_img      = torchvision.transforms.ToPILImage()(ref_rgb[0].squeeze(0))
                    (args.save_dir / mon.SAVE_DEBUG_DIR).mkdir(parents=True, exist_ok=True)
                    enhanced_img.save(str(args.save_dir / mon.SAVE_DEBUG_DIR / "enhanced.jpg"))
                    ref_img.save(str(args.save_dir / mon.SAVE_DEBUG_DIR / "ref.jpg"))
            
            scheduler.step()
            
            # Log
            avg_loss = loss_last_10 / pic_last_10
            mon.log(f"===> Epoch[{i}]: Loss: {avg_loss:.4f} | "
                            f"Learning rate: lr={optimizer.param_groups[0]['lr']}.")
            
            # Save the latest model
            torch.save(model.state_dict(), str(args.save_dir / "last.pt"))
            

# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(root=root_dir, model_root=root_dir)
    train(args)


if __name__ == "__main__":
    main()
