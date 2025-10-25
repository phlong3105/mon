#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileIE model training pipeline for real-time image enhancement
on mobile devices.

References:
    - Paper: "MobileIE: An Extremely Lightweight and Effective ConvNet for
      Real-Time Image Enhancement on Mobile Devices," ICCV 2025.
    - Code: https://github.com/AVC2-UESTC/MobileIE
"""

import box
import torch

import mobileie
import mon

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Train -----
def train(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)
    
    # Device
    device = mon.create_device(args.device)
    
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
    model = mobileie.MobileIELLE(weights=pretrained, inference=False, **args.network)
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer    = mon.nn.Adam(model.parameters(), **args.optimizer)
    lr_scheduler = mon.nn.CosineAnnealingWarmRestarts(optimizer, 50, 2, 1e-7)
    
    # Loss
    lle_loss = mobileie.LLELoss(reduction="mean")
    
    # Data I/O
    args["train_dataloader"]["dataset"]["root"] = mon.data.parse_data_dir(args.root)
    args["val_dataloader"]["dataset"]["root"]   = mon.data.parse_data_dir(args.root)
    train_dataloader = mon.data.DataLoader(**args.train_dataloader)
    val_dataloader   = mon.data.DataLoader(**args.val_dataloader)
    
    # Train: Warming-up
    if args.trainer.warmup:
        warmup_epochs = args.trainer.warmup_epoch
        warmup_lr     = args.trainer.warmup_lr
        warmup_optim  = mon.nn.Adam(model.parameters(), lr=warmup_lr, weight_decay=0)
        warmup_loss   = L.WarmupLoss()
        mon.log(f"Warming-up for {warmup_epochs} epochs.")
        with (mon.create_progress_bar() as pbar):
            for i in pbar.track(
                sequence    = range(warmup_epochs),
                total       = warmup_epochs,
                description = f"[bright_yellow]Warming-up"
            ):
                loss_li = []
                for j, datapoint in enumerate(train_dataloader):
                    image = datapoint["image"].to(device)
                    ref   = datapoint["ref"].to(device)
                    warmup_out1, warmup_out2 = model.forward_warm(image)
                    loss  = warmup_loss(image, ref, warmup_out1, warmup_out2)
                    warmup_optim.zero_grad()
                    loss.backward()
                    warmup_optim.step()
                    loss_li.append(loss.item())
                mon.log(f"Epoch: {i + 1} | Loss: {sum(loss_li) / len(loss_li)}")
                torch.save(model.state_dict(), args.save_dir / "model_pre.pt")
            mon.log(f"Warming-up phase done.")
    
    # Train: Warming-up
    best_psnr  = 0
    save_every = args.trainer.save_every
    mon.log(f"Start training.")
    with (mon.create_progress_bar() as pbar):
        for i in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow]Training"
        ):
            loss_li  = []
            val_psnr = []
            model.train()
            for j, datapoint in enumerate(train_dataloader):
                image   = datapoint["image"].to(device)
                ref     = datapoint["ref"].to(device)
                outputs = model(image)
                loss    = lle_loss(outputs, ref)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_li.append(loss.item())
            lr_scheduler.step()
            mean_loss = sum(loss_li) / len(loss_li)
            
            # Validation
            model.eval()
            for j, datapoint in enumerate(val_dataloader):
                with torch.no_grad():
                    image   = datapoint["image"].to(device)
                    ref     = datapoint["ref"].to(device)
                    outputs = model(image)
                    mse     = ((outputs - ref) ** 2).mean((2, 3))
                    psnr    = (1 / mse).log10().mean() * 10
                val_psnr.append(psnr.item())
            mean_psnr = sum(val_psnr) / len(val_psnr)
            
            # Log
            if args.verbose:
                mon.log(f"Epoch: {(i + 1):03} | Train Loss: {mean_loss:.8f} | Val PSNR: {mean_psnr:.8f}")

            # Save
            if ((i + 1) % save_every) == 0:
                torch.save(model.state_dict(), args.save_dir / "last.pt")
            
            if mean_psnr > best_psnr:
                best_psnr  = mean_psnr
                model_slim = model.slim().to(device)
                torch.save(model.state_dict(),      args.save_dir / "best.pt")
                torch.save(model_slim.state_dict(), args.save_dir / "best_slim.pt")
            

# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(root=root_dir, model_root=root_dir)
    train(args)


if __name__ == "__main__":
    main()
