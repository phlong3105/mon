#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Zero-DiDCE model training pipeline for low-light image enhancement.

References:
    - Paper: "Rethinking Zero-DCE for Low-Light Image Enhancement,"
      Neural Processing Letters 2024.
    - Code: https://github.com/Wenhui-Luo/Zero-DiDCE
"""

import box
import torch

import mon
import zerodidce

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
    model = zerodidce.ZeroDiDCE(weights=pretrained)
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = mon.nn.Adam(model.parameters(), **args.optimizer)
    
    # Loss
    L_piece = zerodidce.PiecewiseNonReferenceLoss().to(device)
    
    # Data I/O
    args["train_dataloader"]["dataset"]["root"] = mon.data.parse_data_dir(args.root)
    args["val_dataloader"]["dataset"]["root"]   = mon.data.parse_data_dir(args.root)
    train_dataloader = mon.data.DataLoader(**args.train_dataloader)
    val_dataloader   = mon.data.DataLoader(**args.val_dataloader)

    # Train
    best_loss      = 9999
    best_psnr      = 0
    grad_clip_norm = args["trainer"]["grad_clip_norm"]
    with (mon.create_progress_bar() as pbar):
        for i in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow]Training"
        ):
            losses    = []
            val_psnrs = []
            model.train()
            for j, datapoint in enumerate(train_dataloader):
                image    = datapoint["image"]
                image    = image.to(device)
                outputs  = model(image)
                enhanced = outputs[-1]
                loss     = L_piece(enhanced, image)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            
            # Validation
            model.eval()
            for j, datapoint in enumerate(val_dataloader):
                with torch.no_grad():
                    image    = datapoint["image"]
                    image    = image.to(device)
                    ref      = datapoint["ref"]
                    ref      = ref.to(device)
                    outputs  = model(image)
                    enhanced = outputs[-1]
                    mse      = ((enhanced - ref) ** 2).mean((2, 3))
                    psnr     = (1 / mse).log10().mean() * 10
                val_psnrs.append(psnr.item())
            mean_psnr = sum(val_psnrs) / len(val_psnrs)
            
            # Log
            if args.verbose:  # and ((i + 1) % display_iter) == 0:
                mon.log(f"Epoch: {(i + 1):03} | Train Loss: {mean_loss:08.6f} | Val PSNR: {mean_psnr:08.6f}")
            
            # Save
            torch.save(model.state_dict(), args.save_dir / "last.pt")
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(model.state_dict(), args.save_dir / "best_loss.pt")
            if mean_psnr > best_psnr:
                best_psnr = mean_psnr
                torch.save(model.state_dict(), args.save_dir / "best_psnr.pt")


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(root=root_dir, model_root=root_dir)
    train(args)


if __name__ == "__main__":
    main()
