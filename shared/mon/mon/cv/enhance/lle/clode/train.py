#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements CLODE model training pipeline for low-light image enhancement.

References:
    - Paper: "Continuous Exposure Learning for Low-light Image Enhancement using
      Neural ODEs," ICLR 2025.
    - Code: https://github.com/dgjung0220/CLODE
"""

import box
import torch

import clode
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
    model = clode.CLODE(weights=pretrained)
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = mon.nn.Adam(model.parameters(), **args.optimizer)
    
    # Loss
    L_tv      = clode.L_tv().to(device)
    L_spa     = clode.L_spa().to(device)
    L_col     = clode.L_col().to(device)
    L_exp_val = clode.L_exp_value(16, args.loss.L_exp_mean).to(device)
    L_tv_w    = args.loss.L_tv_w
    L_spa_w   = args.loss.L_spa_w
    L_col_w   = args.loss.L_col_w
    L_exp_w   = args.loss.L_exp_w
    
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
                image     = datapoint["image"]
                image     = image.to(device)
                eval_time = torch.tensor([0, 3]).float().to(device)
                outputs   = model(image, eval_time)
                enhanced  = outputs["output"]
                A_map     = outputs["curve_map"]
                noise_map = outputs["noise_map"]
                
                l_param = L_tv_w  * torch.mean(A_map)
                l_col   = L_col_w * torch.mean(L_col(enhanced))
                l_spa   = L_spa_w * torch.mean(L_spa(enhanced, image))
                l_exp   = L_exp_w * torch.mean(L_exp_val(enhanced))
                l_noise = torch.mean(noise_map)
                loss    = l_spa + l_col + l_exp + l_param + l_noise
                
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
                    image     = datapoint["image"]
                    image     = image.to(device)
                    ref       = datapoint["ref"]
                    ref       = ref.to(device)
                    eval_time = torch.tensor([0, 3]).float().to(device)
                    outputs   = model(image, eval_time, inference=True)
                    enhanced  = outputs["output"]
                    mse       = ((enhanced - ref) ** 2).mean((2, 3))
                    psnr      = (1 / mse).log10().mean() * 10
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
