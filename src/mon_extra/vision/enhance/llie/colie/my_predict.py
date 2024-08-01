#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import torch.optim

import mon
from color import hsv2rgb_torch, rgb2hsv_torch
from loss import *
from siren import INF
from utils import *

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data      = args.data
    save_dir  = args.save_dir
    weights   = args.weights
    device    = mon.set_device(args.device)
    epochs    = args.epochs
    imgsz     = args.imgsz[0]
    resize    = args.resize
    benchmark = args.benchmark
    window    = int(args.window)
    L         = float(args.L)
    alpha     = float(args.alpha)
    beta      = float(args.beta)
    gamma     = float(args.gamma)
    delta     = float(args.delta)
    # print(window, L, alpha, beta, gamma, delta)
    
    # Benchmark
    if benchmark:
        model = INF(patch_dim=window ** 2, num_layers=4, hidden_dim=256, add_layer=2)
        flops, params, avg_time = mon.calculate_efficiency_score(
            model      = model,
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=data, dst=save_dir, denormalize=True)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    timer = mon.Timer()
    with mon.get_progress_bar() as pbar:
        for images, target, meta in pbar.track(
            sequence    = data_loader,
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Model
            img_siren  = INF(patch_dim=window ** 2, num_layers=4, hidden_dim=256, add_layer=2)
            img_siren  = img_siren.to(device)
            # Optimizer
            optimizer  = torch.optim.Adam(img_siren.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)
            # Loss Functions
            l_exp      = L_exp(16, L)
            l_TV       = L_TV()
            # Input
            image_path = meta["path"]
            img_rgb    = get_image(str(image_path))
            # h0, w0     = img_rgb.shape[0], img_rgb.shape[1]
            # img_rgb    = mon.resize(img_rgb, (imgsz, imgsz))
            img_hsv    = rgb2hsv_torch(img_rgb)
            img_v      = get_v_component(img_hsv)
            img_v_lr   = interpolate_image(img_v, imgsz, imgsz)
            coords     = get_coords(imgsz, imgsz)
            patches    = get_patches(img_v_lr, window)
            # Training
            timer.tick()
            for epoch in range(epochs):
                img_siren.train()
                optimizer.zero_grad()
                #
                illu_res_lr    = img_siren(patches, coords)
                illu_res_lr    = illu_res_lr.view(1, 1, imgsz, imgsz)
                illu_lr        = illu_res_lr + img_v_lr
                img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)
                #
                loss_spa      = torch.mean(torch.abs(torch.pow(illu_lr - img_v_lr, 2))) * alpha
                loss_tv       = l_TV(illu_lr) * beta
                loss_exp      = torch.mean(l_exp(illu_lr)) * gamma
                loss_sparsity = torch.mean(img_v_fixed_lr) * delta
                loss          = loss_spa * alpha + loss_tv * beta + loss_exp * gamma + loss_sparsity * delta  # ???
                loss.backward()
                optimizer.step()
            img_v_fixed   = filter_up(img_v_lr, img_v_fixed_lr, img_v)
            img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
            img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
            img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)
            # img_rgb_fixed = mon.resize(img_rgb_fixed, (h0, w0))
            timer.tock()
            output_path   = save_dir / image_path.name
            Image.fromarray(
                (torch.movedim(img_rgb_fixed, 1, -1)[0].detach().cpu().numpy() * 255).astype(np.uint8)
            ).save(str(output_path))
    # avg_time = float(timer.total_time / len(data_loader))
    avg_time   = float(timer.avg_time)
    console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
