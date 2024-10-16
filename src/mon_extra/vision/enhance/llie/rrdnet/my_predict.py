#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy

import torch
import torch.optim
import torch.optim as optim
import torchvision

import conf
import mon
from loss.loss_functions import (
    illumination_smooth_loss, noise_loss,
    reconstruction_loss, reflectance_smooth_loss,
)
from model.RRDNet import RRDNet

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data           = args.data
    save_dir       = args.save_dir
    weights        = args.weights
    device         = mon.set_device(args.device)
    epochs         = args.epochs
    imgsz          = args.imgsz
    resize         = args.resize
    benchmark      = args.benchmark
    save_image     = args.save_image
    save_debug     = args.save_debug
    use_fullpath   = args.use_fullpath
    # Model specific
    lr             = args.lr
    illu_factor    = args.illu_factor
    reflect_factor = args.reflect_factor
    noise_factor   = args.noise_factor
    reffac         = args.reffac
    gamma          = args.gamma
    g_kernel_size  = args.g_kernel_size
    g_padding      = args.g_padding
    sigma          = args.sigma
    
    # Model
    net = RRDNet()
    net = net.to(device)
    
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(net),
            image_size = imgsz,
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with mon.get_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Input
            image      = datapoint.get("image")
            image      = image.to(device)
            meta       = datapoint.get("meta")
            image_path = mon.Path(meta["path"])
            
            for j in range(epochs + 1):
                # forward
                illumination, reflectance, noise = net(image)  # [1, c, h, w]
                # loss computing
                loss_recons  = reconstruction_loss(image, illumination, reflectance, noise)
                loss_illu    = illumination_smooth_loss(image, illumination)
                loss_reflect = reflectance_smooth_loss(image, illumination, reflectance)
                loss_noise   = noise_loss(image, illumination, reflectance, noise)
                loss         = loss_recons + conf.illu_factor * loss_illu + conf.reflect_factor * loss_reflect + conf.noise_factor * loss_noise
                # backward
                net.zero_grad()
                loss.backward()
                optimizer.step()
                # log
                if j % 100 == 0:
                    print("iter:", j, '  reconstruction loss:', float(loss_recons.data), '  illumination loss:', float(loss_illu.data), '  reflectance loss:', float(loss_reflect.data), '  noise loss:', float(loss_noise.data))
            
            # Infer
            timer.tick()
            adjust_illu = torch.pow(illumination, conf.gamma)
            res_image   = adjust_illu * ((image - noise) / illumination)
            res_image   = torch.clamp(res_image, min=0, max=1)
            timer.tock()
            
            # Save
            if save_image:
                if use_fullpath:
                    rel_path    = image_path.relative_path(data_name)
                    output_path = save_dir / rel_path.parent / image_path.name
                else:
                    output_path = save_dir / data_name / image_path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(res_image, str(output_path))
            if save_debug:
                if use_fullpath:
                    rel_path    = image_path.relative_path(data_name)
                    output_path = save_dir / f"{rel_path.parent}_debug"
                else:
                    output_path = save_dir / f"{rel_path.parent}_debug"
                output_path.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(illumination, str(output_path / f"{image_path.stem}_illumination.jpg"))
                torchvision.utils.save_image(adjust_illu,  str(output_path / f"{image_path.stem}_adjust_illumination.jpg"))
                torchvision.utils.save_image(reflectance,  str(output_path / f"{image_path.stem}_reflectance.jpg"))
                torchvision.utils.save_image(noise,        str(output_path / f"{image_path.stem}_noise.jpg"))
    
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
