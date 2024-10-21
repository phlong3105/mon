#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy

import torch.optim
import torchvision

import mon
from models import create_model
from options.test_options import TestOptions

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Hard-code some parameters for test
    # opt                = TestOptions().parse()  # get test options
    opt                = argparse.Namespace(**args.opt)
    opt.num_threads    = 0       # test code only supports num_threads = 0
    opt.batch_size     = 1       # test code only supports batch_size  = 1
    opt.serial_batches = True    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip        = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id     = -1      # no visdom display; the test code saves the results to a HTML file.
    # opt.gpu_ids        = [args.device]  # set GPU ids
    
    # Model
    model = create_model(opt)    # create a model given opt.model and other options
    model.setup(weights, opt)    # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()
        
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(model),
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
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                image      = datapoint.get("image").to(device)
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                h0, w0     = mon.get_image_size(image)
                if resize:
                    image = mon.resize(image, imgsz)
                else:
                    image = mon.resize(image, divisible_by=32)
                    
                datapoint["A"]       = image
                datapoint["B"]       = image
                datapoint["A_paths"] = image_path
                datapoint["B_paths"] = image_path
                
                # Infer
                timer.tick()
                model.set_input(datapoint)
                model.test()
                timer.tock()
                
                # Post-process
                visuals = model.get_current_visuals()
                fake_B  = visuals.get("fake_B")
                
                h1, w1 = mon.get_image_size(fake_B)
                if h1 != h0 or w1 != w0:
                    fake_B = mon.resize(fake_B, (h0, w0))
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(fake_B, str(output_path))
                '''
                if save_debug:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / f"{rel_path.parent}_debug"
                    else:
                        output_path = save_dir / f"{rel_path.parent}_debug"
                    output_path.mkdir(parents=True, exist_ok=True)
                    # torchvision.utils.save_image(g_a, str(output_path / f"{image_path.stem}_g_a.jpg"))
                    # torchvision.utils.save_image(pre, str(output_path / f"{image_path.stem}_pre.jpg"))
                '''
                
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
