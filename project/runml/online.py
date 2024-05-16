#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements online learning pipeline."""

from __future__ import annotations

import socket
import time

import click
import torch

import mon

console = mon.console


# region Online Learning

def online_learning(args: dict) -> str:
    # Get arguments
    fullname   = args["fullname"]
    imgsz      = args["image_size"]
    seed       = args["seed"]
    save_dir   = args["predictor"]["default_root_dir"]
    source     = args["predictor"]["source"]
    augment    = mon.to_list(args["predictor"]["augment"])
    devices    = args["predictor"]["devices"] or "auto"
    resize     = args["predictor"]["resize"]
    benchmark  = args["predictor"]["benchmark"]
    save_image = args["predictor"]["save_image"]
    
    # Initialization
    console.rule(f"[bold red] {fullname}")
    
    mon.set_random_seed(seed)
    
    devices          = torch.device(("cpu" if not torch.cuda.is_available() else devices))
    model: mon.Model = mon.MODELS.build(config=args["model"])
    model            = model.to(devices)
    model.eval()
    
    # Benchmark
    if benchmark and torch.cuda.is_available():
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
    console.log(f"[bold red] {source}")
    data_name, data_loader, data_writer = mon.parse_io_worker(src=source, dst=save_dir, denormalize=True)
    save_dir = save_dir if save_dir not in [None, "None", ""] else model.root
    save_dir = mon.Path(save_dir)
    save_dir = save_dir / data_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicting
    with torch.no_grad():
        sum_time = 0
        with mon.get_progress_bar() as pbar:
            for images, target, meta in pbar.track(
                sequence    = data_loader,
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                images = images.to(model.device)
                input  = images.clone()
                if resize:
                    h0, w0 = mon.get_image_size(images)
                    input  = mon.resize(input, imgsz)
                
                # Forward
                start_time = time.time()
                output     = model.fit_one(input=input)
                run_time   = time.time() - start_time
                
                # Post-process
                output = output[-1] if isinstance(output, list | tuple) else output
                if resize:
                    output = mon.resize(output, (h0, w0))
                
                # Save
                if save_image:
                    output_path = save_dir / f"{meta['stem']}.png"
                    mon.write_image(output_path, output, denormalize=True)
                    if data_writer is not None:
                        data_writer.write_batch(data=output)
                sum_time += run_time
        avg_time = float(sum_time / len(data_loader))
        console.log(f"Average time: {avg_time}")
        
        return str(save_dir)
        
# endregion


# region Main

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--data",       type=str, default=None, help="Source data directory.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--epochs",     type=int, default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",      type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--imgsz",      type=int, default=None, help="Image sizes.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    data      : str,
    fullname  : str,
    save_dir  : str,
    device    : int | list[int] | str,
    epochs    : int,
    steps     : int,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=root, config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    root     = root     or args["root"]
    weights  = weights  or args["model"]["weights"]
    data     = data     or args["predictor"]["source"]
    fullname = fullname or args["fullname"]
    save_dir = save_dir or args["predictor"]["default_root_dir"]
    device   = device   or args["predictor"]["devices"]
    epochs   = epochs if epochs > 0 else args["trainer"]["max_epochs"]
    steps    = steps  if steps  > 0 else args["trainer"]["max_steps"]
    imgsz    = imgsz    or args["image_size"]
    
    # Parse arguments
    root     = mon.Path(root)
    weights  = mon.to_list(weights)
    weights  = weights[0] if isinstance(weights, list | tuple) else weights
    save_dir = save_dir or root / "run" / "predict" / model
    save_dir = mon.Path(save_dir)
    
    # Update arguments
    args["hostname"]    = hostname
    args["config"]      = config
    args["root"]        = root
    args["fullname"]    = fullname
    args["image_size"]  = imgsz
    args["verbose"]     = verbose
    args["model"]      |= {
        "root"    : save_dir,
        "fullname": fullname,
        "weights" : weights,
        "verbose" : verbose,
    }
    args["trainer"]    |= {
        "default_root_dir": save_dir,
        "devices"         : device,
        "max_epochs"      : epochs if steps is not None else None,
        "max_steps"       : steps,
    }
    args["predictor"]  |= {
        "default_root_dir": save_dir,
        "source"          : data,
        "devices"         : device,
        "resize"          : resize,
        "benchmark"       : benchmark,
        "save_image"      : save_image,
        "verbose"         : verbose,
    }
    
    return online_learning(args=args)


if __name__ == "__main__":
    main()

# endregion
